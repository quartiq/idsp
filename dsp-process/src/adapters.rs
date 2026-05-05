use crate::{SplitInplace, SplitProcess};

/// Adapt a scalar optional-input stage to chunk output mode.
///
/// The inner processor is called with `Some(x)` once and then `None` `N-1` times
/// to synthesize one output chunk from one input sample.
///
/// This is convenient for polyphase interpolators and other stages whose natural
/// scalar interface is `Option<X> -> Y`. Unlike [`crate::ChunkOut`], this
/// preserves stream phase and still runs the recursive inner stage once per
/// output sample.
///
/// See also [`Decimator`] for the inverse direction.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{FnSplitProcess, Interpolator, SplitProcess};
///
/// let proc = Interpolator(FnSplitProcess(|_: &mut (), x: Option<i32>| {
///     x.unwrap_or_default()
/// }));
/// let mut state = ();
/// assert_eq!(proc.process(&mut state, 7), [7, 0, 0]);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Interpolator<P>(pub P);
impl<X: Copy, Y, C: SplitProcess<Option<X>, Y, S>, S, const N: usize> SplitProcess<X, [Y; N], S>
    for Interpolator<C>
{
    fn process(&self, state: &mut S, x: X) -> [Y; N] {
        core::array::from_fn(|i| self.0.process(state, (i == 0).then_some(x)))
    }
}
impl<X: Copy, C, S> SplitInplace<X, S> for Interpolator<C> where Self: SplitProcess<X, X, S> {}

/// Scalar downsampler with explicit tick phase.
///
/// The first input sample produces `Some(x)`, then `rate` input samples produce
/// `None`, and the pattern repeats. This matches the phase convention used by
/// [`crate::Decimator`] when wrapping a scalar `X -> Option<Y>` processor into a
/// chunked one.
///
/// Use this when the stream is still scalar and phase must be tracked across
/// time. It does not by itself turn a chunk `[X; N]` into one output `Y`; pair
/// it with [`Decimator`] or [`TryDecimator`] for that.
///
/// Together with [`crate::Hold`], this forms the scalar optional-sample pair:
/// `Downsample` removes samples by emitting `None`, while `Hold` fills those
/// gaps again by repeating the last present sample.
///
/// Compare with:
/// - [`crate::Rate`]: stateless chunk-slot conversion
/// - [`Decimator`]: chunk adapter over a scalar `X -> Option<Y>` stage
///
/// State is the current countdown and should usually be initialized to `0`.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Downsample, SplitProcess};
///
/// let ds = Downsample(2);
/// let mut state = 0;
/// assert_eq!(ds.process(&mut state, 10), Some(10));
/// assert_eq!(ds.process(&mut state, 11), None);
/// assert_eq!(ds.process(&mut state, 12), None);
/// assert_eq!(ds.process(&mut state, 13), Some(13));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Downsample(pub u32);

impl<X: Copy> SplitProcess<X, Option<X>, u32> for Downsample {
    fn process(&self, state: &mut u32, x: X) -> Option<X> {
        if let Some(index) = state.checked_sub(1) {
            *state = index;
            None
        } else {
            *state = self.0;
            Some(x)
        }
    }
}

/// Zero-order hold over optional input samples.
///
/// `Some(x)` updates the held value, while `None` repeats the previous one.
/// This is useful for interpolation pipelines and event-driven sample streams.
///
/// Together with [`Downsample`], this forms the scalar optional-sample pair:
/// `Downsample` creates gaps by emitting `None`, while `Hold` turns those
/// gaps back into a continuous stream by repeating the last present sample.
///
/// At the chunk level, [`Interpolator`] plays the analogous role for turning a
/// scalar stream into chunk output.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Hold, Process};
///
/// let mut hold = Hold(5);
/// assert_eq!(hold.process(None), 5);
/// assert_eq!(hold.process(Some(7)), 7);
/// assert_eq!(hold.process(None), 7);
/// ```
#[derive(Debug, Copy, Clone, Default)]
#[repr(transparent)]
pub struct Hold<T>(pub T);

impl<T: Copy> crate::Process<Option<T>, T> for Hold<T> {
    fn process(&mut self, x: Option<T>) -> T {
        if let Some(x) = x {
            self.0 = x;
        }
        self.0
    }
}

/// Adapt a scalar optional-output stage to chunk input mode.
///
/// Synchronizes to the inner tick by discarding samples after tick.
/// Panics if tick does not match `N`.
///
/// This is the chunked counterpart to [`Interpolator`].
///
/// The inner processor must tick exactly once per input chunk. `Decimator`
/// processes the whole chunk and panics if the contract is violated. Use
/// [`TryDecimator`] when violating that contract should be reported instead of
/// panicking.
///
/// Unlike [`crate::Rate`], this adapter still runs the
/// inner processor on every sample in the chunk before choosing the output.
/// That is the right semantics for recursive stages such as CIC decimators.
///
/// Conceptually, this is the chunk-level companion to [`crate::Downsample`]:
/// `Downsample` gates a scalar stream into `Option<Y>`, while `Decimator`
/// turns that exact-one-tick-per-chunk protocol into `[X; N] -> Y`.
/// Unlike [`crate::ChunkIn`], this still executes the inner stage on every
/// sample in the chunk and is therefore the right adapter for recursive
/// decimators.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Decimator, FnSplitProcess, SplitProcess};
///
/// let proc = Decimator(FnSplitProcess(|state: &mut bool, x: i32| {
///     let y = if *state { Some(x) } else { None };
///     *state = !*state;
///     y
/// }));
///
/// let mut tick = false;
/// assert_eq!(proc.process(&mut tick, [1, 2]), 2);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Decimator<P>(pub P);
impl<X: Copy, Y, C: SplitProcess<X, Option<Y>, S>, S, const N: usize> SplitProcess<[X; N], Y, S>
    for Decimator<C>
{
    fn process(&self, state: &mut S, x: [X; N]) -> Y {
        const { assert!(N > 0) }
        TryDecimator(&self.0).process(state, x).unwrap()
    }
}
impl<X: Copy, C, S> SplitInplace<X, S> for Decimator<C> where Self: SplitProcess<X, X, S> {}

/// Error returned by [`TryDecimator`] when the inner decimator does not tick
/// exactly once per input chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecimatorError {
    /// No output sample was produced for the chunk.
    NoTick,
    /// More than one output sample was produced for the chunk.
    ExtraTick,
}

/// Checked variant of [`Decimator`].
///
/// This preserves the same chunked interface but reports contract violations
/// instead of panicking.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{DecimatorError, FnSplitProcess, SplitProcess, TryDecimator};
///
/// let proc = TryDecimator(FnSplitProcess(|state: &mut bool, x: i32| {
///     let y = if *state { Some(x) } else { None };
///     *state = !*state;
///     y
/// }));
///
/// let mut tick = false;
/// assert_eq!(proc.process(&mut tick, [1, 2]), Ok(2));
///
/// let never = TryDecimator(FnSplitProcess(|_: &mut (), _: i32| None::<i32>));
/// let mut state = ();
/// assert_eq!(
///     never.process(&mut state, [1, 2]),
///     Err(DecimatorError::NoTick)
/// );
/// ```
#[derive(Clone, Debug, Default)]
pub struct TryDecimator<P>(pub P);
impl<X: Copy, Y, C: SplitProcess<X, Option<Y>, S>, S, const N: usize>
    SplitProcess<[X; N], Result<Y, DecimatorError>, S> for TryDecimator<C>
{
    fn process(&self, state: &mut S, x: [X; N]) -> Result<Y, DecimatorError> {
        const { assert!(N > 0) }
        let mut y = None;
        for x in x {
            if let Some(next) = self.0.process(state, x)
                && y.replace(next).is_some()
            {
                return Err(DecimatorError::ExtraTick);
            }
        }
        y.ok_or(DecimatorError::NoTick)
    }
}

/// Lift a processor through `Option` or `Result`.
///
/// This is useful when a processor should only run on present/valid samples
/// while preserving outer framing or error signaling. It changes control-flow
/// shape, not block layout.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Map, Offset, SplitProcess};
///
/// let proc = Map(Offset(3));
/// let mut state = ();
/// assert_eq!(proc.process(&mut state, Some(4)), Some(7));
/// assert_eq!(proc.process(&mut state, None::<i32>), None);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Map<P>(pub P);
impl<X: Copy, Y, C: SplitProcess<X, Y, S>, S> SplitProcess<Option<X>, Option<Y>, S> for Map<C> {
    fn process(&self, state: &mut S, x: Option<X>) -> Option<Y> {
        x.map(|x| self.0.process(state, x))
    }
}
impl<X: Copy, Y, C: SplitProcess<X, Y, S>, S, E: Copy> SplitProcess<Result<X, E>, Result<Y, E>, S>
    for Map<C>
{
    fn process(&self, state: &mut S, x: Result<X, E>) -> Result<Y, E> {
        x.map(|x| self.0.process(state, x))
    }
}
impl<X: Copy, C: SplitInplace<X, S>, S> SplitInplace<X, S> for Map<C> where
    Self: SplitProcess<X, X, S>
{
}

/// Elementwise fixed-size chunk lifting.
///
/// Adapt a `X -> Y` processor into a `[X; N] -> [Y; N]` processor
/// by flattening input and output.
///
/// This is the simplest array-lifting adapter and is often the right choice
/// when a scalar stage should run elementwise over fixed-size chunks with no
/// rate change and no frame semantics beyond flattening.
///
/// Prefer the more specific adapters when the inner stage consumes or produces
/// grouped samples (`ChunkIn`, `ChunkOut`, `ChunkInOut`) or when stream phase is
/// part of the semantics (`Interpolator`, `Decimator`).
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Chunk, Offset, Process, Split};
///
/// let mut p = Split::stateless(Chunk(Offset(3)));
/// assert_eq!(p.process([1, 2, 3]), [4, 5, 6]);
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct Chunk<P>(pub P);
impl<C: SplitProcess<X, Y, S>, S, X: Copy, Y, const N: usize> SplitProcess<[X; N], [Y; N], S>
    for Chunk<C>
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; N] {
        x.map(|x| self.0.process(state, x))
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; N]]) {
        self.0.block(state, x.as_flattened(), y.as_flattened_mut())
    }
}
impl<C: SplitInplace<X, S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S> for Chunk<C> {
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        self.0.inplace(state, xy.as_flattened_mut())
    }
}

/// Fixed-ratio chunk adapter for grouped input.
///
/// Adapt a `[X; R] -> Y` processor to `[X; N=R*M]->[Y; M]` for any `M`
/// by flattening and re-chunking input.
///
/// Use this when the inner stage consumes several input samples per output, such
/// as a small decimating FIR kernel. This is a structural regrouping adapter:
/// it does not track stream phase across calls.
///
/// See also [`ChunkOut`] and [`ChunkInOut`].
///
/// # Examples
///
/// ```rust
/// use dsp_process::{ChunkIn, FnSplitProcess, Process, Split};
///
/// let mut p = Split::stateless(ChunkIn::<_, 2>(FnSplitProcess(
///     |_: &mut (), [a, b]: [i32; 2]| a + b,
/// )));
/// assert_eq!(p.process([1, 2, 3, 4]), [3, 7]);
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct ChunkIn<P, const R: usize>(pub P);
impl<C: SplitProcess<[X; R], Y, S>, S, X: Copy, Y, const N: usize, const R: usize, const M: usize>
    SplitProcess<[X; N], [Y; M], S> for ChunkIn<C, R>
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(R * M == N) }
        let (x, []) = x.as_chunks() else {
            unreachable!()
        };
        core::array::from_fn(|i| self.0.process(state, x[i]))
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(R * M == N) }
        let (x, []) = x.as_flattened().as_chunks() else {
            unreachable!()
        };
        self.0.block(state, x, y.as_flattened_mut())
    }
}
impl<C: SplitInplace<[X; 1], S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S>
    for ChunkIn<C, 1>
where
    Self: SplitProcess<[X; N], [X; N], S>,
{
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(state, xy)
    }
}

/// Fixed-ratio chunk adapter for grouped output.
///
/// Adapt a `X -> [Y; R]` processor to `[X; N]->[Y; M = R*N]` for any `N`
/// by flattening and re-chunking output.
///
/// This is the natural adapter for small fixed-ratio interpolation kernels.
/// Use [`ChunkOutPod`] when the output type is POD and the flattening cost
/// matters. This is a structural regrouping adapter, not a phased stream
/// interpolator; use [`Interpolator`] when the inner stage is naturally
/// `Option<X> -> Y`.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{ChunkOut, FnSplitProcess, Process, Split};
///
/// let mut p = Split::stateless(ChunkOut::<_, 2>(FnSplitProcess(|_: &mut (), x: i32| {
///     [x, -x]
/// })));
/// assert_eq!(p.process([2, 3]), [2, -2, 3, -3]);
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct ChunkOut<P, const R: usize>(pub P);
impl<C, S, X: Copy, Y: Default + Copy, const N: usize, const R: usize, const M: usize>
    SplitProcess<[X; N], [Y; M], S> for ChunkOut<C, R>
where
    C: SplitProcess<X, [Y; R], S>,
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(R * N == M) }
        // `poor-codegen-from-fn-iter-next`: if this changes, use a real conversion primitive.
        let mut y = [Y::default(); M];
        let (yy, []) = y.as_chunks_mut() else {
            unreachable!()
        };
        for (x, y) in x.into_iter().zip(yy) {
            *y = self.0.process(state, x);
        }
        y
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(R * N == M) }
        let (y, []) = y.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.block(state, x.as_flattened(), y)
    }
}
impl<C: SplitInplace<[X; 1], S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S>
    for ChunkOut<C, 1>
where
    Self: SplitProcess<[X; N], [X; N], S>,
{
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(state, xy)
    }
}

/// POD-specialized [`ChunkOut`] variant.
///
/// This keeps the same semantics as [`ChunkOut`] but uses a bytemuck-backed
/// representation cast to flatten `[[Y; R]; N]` into `[Y; R * N]` without the
/// generic scratch-buffer path.
///
/// This is only available when `Y` is `bytemuck::Pod` and is mainly a codegen/cache
/// choice, not a semantic one.
///
/// # Examples
#[cfg_attr(
    feature = "bytemuck",
    doc = r##"/// ```rust
/// use dsp_process::{ChunkOutPod, FnSplitProcess, Process, Split};
///
/// let mut p = Split::stateless(ChunkOutPod::<_, 2>(FnSplitProcess(|_: &mut (), x: i32| {
///     [x, -x]
/// })));
/// assert_eq!(p.process([2, 3]), [2, -2, 3, -3]);
/// ```"##
)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ChunkOutPod<P, const R: usize>(pub P);
#[cfg(feature = "bytemuck")]
impl<C, S, X: Copy, Y: bytemuck::Pod, const N: usize, const R: usize, const M: usize>
    SplitProcess<[X; N], [Y; M], S> for ChunkOutPod<C, R>
where
    C: SplitProcess<X, [Y; R], S>,
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(R * N == M) }
        bytemuck::cast::<[[Y; R]; N], [Y; M]>(x.map(|x| self.0.process(state, x)))
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(R * N == M) }
        let (y, []) = y.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.block(state, x.as_flattened(), y)
    }
}
#[cfg(feature = "bytemuck")]
impl<C: SplitInplace<[X; 1], S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S>
    for ChunkOutPod<C, 1>
where
    Self: SplitProcess<[X; N], [X; N], S>,
{
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(state, xy)
    }
}

/// General fixed-ratio regrouping adapter for chunked input and output.
///
/// Adapt a `[X; Q] -> [Y; R]` processor to `[X; N = Q*I]->[Y; M = R*I]` for any `I`
/// by flattening and re-chunking input and output.
///
/// This is the most general fixed-ratio chunk adapter in the crate. It requires
/// the input and output to represent the same number of inner chunks. Reach for
/// it when neither plain [`Chunk`], [`ChunkIn`], nor [`ChunkOut`] captures the
/// actual grouping relation.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{ChunkInOut, FnSplitProcess, Process, Split};
///
/// let mut p = Split::stateless(ChunkInOut::<_, 2, 1>(FnSplitProcess(
///     |_: &mut (), [a, b]: [i32; 2]| [a + b],
/// )));
/// assert_eq!(p.process([1, 2, 3, 4]), [3, 7]);
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct ChunkInOut<P, const Q: usize, const R: usize>(pub P);
impl<
    C,
    S,
    X: Copy,
    Y: Default + Copy,
    const Q: usize,
    const N: usize,
    const R: usize,
    const M: usize,
> SplitProcess<[X; N], [Y; M], S> for ChunkInOut<C, Q, R>
where
    C: SplitProcess<[X; Q], [Y; R], S>,
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(Q > 0) }
        const { assert!(R > 0) }
        const { assert!(N.is_multiple_of(Q)) }
        const { assert!(M.is_multiple_of(R)) }
        const { assert!(N / Q == M / R) }
        // `poor-codegen-from-fn-iter-next`: if this changes, use a real conversion primitive.
        let mut y = [Y::default(); M];
        let (yy, []) = y.as_chunks_mut() else {
            unreachable!()
        };
        let (x, []) = x.as_chunks() else {
            unreachable!()
        };
        for (x, y) in x.iter().zip(yy) {
            *y = self.0.process(state, *x);
        }
        y
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(Q > 0) }
        const { assert!(R > 0) }
        const { assert!(N.is_multiple_of(Q)) }
        const { assert!(M.is_multiple_of(R)) }
        const { assert!(N / Q == M / R) }
        let (x, []) = x.as_flattened().as_chunks() else {
            unreachable!()
        };
        let (y, []) = y.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.block(state, x, y)
    }
}
impl<C: SplitInplace<[X; 1], S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S>
    for ChunkInOut<C, 1, 1>
where
    Self: SplitProcess<[X; N], [X; N], S>,
{
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(state, xy)
    }
}
