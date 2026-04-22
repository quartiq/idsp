use crate::{SplitInplace, SplitProcess};

/// Adapts an interpolator to output chunk mode
///
/// The inner processor is called with `Some(x)` once and then `None` `N-1` times
/// to synthesize one output chunk from one input sample.
///
/// This is convenient for polyphase interpolators and other stages whose natural
/// scalar interface is `Option<X> -> Y`.
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

/// Adapts a decimator to input chunk mode
///
/// Synchronizes to the inner tick by discarding samples after tick.
/// Panics if tick does not match `N`.
///
/// This is the chunked counterpart to [`Interpolator`].
/// Use [`TryDecimator`] when violating the one-tick-per-chunk contract should
/// be reported instead of panicking.
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
        x.into_iter()
            .find_map(|x| self.0.process(state, x))
            .unwrap()
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
/// assert_eq!(never.process(&mut state, [1, 2]), Err(DecimatorError::NoTick));
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

/// Map `Option` and `Result`
///
/// This is useful when a processor should only run on present/valid samples while
/// preserving outer framing or error signaling.
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

/// Chunked processing
///
/// Adapt a `X -> Y` processor into a `[X; N] -> [Y; N]` processor
/// by flattening input and output.
///
/// This is the simplest array-lifting adapter and is often the right choice when
/// a scalar processor should run elementwise over fixed-size blocks.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Chunk, Process, Split, Offset};
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

/// Chunked input
///
/// Adapt a `[X; R] -> Y` processor to `[X; N=R*M]->[Y; M]` for any `M`
/// by flattening and re-chunking input.
///
/// Use this when the inner stage consumes several input samples per output, such
/// as a small decimating FIR kernel.
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

/// Chunked output
///
/// Adapt a `X -> [Y; R]` processor to `[X; N]->[Y; M = R*N]` for any `N`
/// by flattening and re-chunking output.
///
/// This is the natural adapter for small fixed-ratio interpolation kernels.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{ChunkOut, FnSplitProcess, Process, Split};
///
/// let mut p = Split::stateless(ChunkOut::<_, 2>(FnSplitProcess(
///     |_: &mut (), x: i32| [x, -x],
/// )));
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

/// Chunked input and output
///
/// Adapt a `[X; Q] -> [Y; R]` processor to `[X; N = Q*I]->[Y; M = R*I]` for any `I`
/// by flattening and re-chunking input and output.
///
/// This is the most general fixed-ratio chunk adapter in the crate. It requires
/// the input and output to represent the same number of inner chunks.
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
