use core::array::{from_fn, repeat};

use crate::{
    Channels, Chunk, Decimator, Frames, Inplace, Interpolator, Major, Map, Minor, Parallel,
    Process, SplitInplace, SplitProcess, Transpose, TryDecimator,
};

//////////// SPLIT ////////////

/// A stateful processor assembled from split configuration and state.
///
/// [`Split`] is the bridge between [`SplitProcess`] and [`Process`]: it stores
/// the immutable configuration and mutable runtime state together so the pair can
/// be passed around as a conventional stateful processor.
///
/// Reach for this when a split-state filter needs to be owned as one value, and
/// use [`channels()`](Self::channels), [`minor()`](Self::minor), or
/// [`major()`](Self::major) when changing how that owned processor is composed.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Offset, Process, Split};
///
/// let mut p = Split::stateless(Offset(3));
/// assert_eq!(p.process(5), 8);
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct Split<C, S> {
    /// Processor configuration
    pub config: C,
    /// Processor state
    pub state: S,
}

impl<X: Copy, Y, S, C: SplitProcess<X, Y, S>> Process<X, Y> for Split<C, S> {
    fn process(&mut self, x: X) -> Y {
        self.config.process(&mut self.state, x)
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.config.block(&mut self.state, x, y)
    }
}

impl<X: Copy, S, C: SplitInplace<X, S>> Inplace<X> for Split<C, S> {
    fn inplace(&mut self, xy: &mut [X]) {
        self.config.inplace(&mut self.state, xy);
    }
}

impl<C, S> Split<C, S> {
    /// Create a new [`Split`] from explicit configuration and state values.
    #[must_use]
    pub const fn new(config: C, state: S) -> Self {
        Self { config, state }
    }

    /// Statically assert that this implements Process<X, Y>
    pub const fn assert_process<X: Copy, Y>(&self)
    where
        Self: Process<X, Y>,
    {
    }
}

/// Marker for values that should live in the opposite half of a [`Split`].
///
/// To be used in `Split<Unsplit<P>, ()>` and `Split<(), Unsplit<P>>`
/// to mark processors requiring no state and no configuration respectively.
///
/// Most users will not construct this directly and should prefer
/// [`Split::stateless`] and [`Split::stateful`].
#[derive(Debug, Copy, Clone, Default)]
#[repr(transparent)]
pub struct Unsplit<P>(pub P);

impl<C> Split<C, ()> {
    /// Create a [`Split`] with configuration only and unit state.
    #[must_use]
    pub fn stateless(config: C) -> Self {
        Self::new(config, ())
    }
}

impl<S> Split<(), Unsplit<S>> {
    /// Create a [`Split`] with state only and unit configuration.
    #[must_use]
    pub fn stateful(state: S) -> Self {
        Self::new((), Unsplit(state))
    }
}

/// Unzip two splits into one
impl<C0, C1, S0, S1> core::ops::Mul<Split<C1, S1>> for Split<C0, S0> {
    type Output = Split<(C0, C1), (S0, S1)>;

    fn mul(self, rhs: Split<C1, S1>) -> Self::Output {
        Split::from((self, rhs))
    }
}

/// Unzip two splits into one parallel
impl<C0, C1, S0, S1> core::ops::Add<Split<C1, S1>> for Split<C0, S0> {
    type Output = Split<Parallel<(C0, C1)>, (S0, S1)>;

    fn add(self, rhs: Split<C1, S1>) -> Self::Output {
        Split::from((self, rhs)).parallel()
    }
}

/// Unzip two splits
impl<C0, C1, S0, S1> From<(Split<C0, S0>, Split<C1, S1>)> for Split<(C0, C1), (S0, S1)> {
    fn from(value: (Split<C0, S0>, Split<C1, S1>)) -> Self {
        Split::new(
            (value.0.config, value.1.config),
            (value.0.state, value.1.state),
        )
    }
}

/// Unzip multiple splits
impl<C, S, const N: usize> From<[Split<C, S>; N]> for Split<[C; N], [S; N]> {
    fn from(splits: [Split<C, S>; N]) -> Self {
        // Not efficient or nice, but this is usually not a hot path
        let mut splits = splits.map(|s| (Some(s.config), Some(s.state)));
        Self::new(
            from_fn(|i| splits[i].0.take().unwrap()),
            from_fn(|i| splits[i].1.take().unwrap()),
        )
    }
}

impl<C, S> Split<C, S> {
    /// Convert to [`Minor`] composition.
    ///
    /// This keeps the same logical processor but requests sample-by-sample
    /// `block()`/`inplace()` execution of the wrapped serial composition.
    ///
    /// Use this for small fine-grained stages, or when tuple composition must
    /// cross an intermediate type and the downstream stage is not
    /// [`SplitInplace`] for that intermediate. Avoid it when preserving
    /// stage-major block processing is important for cache behavior or SIMD.
    #[must_use]
    pub fn minor<U>(self) -> Split<Minor<C, U>, S> {
        Split::new(Minor::new(self.config), self.state)
    }

    /// Convert to [`Major`] composition with an explicit intermediate buffer.
    ///
    /// Use this when preserving stage-major block processing is more important
    /// than avoiding an intermediate scratch buffer, especially for larger
    /// stages or stages with meaningful `block()` specializations.
    #[must_use]
    pub fn major<U>(self) -> Split<Major<C, U>, S> {
        Split::new(Major::new(self.config), self.state)
    }

    /// Convert to [`Parallel`] composition.
    ///
    /// This expresses structural branching: each input lane is routed to the
    /// matching branch and outputs stay separate unless reduced explicitly.
    #[must_use]
    pub fn parallel(self) -> Split<Parallel<C>, S> {
        Split::new(Parallel::new(self.config), self.state)
    }

    /// Map `Option` and `Result` around this processor.
    ///
    /// This lifts the processor through outer `Option`/`Result` control flow
    /// while preserving the current state unchanged.
    #[must_use]
    pub fn map(self) -> Split<Map<C>, S> {
        Split::new(Map(self.config), self.state)
    }

    /// Convert to elementwise fixed-size chunk processing.
    ///
    /// This is the basic array-lifting adapter. Use the more specific chunk or
    /// rate adapters when samples must be regrouped rather than processed
    /// elementwise.
    #[must_use]
    pub fn chunk(self) -> Split<Chunk<C>, S> {
        Split::new(Chunk(self.config), self.state)
    }

    /// Convert a scalar optional-input stage into chunk output mode.
    ///
    /// This preserves stream phase across one input sample expanded into one
    /// output chunk. Prefer this over structural chunk regrouping when the
    /// inner stage is naturally `Option<X> -> Y`.
    #[must_use]
    pub fn interpolate(self) -> Split<Interpolator<C>, S> {
        Split::new(Interpolator(self.config), self.state)
    }

    /// Convert a scalar optional-output stage into unchecked chunk input mode.
    ///
    /// This preserves stream phase across one input chunk collapsed into one
    /// output sample. Prefer this over structural chunk regrouping when the
    /// inner stage is naturally `X -> Option<Y>`.
    #[must_use]
    pub fn decimate(self) -> Split<Decimator<C>, S> {
        Split::new(Decimator(self.config), self.state)
    }

    /// Convert a scalar optional-output stage into checked chunk input mode.
    ///
    /// This is the checked form of [`decimate()`](Self::decimate), returning an
    /// error when the inner stage does not tick exactly once per input chunk.
    #[must_use]
    pub fn try_decimate(self) -> Split<TryDecimator<C>, S> {
        Split::new(TryDecimator(self.config), self.state)
    }

    /// Treat each frame of a frame-major block as one chunk sample.
    ///
    /// This bridges chunk-style processors such as [`crate::Chunk`],
    /// [`crate::ChunkIn`], [`crate::ChunkOut`], and [`crate::ChunkInOut`] into
    /// the typed block-view API without changing the backing layout.
    #[must_use]
    pub fn frames(self) -> Split<Frames<C>, S> {
        Split::new(Frames(self.config), self.state)
    }

    /// Duplicate the processor by cloning both configuration and current state.
    ///
    /// The current state is copied as-is. Use this only when duplicating the
    /// existing state is intentional, for example when seeding several identical
    /// branches from a known starting point.
    #[must_use]
    pub fn repeat<const N: usize>(self) -> Split<[C; N], [S; N]>
    where
        C: Clone,
        S: Clone,
    {
        Split::new(repeat(self.config), repeat(self.state))
    }

    /// Share one configuration across multiple cloned states via [`Channels`].
    ///
    /// This is usually preferable to [`repeat()`](Self::repeat) when the
    /// configuration should be shared but each channel needs its own mutable
    /// runtime state. For channel-major block processing, pair this with
    /// [`crate::Block`] using [`crate::ChannelMajor`].
    #[must_use]
    pub fn channels<const N: usize>(self) -> Split<Channels<C>, [S; N]>
    where
        S: Clone,
    {
        Split::new(Channels::new(self.config), repeat(self.state))
    }

    /// Convert to [`Transpose`] block semantics.
    ///
    /// Scalar `process()` is unchanged. Use this when parallel branches should
    /// process channel-major blocks as long contiguous per-channel slices.
    #[must_use]
    pub fn transpose(self) -> Split<Transpose<C>, S> {
        Split::new(Transpose::new(self.config), self.state)
    }
}

impl<C, S, U> Split<Minor<C, U>, S> {
    /// Strip minor
    #[must_use]
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.into_inner(), self.state)
    }
}

impl<C, S> Split<Parallel<C>, S> {
    /// Convert to serial
    #[must_use]
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.into_inner(), self.state)
    }
}

impl<C, S> Split<Frames<C>, S> {
    /// Remove frame-wise block adaptation.
    #[must_use]
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.0, self.state)
    }
}

impl<C, S> Split<Transpose<C>, S> {
    /// Convert to non-transposing
    #[must_use]
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.into_inner(), self.state)
    }
}

impl<C, S, B> Split<Major<C, B>, S> {
    /// Remove major intermediate buffering
    #[must_use]
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.into_inner(), self.state)
    }
}

impl<C0, C1, S0, S1> Split<(C0, C1), (S0, S1)> {
    /// Zip up a split
    #[must_use]
    pub fn zip(self) -> (Split<C0, S0>, Split<C1, S1>) {
        (
            Split::new(self.config.0, self.state.0),
            Split::new(self.config.1, self.state.1),
        )
    }
}

impl<C, S, const N: usize> Split<[C; N], [S; N]> {
    /// Zip up a split
    #[must_use]
    pub fn zip(self) -> [Split<C, S>; N] {
        let mut it = self.config.into_iter().zip(self.state);
        from_fn(|_| {
            let (c, s) = it.next().unwrap();
            Split::new(c, s)
        })
    }
}

/// Configuration-less filters
impl<X: Copy, Y, P: Process<X, Y>> SplitProcess<X, Y, Unsplit<P>> for () {
    fn process(&self, state: &mut Unsplit<P>, x: X) -> Y {
        state.0.process(x)
    }

    fn block(&self, state: &mut Unsplit<P>, x: &[X], y: &mut [Y]) {
        state.0.block(x, y)
    }
}

impl<X: Copy, P: Inplace<X>> SplitInplace<X, Unsplit<P>> for () {
    fn inplace(&self, state: &mut Unsplit<P>, xy: &mut [X]) {
        state.0.inplace(xy)
    }
}
