use crate::{
    Block, BlockMut, ChannelMajor, SplitBlockInplace, SplitBlockProcess, SplitInplace, SplitProcess,
};
use core::marker::PhantomData;

//////////// SPLIT COMPOSE ////////////

/// Chain two different processors with an explicit intermediate type.
///
/// This is the heterogeneous serial-composition primitive for tuples. The first
/// stage may change the sample type, while the second stage must accept that
/// intermediate value in place.
impl<X: Copy, Y: Copy, C0, C1, S0, S1> SplitProcess<X, Y, (S0, S1)> for (C0, C1)
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitInplace<Y, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: X) -> Y {
        self.1
            .process(&mut state.1, self.0.process(&mut state.0, x))
    }

    fn block(&self, state: &mut (S0, S1), x: &[X], y: &mut [Y]) {
        self.0.block(&mut state.0, x, y);
        self.1.inplace(&mut state.1, y);
    }
}

impl<X: Copy, C0, C1, S0, S1> SplitInplace<X, (S0, S1)> for (C0, C1)
where
    C0: SplitInplace<X, S0>,
    C1: SplitInplace<X, S1>,
{
    fn inplace(&self, state: &mut (S0, S1), xy: &mut [X]) {
        self.0.inplace(&mut state.0, xy);
        self.1.inplace(&mut state.1, xy);
    }
}

/// Chain multiple homogeneous processors over one sample type.
///
/// The slice may be empty, in which case `block()` acts as identity.
impl<X: Copy, C, S> SplitProcess<X, X, [S]> for [C]
where
    C: SplitInplace<X, S>,
{
    fn process(&self, state: &mut [S], x: X) -> X {
        debug_assert_eq!(self.len(), state.len());
        self.iter()
            .zip(state.iter_mut())
            .fold(x, |x, (c, s)| c.process(s, x))
    }

    fn block(&self, state: &mut [S], x: &[X], y: &mut [X]) {
        debug_assert_eq!(self.len(), state.len());
        if let Some(((c0, c), (s0, s))) = self.split_first().zip(state.split_first_mut()) {
            c0.block(s0, x, y);
            for (c, s) in c.iter().zip(s) {
                c.inplace(s, y);
            }
        } else {
            y.copy_from_slice(x);
        }
    }
}

impl<X: Copy, C, S> SplitInplace<X, [S]> for [C]
where
    C: SplitInplace<X, S>,
{
    fn inplace(&self, state: &mut [S], xy: &mut [X]) {
        debug_assert_eq!(self.len(), state.len());
        for (c, s) in self.iter().zip(state.iter_mut()) {
            c.inplace(s, xy);
        }
    }
}

/// Chain a non-empty homogeneous array of processors with one initial type change.
impl<X: Copy, Y: Copy, C, S, const N: usize> SplitProcess<X, Y, [S; N]> for [C; N]
where
    C: SplitProcess<X, Y, S> + SplitInplace<Y, S>,
{
    fn process(&self, state: &mut [S; N], x: X) -> Y {
        const { assert!(N > 0) }
        let Some(((c0, c), (s0, s))) = self.split_first().zip(state.split_first_mut()) else {
            unreachable!()
        };
        c.iter()
            .zip(s.iter_mut())
            .fold(c0.process(s0, x), |x, (c, s)| c.process(s, x))
    }

    fn block(&self, state: &mut [S; N], x: &[X], y: &mut [Y]) {
        const { assert!(N > 0) }
        let Some(((c0, c), (s0, s))) = self.split_first().zip(state.split_first_mut()) else {
            unreachable!()
        };
        c0.block(s0, x, y);
        for (c, s) in c.iter().zip(s.iter_mut()) {
            c.inplace(s, y)
        }
    }
}

impl<X: Copy, C, S, const N: usize> SplitInplace<X, [S; N]> for [C; N]
where
    C: SplitInplace<X, S>,
{
    fn inplace(&self, state: &mut [S; N], xy: &mut [X]) {
        self.as_ref().inplace(state.as_mut(), xy)
    }
}

//////////// SPLIT MINOR ////////////

/// Processor-minor, data-major
///
/// The various Process tooling implementations for `Minor`
/// place the data loop as the outer-most loop (processor-minor, data-major).
/// This is optimal for processors with small or no state and configuration.
///
/// Chain of large processors are implemented through native
/// tuples and slices/arrays or [`Major`].
/// Those optimize well if the sizes obey configuration ~ state > data.
/// If they do not, use `Minor`.
///
/// Note that the major implementations only override the behavior
/// for `block()` and `inplace()`. `process()` is unaffected and the same for all.
///
/// This wrapper changes loop ordering rather than signal semantics.
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct Minor<C: ?Sized, U> {
    /// An intermediate data type
    _intermediate: PhantomData<U>,
    /// The inner configurations
    inner: C,
}

impl<C, U> Minor<C, U> {
    /// Create a [`Minor`] wrapper around an existing composition.
    #[must_use]
    pub const fn new(inner: C) -> Self {
        Self {
            inner,
            _intermediate: PhantomData,
        }
    }

    /// Consume the wrapper and return the inner composition.
    #[must_use]
    pub fn into_inner(self) -> C {
        self.inner
    }

    /// Borrow the wrapped composition.
    #[must_use]
    pub fn inner(&self) -> &C {
        &self.inner
    }
}

impl<X: Copy, U: Copy, Y, C0, C1, S0, S1> SplitProcess<X, Y, (S0, S1)> for Minor<(C0, C1), U>
where
    C0: SplitProcess<X, U, S0>,
    C1: SplitProcess<U, Y, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: X) -> Y {
        self.inner
            .1
            .process(&mut state.1, self.inner.0.process(&mut state.0, x))
    }
}

/// A chain of multiple small filters of the same type
impl<X: Copy, C, S> SplitProcess<X, X, [S]> for Minor<[C], X>
where
    C: SplitProcess<X, X, S>,
{
    fn process(&self, state: &mut [S], x: X) -> X {
        debug_assert_eq!(self.inner.len(), state.len());
        self.inner
            .iter()
            .zip(state.iter_mut())
            .fold(x, |x, (c, s)| c.process(s, x))
    }
}

/// A chain of multiple small filters of the same type
impl<X: Copy, Y: Copy, C, S, const N: usize> SplitProcess<X, Y, [S; N]> for Minor<[C; N], Y>
where
    C: SplitProcess<X, Y, S> + SplitProcess<Y, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: X) -> Y {
        const { assert!(N > 0) }
        let Some(((c0, c), (s0, s))) = self.inner.split_first().zip(state.split_first_mut()) else {
            unreachable!()
        };
        c.iter()
            .zip(s.iter_mut())
            .fold(c0.process(s0, x), |x, (c, s)| c.process(s, x))
    }
}

impl<X: Copy, U, C, S> SplitInplace<X, S> for Minor<C, U> where Self: SplitProcess<X, X, S> {}

//////////// SPLIT PARALLEL ////////////

/// Fan out parallel input to parallel processors.
///
/// Use this with [`crate::Add`], [`crate::Sub`], or [`crate::Mul`] when the
/// branch outputs should be reduced again.
#[derive(Clone, Copy, Debug, Default)]
pub struct Parallel<P>(P);

impl<P> Parallel<P> {
    /// Create a [`Parallel`] wrapper around an existing composition.
    #[must_use]
    pub const fn new(inner: P) -> Self {
        Self(inner)
    }

    /// Consume the wrapper and return the inner composition.
    #[must_use]
    pub fn into_inner(self) -> P {
        self.0
    }
}

impl<X0: Copy, X1: Copy, Y0, Y1, C0, C1, S0, S1> SplitProcess<(X0, X1), (Y0, Y1), (S0, S1)>
    for Parallel<(C0, C1)>
where
    C0: SplitProcess<X0, Y0, S0>,
    C1: SplitProcess<X1, Y1, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: (X0, X1)) -> (Y0, Y1) {
        (
            self.0.0.process(&mut state.0, x.0),
            self.0.1.process(&mut state.1, x.1),
        )
    }
}

impl<X: Copy, Y, C0, C1, S0, S1> SplitProcess<[X; 2], [Y; 2], (S0, S1)> for Parallel<(C0, C1)>
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitProcess<X, Y, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: [X; 2]) -> [Y; 2] {
        [
            self.0.0.process(&mut state.0, x[0]),
            self.0.1.process(&mut state.1, x[1]),
        ]
    }
}

impl<X: Copy, Y, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]> for Parallel<[C; N]>
where
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        // `poor-codegen-from-fn-iter-next`: keep this as direct indexed construction.
        core::array::from_fn(|i| self.0[i].process(&mut state[i], x[i]))
    }
}

impl<X: Copy, C, S> SplitInplace<X, S> for Parallel<C> where Self: SplitProcess<X, X, S> {}

//////////// TRANSPOSE ////////////

/// Data block transposition wrapper
///
/// Like [`Parallel`] for scalar processing, but with explicit channel-major
/// block processing through [`Block<_, _, ChannelMajor, _>`].
///
/// This wrapper does not allocate or physically transpose memory.
#[derive(Clone, Copy, Debug, Default)]
pub struct Transpose<C>(C);

impl<C> Transpose<C> {
    /// Create a [`Transpose`] wrapper around an existing composition.
    #[must_use]
    pub const fn new(inner: C) -> Self {
        Self(inner)
    }

    /// Consume the wrapper and return the inner composition.
    #[must_use]
    pub fn into_inner(self) -> C {
        self.0
    }
}

impl<X: Copy, Y, C0, C1, S0, S1> SplitProcess<[X; 2], [Y; 2], (S0, S1)> for Transpose<(C0, C1)>
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitProcess<X, Y, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: [X; 2]) -> [Y; 2] {
        [
            self.0.0.process(&mut state.0, x[0]),
            self.0.1.process(&mut state.1, x[1]),
        ]
    }
}

impl<'a, 'b, X: Copy, Y, C0, C1, S0, S1>
    SplitBlockProcess<Block<'a, X, ChannelMajor, 2>, BlockMut<'b, Y, ChannelMajor, 2>, (S0, S1)>
    for Transpose<(C0, C1)>
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitProcess<X, Y, S1>,
{
    fn process_block(
        &self,
        state: &mut (S0, S1),
        x: Block<'a, X, ChannelMajor, 2>,
        mut y: BlockMut<'b, Y, ChannelMajor, 2>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        self.0.0.block(&mut state.0, x.channel(0), y.channel_mut(0));
        self.0.1.block(&mut state.1, x.channel(1), y.channel_mut(1));
    }
}

impl<X: Copy, Y, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]> for Transpose<[C; N]>
where
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        // `poor-codegen-from-fn-iter-next`: keep this as direct indexed construction.
        core::array::from_fn(|i| self.0[i].process(&mut state[i], x[i]))
    }
}

impl<'a, 'b, X: Copy, Y, C, S, const N: usize>
    SplitBlockProcess<Block<'a, X, ChannelMajor, N>, BlockMut<'b, Y, ChannelMajor, N>, [S; N]>
    for Transpose<[C; N]>
where
    C: SplitProcess<X, Y, S>,
{
    fn process_block(
        &self,
        state: &mut [S; N],
        x: Block<'a, X, ChannelMajor, N>,
        mut y: BlockMut<'b, Y, ChannelMajor, N>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        for ((c, s), i) in self.0.iter().zip(state.iter_mut()).zip(0..) {
            c.block(s, x.channel(i), y.channel_mut(i))
        }
    }
}

impl<X: Copy, C, S> SplitInplace<X, S> for Transpose<C> where Self: SplitProcess<X, X, S> {}

impl<'a, X: Copy, C0, C1, S0, S1> SplitBlockInplace<BlockMut<'a, X, ChannelMajor, 2>, (S0, S1)>
    for Transpose<(C0, C1)>
where
    C0: SplitInplace<X, S0>,
    C1: SplitInplace<X, S1>,
{
    fn inplace_block(&self, state: &mut (S0, S1), mut xy: BlockMut<'a, X, ChannelMajor, 2>) {
        self.0.0.inplace(&mut state.0, xy.channel_mut(0));
        self.0.1.inplace(&mut state.1, xy.channel_mut(1));
    }
}

impl<'a, X: Copy, C, S, const N: usize> SplitBlockInplace<BlockMut<'a, X, ChannelMajor, N>, [S; N]>
    for Transpose<[C; N]>
where
    C: SplitInplace<X, S>,
{
    fn inplace_block(&self, state: &mut [S; N], mut xy: BlockMut<'a, X, ChannelMajor, N>) {
        for ((c, s), i) in self.0.iter().zip(state.iter_mut()).zip(0..) {
            c.inplace(s, xy.channel_mut(i));
        }
    }
}

//////////// CHANNELS ////////////

/// Multiple channels processed with one shared configuration.
///
/// This is the natural companion to [`SplitProcess`]: immutable coefficients are
/// stored once while each channel keeps separate mutable state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Channels<C>(C);

impl<C> Channels<C> {
    /// Create a [`Channels`] wrapper around an existing composition.
    #[must_use]
    pub const fn new(inner: C) -> Self {
        Self(inner)
    }

    /// Consume the wrapper and return the inner composition.
    #[must_use]
    pub fn into_inner(self) -> C {
        self.0
    }
}

/// Process data from multiple channels with a common configuration.
///
/// For layout-sensitive block processing use [`Block<_, _, ChannelMajor, _>`].
impl<X: Copy, Y, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]> for Channels<C>
where
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        // `poor-codegen-from-fn-iter-next`: keep this as direct indexed construction.
        core::array::from_fn(|i| self.0.process(&mut state[i], x[i]))
    }
}

impl<'a, 'b, X: Copy, Y, C, S, const N: usize>
    SplitBlockProcess<Block<'a, X, ChannelMajor, N>, BlockMut<'b, Y, ChannelMajor, N>, [S; N]>
    for Channels<C>
where
    C: SplitProcess<X, Y, S>,
{
    fn process_block(
        &self,
        state: &mut [S; N],
        x: Block<'a, X, ChannelMajor, N>,
        mut y: BlockMut<'b, Y, ChannelMajor, N>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        for (state, i) in state.iter_mut().zip(0..) {
            self.0.block(state, x.channel(i), y.channel_mut(i))
        }
    }
}

impl<X: Copy, C, S> SplitInplace<X, S> for Channels<C> where Self: SplitProcess<X, X, S> {}

impl<'a, X: Copy, C, S, const N: usize> SplitBlockInplace<BlockMut<'a, X, ChannelMajor, N>, [S; N]>
    for Channels<C>
where
    C: SplitInplace<X, S>,
{
    fn inplace_block(&self, state: &mut [S; N], mut xy: BlockMut<'a, X, ChannelMajor, N>) {
        for (state, i) in state.iter_mut().zip(0..) {
            self.0.inplace(state, xy.channel_mut(i));
        }
    }
}

//////////// SPLIT MAJOR ////////////

/// Chain of major processors with intermediate buffer supporting block() processing
/// from individual block()s
///
/// Prefer default composition for X->X->X, arrays/slices where inplace is possible
///
/// Use [`Major`] when block processing benefits from explicit scratch storage
/// between stages, for example to improve locality for larger processors.
#[derive(Debug, Clone, Copy, Default)]
pub struct Major<P: ?Sized, U> {
    /// Intermediate buffer
    _buf: PhantomData<U>,
    /// The inner processors
    inner: P,
}
impl<P, U> Major<P, U> {
    /// Create a [`Major`] wrapper around an existing composition.
    #[must_use]
    pub const fn new(inner: P) -> Self {
        Self {
            inner,
            _buf: PhantomData,
        }
    }

    /// Consume the wrapper and return the inner composition.
    #[must_use]
    pub fn into_inner(self) -> P {
        self.inner
    }

    /// Borrow the wrapped composition.
    #[must_use]
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<X: Copy, U: Copy + Default, Y, C0, C1, S0, S1, const N: usize> SplitProcess<X, Y, (S0, S1)>
    for Major<(C0, C1), [U; N]>
where
    C0: SplitProcess<X, U, S0>,
    C1: SplitProcess<U, Y, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: X) -> Y {
        self.inner
            .1
            .process(&mut state.1, self.inner.0.process(&mut state.0, x))
    }

    fn block(&self, state: &mut (S0, S1), x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        let mut u = [U::default(); N];
        let (x, xr) = x.as_chunks::<N>();
        let (y, yr) = y.as_chunks_mut::<N>();
        for (x, y) in x.iter().zip(y) {
            self.inner.0.block(&mut state.0, x, &mut u);
            self.inner.1.block(&mut state.1, &u, y);
        }
        let ur = &mut u[..xr.len()];
        self.inner.0.block(&mut state.0, xr, ur);
        self.inner.1.block(&mut state.1, ur, yr);
    }
}

impl<X: Copy, U: Copy + Default, C0, C1, S0, S1, const N: usize> SplitInplace<X, (S0, S1)>
    for Major<(C0, C1), [U; N]>
where
    C0: SplitProcess<X, U, S0>,
    C1: SplitProcess<U, X, S1>,
{
    fn inplace(&self, state: &mut (S0, S1), xy: &mut [X]) {
        let mut u = [U::default(); N];
        let (xy, xyr) = xy.as_chunks_mut::<N>();
        for xy in xy {
            self.inner.0.block(&mut state.0, xy, &mut u);
            self.inner.1.block(&mut state.1, &u, xy);
        }
        let ur = &mut u[..xyr.len()];
        self.inner.0.block(&mut state.0, xyr, ur);
        self.inner.1.block(&mut state.1, ur, xyr);
    }
}
