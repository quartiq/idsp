//! Sample processing, filtering, combination of filters.
use core::{
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
};

/// Processing block
///
/// Single input, single output
///
/// Process impls can be cascaded in (homogeneous) `[C; N]` arrays/`[C]` slices, and heterogeneous
/// `(C1, C2)` tuples. They can be used as configuration-major or
/// configuration-minor (through [`Minor`]) or in [`Sum`]s on complementary allpasses and polyphase banks.
/// Tuples, arrays, and Pairs, and Minor can be mixed and nested ad lib.
///
/// For a given filter configuration `C` and state `S` pair the trait is usually implemented
/// through [`Stateful<&'a C, &mut S>`] (created ad-hoc from by borrowing configuration and state)
/// or [`Stateful<C, S>`] (owned configuration and state).
/// Stateless filters should implement `Process` on `&Self` for composability through
/// `Stateful<&Stateless<Self>, &mut ()>`.
pub trait Process<X: Copy, Y = X> {
    /// Update the state with a new input and obtain an output
    fn process(&mut self, x: X) -> Y;

    /// Process a block of inputs into a block of outputs
    ///
    /// Input and output must be of the same size.
    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(y) {
            *y = self.process(*x);
        }
    }
}

/// Process a block in place.
pub trait Inplace<X: Copy>: Process<X> {
    /// Process an input block into the same data as output
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        for xy in xy.iter_mut() {
            *xy = self.process(*xy);
        }
    }
}

impl<X: Copy, Y, T: Process<X, Y>> Process<X, Y> for &mut T {
    #[inline]
    fn process(&mut self, x: X) -> Y {
        (*self).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        (*self).block(x, y)
    }
}

impl<X: Copy, T: Inplace<X>> Inplace<X> for &mut T {
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        (*self).inplace(xy)
    }
}

/// Configuration-minor, sample-major
///
/// The various Process tooling implementations for `Minor`
/// place the sample loop as the outer-most loop (configuration-minor, sample-major).
/// This is optimal for filters with small or no state and configuration.
///
/// Chain of large filters are implemented through tuples and slices/arrays.
///
/// These optimize well (especially for sample arrays) if the sizes obey
/// configuration ~ state > sample.
/// If they do not, use `Minor`.
///
/// Note that the major implementations only override the behavior
/// for `block()` and `inplace()`. `process()` is unaffected.
#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Minor<C, U> {
    /// The inner configurations
    pub inner: C,
    /// The intermediate type
    intermediate: PhantomData<U>,
}

impl<C, U> Minor<C, U> {
    /// Create a new chain
    #[inline]
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            intermediate: PhantomData,
        }
    }
}

impl<C, U, const N: usize> Minor<[C; N], U> {
    /// Borrowed Self
    #[inline]
    pub fn as_ref(&self) -> Minor<&[C], U> {
        Minor::new(&self.inner[..])
    }

    /// Mutably borrowed Self
    #[inline]
    pub fn as_mut(&mut self) -> Minor<&mut [C], U> {
        Minor::new(&mut self.inner[..])
    }
}

impl<X: Copy, U: Copy, Y, P1: Process<X, U>, P2: Process<U, Y>> Process<X, Y>
    for Minor<(P1, P2), U>
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        self.inner.1.process(self.inner.0.process(x))
    }
}

impl<X: Copy, U, P1, P2> Inplace<X> for Minor<(P1, P2), U> where Self: Process<X> {}

impl<X: Copy, Y: Copy, P1: Process<X, Y>, P2: Inplace<Y>> Process<X, Y> for (P1, P2) {
    #[inline]
    fn process(&mut self, x: X) -> Y {
        self.1.process(self.0.process(x))
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.0.block(x, y);
        self.1.inplace(y);
    }
}

impl<X: Copy, P1: Inplace<X>, P2: Inplace<X>> Inplace<X> for (P1, P2) {
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        self.0.inplace(xy);
        self.1.inplace(xy);
    }
}

impl<X: Copy, P: Process<X>> Process<X> for Minor<&mut [P], X> {
    #[inline]
    fn process(&mut self, x: X) -> X {
        self.inner.iter_mut().fold(x, |x, p| p.process(x))
    }
}

impl<X: Copy, P> Inplace<X> for Minor<&mut [P], X> where Self: Process<X> {}

impl<X: Copy, P: Inplace<X>> Process<X> for [P] {
    #[inline]
    fn process(&mut self, x: X) -> X {
        Minor::new(&mut self[..]).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        debug_assert_eq!(x.len(), y.len());
        if let Some((p0, p)) = self.split_first_mut() {
            p0.block(x, y);
            for p in p.iter_mut() {
                p.inplace(y);
            }
        } else {
            y.copy_from_slice(x);
        }
    }
}

impl<X: Copy, P: Inplace<X>> Inplace<X> for [P] {
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        for p in self.iter_mut() {
            p.inplace(xy)
        }
    }
}

impl<X: Copy, P: Process<X>, const N: usize> Process<X> for Minor<[P; N], X> {
    #[inline]
    fn process(&mut self, x: X) -> X {
        self.as_mut().process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        self.as_mut().block(x, y)
    }
}

impl<X: Copy, P: Process<X>, const N: usize> Inplace<X> for Minor<[P; N], X> {
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        self.as_mut().inplace(xy)
    }
}

impl<X: Copy, P: Inplace<X>, const N: usize> Process<X> for [P; N] {
    #[inline]
    fn process(&mut self, x: X) -> X {
        self.as_mut().process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        self.as_mut().block(x, y)
    }
}

impl<X: Copy, P: Inplace<X>, const N: usize> Inplace<X> for [P; N] {
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        self.as_mut().inplace(xy)
    }
}

/// A stateful processor
#[derive(Debug, Clone, Default)]
pub struct Stateful<C, S> {
    /// Processor configuration
    pub config: C,
    /// Processor state
    pub state: S,
}

impl<C, S> Stateful<C, S> {
    /// Create a new stateful processor
    #[inline]
    pub fn new(config: C, state: S) -> Self {
        Self { config, state }
    }

    /// Obtain a borrowed processor
    ///
    /// Stateful `Process` is typically implemented on the borrowed processor.
    #[inline]
    pub fn as_mut(&mut self) -> Stateful<&C, &mut S> {
        Stateful {
            config: &self.config,
            state: &mut self.state,
        }
    }
}

/// Stateless marker
#[derive(Debug, Clone, Default)]
#[repr(transparent)]
pub struct Stateless<C>(pub C);

/// Stateless filters
impl<X, Y, C> Process<X, Y> for Stateful<&Stateless<C>, &mut ()>
where
    X: Copy,
    for<'a> &'a C: Process<X, Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        (&self.config.0).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        (&self.config.0).block(x, y)
    }
}

impl<X, C> Inplace<X> for Stateful<&Stateless<C>, &mut ()>
where
    X: Copy,
    for<'a> &'a C: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        (&self.config.0).inplace(xy)
    }
}

/// Configuration-less filters
impl<X, Y, S> Process<X, Y> for Stateful<&(), &mut S>
where
    X: Copy,
    for<'a> &'a mut S: Process<X, Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        self.state.process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.state.block(x, y)
    }
}

impl<X, S> Inplace<X> for Stateful<&(), &mut S>
where
    X: Copy,
    for<'a> &'a mut S: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        self.state.inplace(xy)
    }
}

/// A chain of two small filters of different type.
///
/// This then automatically covers any nested tuple.
///
/// Iterations are sample-major, and filter-minor. This is good if at least
/// one of the filters has no or small state and configuration.
///
/// For large filters/state, use straight tuples, arrays, and slices.
impl<X, U, Y, C1, C2, S1, S2> Process<X, Y> for Stateful<&Minor<(C1, C2), U>, &mut (S1, S2)>
where
    X: Copy,
    U: Copy,
    for<'a> Stateful<&'a C1, &'a mut S1>: Process<X, U>,
    for<'a> Stateful<&'a C2, &'a mut S2>: Process<U, Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        let u = Stateful::new(&self.config.inner.0, &mut self.state.0).process(x);
        Stateful::new(&self.config.inner.1, &mut self.state.1).process(u)
    }
}

impl<X, U, C1, C2, S1, S2> Inplace<X> for Stateful<&Minor<(C1, C2), U>, &mut (S1, S2)>
where
    X: Copy,
    Self: Process<X>,
{
}

/// Chain of two different large filters
impl<X, Y, C1, C2, S1, S2> Process<X, Y> for Stateful<&(C1, C2), &mut (S1, S2)>
where
    X: Copy,
    Y: Copy,
    for<'a> Stateful<&'a C1, &'a mut S1>: Process<X, Y>,
    for<'a> Stateful<&'a C2, &'a mut S2>: Inplace<Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        // TODO: defer to Minor
        let u = Stateful::new(&self.config.0, &mut self.state.0).process(x);
        Stateful::new(&self.config.1, &mut self.state.1).process(u)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        Stateful::new(&self.config.0, &mut self.state.0).block(x, y);
        Stateful::new(&self.config.1, &mut self.state.1).inplace(y);
    }
}

impl<X, C1, C2, S1, S2> Inplace<X> for Stateful<&(C1, C2), &mut (S1, S2)>
where
    X: Copy,
    for<'a> Stateful<&'a C1, &'a mut S1>: Inplace<X>,
    for<'a> Stateful<&'a C2, &'a mut S2>: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        Stateful::new(&self.config.0, &mut self.state.0).inplace(xy);
        Stateful::new(&self.config.1, &mut self.state.1).inplace(xy);
    }
}

/// A chain of multiple small filters of the same type
impl<X, C, S> Process<X> for Stateful<Minor<&[C], X>, &mut [S]>
where
    X: Copy,
    for<'a> Stateful<&'a C, &'a mut S>: Process<X>,
{
    #[inline]
    fn process(&mut self, x: X) -> X {
        debug_assert_eq!(self.config.inner.len(), self.state.len());
        self.config
            .inner
            .iter()
            .zip(self.state.iter_mut())
            .fold(x, |x, (f, s)| Stateful::new(f, s).process(x))
    }
}

impl<X, C, S> Inplace<X> for Stateful<Minor<&[C], X>, &mut [S]>
where
    X: Copy,
    Self: Process<X>,
{
}

/// A chain of multiple large filters of the same type
impl<X, C, S> Process<X> for Stateful<&[C], &mut [S]>
where
    X: Copy,
    for<'a> Stateful<&'a C, &'a mut S>: Inplace<X>,
{
    #[inline]
    fn process(&mut self, x: X) -> X {
        Stateful::new(Minor::new(self.config), &mut *self.state).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        debug_assert_eq!(self.config.len(), self.state.len());
        if let Some(((c0, c), (s0, s))) =
            self.config.split_first().zip(self.state.split_first_mut())
        {
            Stateful::new(c0, s0).block(x, y);
            for (c, s) in c.iter().zip(s) {
                Stateful::new(c, s).inplace(y);
            }
        } else {
            y.copy_from_slice(x);
        }
    }
}

impl<X, C, S> Inplace<X> for Stateful<&[C], &mut [S]>
where
    X: Copy,
    for<'a> Stateful<&'a C, &'a mut S>: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        debug_assert_eq!(self.config.len(), self.state.len());
        for (c, s) in self.config.iter().zip(self.state.iter_mut()) {
            Stateful::new(c, s).inplace(xy);
        }
    }
}

/// A chain of multiple small filters of the same type
impl<X, C, S, const N: usize> Process<X> for Stateful<&Minor<[C; N], X>, &mut [S; N]>
where
    X: Copy,
    for<'a> Stateful<Minor<&'a [C], X>, &'a mut [S]>: Process<X>,
{
    #[inline]
    fn process(&mut self, x: X) -> X {
        Stateful::new(self.config.as_ref(), self.state.as_mut()).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        Stateful::new(self.config.as_ref(), self.state.as_mut()).block(x, y)
    }
}

impl<X, C, S, const N: usize> Inplace<X> for Stateful<&Minor<[C; N], X>, &mut [S; N]>
where
    X: Copy,
    for<'a> Stateful<Minor<&'a [C], X>, &'a mut [S]>: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        Stateful::new(self.config.as_ref(), self.state.as_mut()).inplace(xy)
    }
}

/// A chain of multiple large filters of the same type
impl<X, C, S, const N: usize> Process<X> for Stateful<&[C; N], &mut [S; N]>
where
    X: Copy,
    for<'a> Stateful<&'a [C], &'a mut [S]>: Process<X>,
{
    #[inline]
    fn process(&mut self, x: X) -> X {
        Stateful::new(self.config.as_ref(), self.state.as_mut()).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        Stateful::new(self.config.as_ref(), self.state.as_mut()).block(x, y)
    }
}

impl<X, C, S, const N: usize> Inplace<X> for Stateful<&[C; N], &mut [S; N]>
where
    X: Copy,
    for<'a> Stateful<&'a [C], &'a mut [S]>: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        Stateful::new(self.config.as_ref(), self.state.as_mut()).inplace(xy)
    }
}

/// Fan out the same input to multiple processors
#[derive(Clone, Debug, Default)]
pub struct FanOut<C>(pub C);

impl<X, Y1, Y2, C1, C2> Process<X, (Y1, Y2)> for FanOut<(C1, C2)>
where
    X: Copy,
    C1: Process<X, Y1>,
    C2: Process<X, Y2>,
{
    fn process(&mut self, x: X) -> (Y1, Y2) {
        (self.0.0.process(x), self.0.1.process(x))
    }
}

impl<X, Y, C1, C2> Process<X, [Y; 2]> for FanOut<(C1, C2)>
where
    X: Copy,
    C1: Process<X, Y>,
    C2: Process<X, Y>,
{
    fn process(&mut self, x: X) -> [Y; 2] {
        [self.0.0.process(x), self.0.1.process(x)]
    }
}

impl<X, Y, C, const N: usize> Process<X, [Y; N]> for FanOut<[C; N]>
where
    X: Copy,
    [Y; N]: Default,
    C: Process<X, Y>,
{
    fn process(&mut self, x: X) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for (c, y) in self.0.iter_mut().zip(y.iter_mut()) {
            *y = c.process(x);
        }
        y
    }
}

impl<X, Y1, Y2, C1, C2, S1, S2> Process<X, (Y1, Y2)> for Stateful<&FanOut<(C1, C2)>, &mut (S1, S2)>
where
    X: Copy,
    for<'a> Stateful<&'a C1, &'a mut S1>: Process<X, Y1>,
    for<'a> Stateful<&'a C2, &'a mut S2>: Process<X, Y2>,
{
    #[inline]
    fn process(&mut self, x: X) -> (Y1, Y2) {
        (
            Stateful::new(&self.config.0.0, &mut self.state.0).process(x),
            Stateful::new(&self.config.0.1, &mut self.state.1).process(x),
        )
    }
}

impl<X, Y, C1, C2, S1, S2> Process<X, [Y; 2]> for Stateful<&FanOut<(C1, C2)>, &mut (S1, S2)>
where
    X: Copy,
    for<'a> Stateful<&'a C1, &'a mut S1>: Process<X, Y>,
    for<'a> Stateful<&'a C2, &'a mut S2>: Process<X, Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> [Y; 2] {
        [
            Stateful::new(&self.config.0.0, &mut self.state.0).process(x),
            Stateful::new(&self.config.0.1, &mut self.state.1).process(x),
        ]
    }
}

impl<X, Y, C, S, const N: usize> Process<X, [Y; N]> for Stateful<&FanOut<[C; N]>, &mut [S; N]>
where
    X: Copy,
    [Y; N]: Default,
    for<'a> Stateful<&'a C, &'a mut S>: Process<X, Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((c, s), y) in self
            .config
            .0
            .iter()
            .zip(self.state.iter_mut())
            .zip(y.iter_mut())
        {
            *y = Stateful::new(c, s).process(x);
        }
        y
    }
}

/// Summ outputs of two filters
///
/// Fan in.
#[derive(Debug, Clone, Default)]
pub struct Sum;
impl<X: Copy + Add<Output = X>> Process<[X; 2], X> for &Sum {
    #[inline]
    fn process(&mut self, x: [X; 2]) -> X {
        x[0] + x[1]
    }
}

/// Difference of outputs of two filters
///
/// Fan in.
#[derive(Debug, Clone, Default)]
pub struct Diff;
impl<X: Copy + Sub<Output = X>> Process<[X; 2], X> for &Diff {
    #[inline]
    fn process(&mut self, x: [X; 2]) -> X {
        x[0] - x[1]
    }
}

/// Sum and difference of outputs of two filters
#[derive(Debug, Clone, Default)]
pub struct Butterfly;
impl<X: Copy + Add<Output = X> + Sub<Output = X>> Process<[X; 2], [X; 2]> for &Butterfly {
    #[inline]
    fn process(&mut self, x: [X; 2]) -> [X; 2] {
        [x[0] + x[1], x[0] - x[1]]
    }
}

impl<X: Copy> Inplace<X> for &Butterfly where Self: Process<X> {}

/// Parallel filter pair
///
/// This can be viewed as digital lattice filter or butterfly filter or complementary allpass pair
/// or polyphase interpolator.
/// Candidates for the branches are allpasses like Wdf or Ldi, polyphase banks for resampling or Hilbert filters.
///
/// Potentially required scaling with 0.5 gain is to be performed ahead of the filter or within each branch.
///
/// This uses the default sample-major implementation
/// and may lead to suboptimal cashing and register thrashing for large branches.
/// To avoid this, use `block()` and `inplace()` on a scratch buffer (input or output).
pub type Pair<C1, C2, X> = Minor<(FanOut<(C1, C2)>, Stateless<Sum>), [X; 2]>;

/// Multiple channels to be processed with the same configuration
#[derive(Clone, Debug)]
pub struct Channels<S, const N: usize>(pub [S; N]);

impl<S, const N: usize> Default for Channels<S, N>
where
    [S; N]: Default,
{
    fn default() -> Self {
        Self(Default::default())
    }
}

/// Process samples from multiple channels with a common configuration
///
/// Note that block() and inplace() reinterpret the data as transposed: __not__ as `[[X; N]]` but as `[[X]; N]`.
impl<X, Y, C, S, const N: usize> Process<[X; N], [Y; N]> for Stateful<&C, &mut Channels<S, N>>
where
    X: Copy,
    [Y; N]: Default,
    for<'a> Stateful<&'a C, &'a mut S>: Process<X, Y>,
{
    #[inline]
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((x, y), state) in x.into_iter().zip(y.iter_mut()).zip(self.state.0.iter_mut()) {
            *y = Stateful::new(self.config, state).process(x);
        }
        y
    }

    #[inline]
    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; N]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        for ((x, y), state) in x
            .as_flattened()
            .chunks_exact(n)
            .zip(y.as_flattened_mut().chunks_exact_mut(n))
            .zip(self.state.0.iter_mut())
        {
            Stateful::new(self.config, state).block(x, y)
        }
    }
}

impl<X, C, S, const N: usize> Inplace<[X; N]> for Stateful<&C, &mut Channels<S, N>>
where
    X: Copy,
    [X; N]: Default,
    for<'a> Stateful<&'a C, &'a mut S>: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        let n = xy.len();
        for (xy, state) in xy
            .as_flattened_mut()
            .chunks_exact_mut(n)
            .zip(self.state.0.iter_mut())
        {
            Stateful::new(self.config, state).inplace(xy)
        }
    }
}

/// Identity using [`Copy`]
#[derive(Debug, Clone, Default)]
pub struct Identity;
impl<T: Copy> Process<T> for &Identity {
    #[inline]
    fn process(&mut self, x: T) -> T {
        x
    }

    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        y.copy_from_slice(x);
    }
}

impl<T: Copy> Inplace<T> for &Identity {
    #[inline]
    fn inplace(&mut self, _xy: &mut [T]) {}
}

/// Inversion using [`Neg`].
#[derive(Debug, Clone, Default)]
pub struct Invert;
impl<T: Copy + Neg<Output = T>> Process<T> for &Invert {
    #[inline]
    fn process(&mut self, x: T) -> T {
        x.neg()
    }
}

impl<T: Copy> Inplace<T> for &Invert where Self: Process<T> {}

/// Fixed point with F fractional bits.
///
/// * Q<i32, 32> is [-0.5, 0.5[
/// * Q<i16, 15> is [-1, 1[
/// * Q<u8, 4> is [0, 16-1/16]
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct Q<T, const F: u8>(pub T);

macro_rules! impl_mul_q {
    ($q:ty, $t:ty, $a:ty) => {
        /// Multiplication with truncation
        impl<const F: u8> Mul<$t> for Q<$q, F> {
            type Output = $t;

            #[inline]
            fn mul(self, rhs: $t) -> Self::Output {
                ((self.0 as $a * rhs as $a) >> F) as _
            }
        }
    };
}
impl_mul_q!(i8, i8, i16);
impl_mul_q!(i16, i16, i32);
impl_mul_q!(i32, i32, i64);
impl_mul_q!(i64, i64, i128);

/// Addition of a constant
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct Adder<T>(pub T);

/// Offset using `Add`
impl<T: Copy, G: Add<T, Output = T> + Copy> Process<T> for &Adder<G> {
    #[inline]
    fn process(&mut self, x: T) -> T {
        self.0 + x
    }
}

impl<T: Copy, G> Inplace<T> for &Adder<G> where Self: Process<T> {}

/// Gain using `Mul`
impl<T, G, const F: u8> Process<T> for &Q<G, F>
where
    Q<G, F>: Mul<T, Output = T> + Copy,
    T: Copy,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        **self * x
    }
}

impl<T: Copy, G, const F: u8> Inplace<T> for &Q<G, F> where Self: Process<T> {}

/// Clamp between min and max using `Ord`
#[derive(Debug, Clone, Default)]
pub struct Clamp<T> {
    /// Lowest output value
    pub min: T,
    /// Highest output value
    pub max: T,
}

impl<T> Process<T> for &Clamp<T>
where
    T: Copy + Ord,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        x.clamp(self.min, self.max)
    }
}

impl<T: Copy> Inplace<T> for &Clamp<T> where Self: Process<T> {}

/// Decimate or zero stuff
pub struct Rate;
impl<X, const N: usize> Process<[X; N], X> for &Rate
where
    [X; N]: Copy,
    X: Copy,
{
    #[inline]
    fn process(&mut self, x: [X; N]) -> X {
        x[N - 1]
    }
}

impl<X, const N: usize> Process<X, [X; N]> for &Rate
where
    [X; N]: Default,
    X: Copy,
{
    #[inline]
    fn process(&mut self, x: X) -> [X; N] {
        let mut y = <[X; N]>::default();
        y[0] = x;
        y
    }
}

/// Buffer input or output, or fixed delay line
#[derive(Debug, Clone)]
pub struct Buffer<X, const N: usize> {
    buffer: [X; N],
    idx: usize,
}

impl<X, const N: usize> Buffer<X, N> {
    /// The buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.idx == 0
    }
}

impl<X, const N: usize> Default for Buffer<X, N>
where
    [X; N]: Default,
{
    fn default() -> Self {
        Self {
            buffer: Default::default(),
            idx: 0,
        }
    }
}

impl<X: Copy, const N: usize> Process<X> for Buffer<X, N> {
    #[inline]
    fn process(&mut self, x: X) -> X {
        self.idx = (self.idx + 1) % N;
        core::mem::replace(&mut self.buffer[self.idx], x)
    }

    // TODO: block(), inplace(), Process<[X; M]>
}

impl<X: Copy, const N: usize> Inplace<X> for Buffer<X, N> where Self: Process<X> {}

impl<X: Copy, const N: usize> Process<X, Option<[X; N]>> for Buffer<X, N> {
    #[inline]
    fn process(&mut self, x: X) -> Option<[X; N]> {
        self.buffer[self.idx] = x;
        self.idx += 1;
        (self.idx == N).then(|| {
            self.idx = 0;
            self.buffer
        })
    }

    // TODO: block()
}

impl<X: Copy, const N: usize> Process<Option<[X; N]>, X> for Buffer<X, N> {
    #[inline]
    fn process(&mut self, x: Option<[X; N]>) -> X {
        if let Some(x) = x {
            self.buffer = x;
            self.idx = 0;
        } else {
            self.idx += 1;
        }
        self.buffer[self.idx]
    }

    // TODO: block()
}

// TODO: Nyquist, Integrator, Derivative

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn stateless() {
        assert_eq!((&Identity).process(3), 3);
        assert_eq!(Stateful::new(Stateless(Invert), ()).as_mut().process(9), -9);
        assert_eq!((&Q::<i32, 3>(32)).process(9), 9 * 4);
        assert_eq!((&Adder(7)).process(9), 7 + 9);
        assert_eq!(Minor::new((&Adder(7), &Adder(1))).process(9), 7 + 1 + 9);
        let mut xy = [3, 0, 0];
        let mut dly = Buffer::<_, 2>::default();
        dly.inplace(&mut xy);
        assert_eq!(xy, [0, 0, 3]);
        let y: i32 = Stateful::new(&(), &mut dly).process(4);
        assert_eq!(y, 0);
    }
}
