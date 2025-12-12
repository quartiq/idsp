//! Sample processing, filtering, combination of filters.
use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg};

/// Processing block
///
/// Single input, single output
///
/// Process impls can be cascaded in (homogeneous) `[C; N]` arrays/`[C]` slices, and heterogeneous
/// `(C1, C2)` tuples. They can be used in [`Pair`]s on complementary allpasses and polyphase banks.
/// Tuples and arrays process blocks of samples as sample-major (sample iteration is outer loop).
/// Use [`Chain<(C1, C2)>`], `Chain<[C; N]>` for cascadeds of large filters
/// (large state and/or configuration compared to a single sample) to process blocks sample-minor.
///
/// For a given filter configuration `C` and state `S` pair the trait is usually implemented
/// through [`Processor<&'a C, S>`] (created ad-hoc from by borrowing configuration and state)
/// or [`Processor<C, S>`] (owned configuration and state).
/// Stateless filters should implement `Process` on `Processor<&'a C, S=()>` for composability.
///
/// Tuples, arrays, and Pairs, and Chain can be mixed and nested ad lib. The corresponding
/// [`Processor`]/[`ProcessorRef`] will implement `Process`.
///
/// And the same configuration they can be applied to multiple [`Channels`].
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

/// Chain of large filters
///
/// The various Process tooling implementations for arrays, slices and tuples
/// place the sample loop as the outer-most loop (sample-major). This is optimal for
/// filters with small state and configuration.
///
/// Tho optimize well (especially for arrays) if the sizes obey
/// configuration > state > sample, use `Chain`.
///
/// Note that this only overrides the behavior for `block()` and `in_place()`.
/// `process()`, `interpolate()`, but especially `decimate()` are unaffected.
/// Use inherent `decimate_in_place()` for this.
#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Major<C> {
    /// The inner configurations
    pub inner: C,
}

impl<C> Major<C> {
    /// Create a new chain
    #[inline]
    pub fn new(inner: C) -> Self {
        Self { inner }
    }
}

impl<C, const N: usize> Major<[C; N]> {
    /// Borrowed Self
    #[inline]
    pub fn as_ref(&self) -> Major<&[C]> {
        Major::new(&self.inner[..])
    }

    /// Mutably borrowed Self
    #[inline]
    pub fn as_mut(&mut self) -> Major<&mut [C]> {
        Major::new(&mut self.inner[..])
    }
}

/// Configuration-minor, sample-major
#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Minor<C, U> {
    intermediate: PhantomData<U>,
    /// The inner configurations
    pub inner: C,
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

impl<X: Copy, U: Copy, P1: Process<X, U>, P2: Process<U, X>> Inplace<X> for Minor<(P1, P2), U> {}

impl<X: Copy, Y: Copy, P1: Process<X, Y>, P2: Inplace<Y>> Process<X, Y> for Major<(P1, P2)> {
    #[inline]
    fn process(&mut self, x: X) -> Y {
        // TODO: defer to Minor
        self.inner.1.process(self.inner.0.process(x))
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.inner.0.block(x, y);
        self.inner.1.inplace(y);
    }
}

impl<X: Copy, P1: Inplace<X>, P2: Inplace<X>> Inplace<X> for Major<(P1, P2)> {
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        self.inner.0.inplace(xy);
        self.inner.1.inplace(xy);
    }
}

impl<X: Copy, P: Process<X>> Process<X> for Minor<&mut [P], X> {
    #[inline]
    fn process(&mut self, x: X) -> X {
        self.inner.iter_mut().fold(x, |x, p| p.process(x))
    }
}

impl<X: Copy, P: Process<X>> Inplace<X> for Minor<&mut [P], X> {}

impl<X: Copy, P: Inplace<X>> Process<X> for Major<&mut [P]> {
    #[inline]
    fn process(&mut self, x: X) -> X {
        Minor::new(&mut *self.inner).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        debug_assert_eq!(x.len(), y.len());
        if let Some((p0, p)) = self.inner.split_first_mut() {
            p0.block(x, y);
            for p in p.iter_mut() {
                p.inplace(y);
            }
        } else {
            y.copy_from_slice(x);
        }
    }
}

impl<X: Copy, P: Inplace<X>> Inplace<X> for Major<&mut [P]> {
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        for p in self.inner.iter_mut() {
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
    fn inplace(&mut self, xy: &mut [X]) {
        self.as_mut().inplace(xy)
    }
}

impl<X: Copy, P: Inplace<X>, const N: usize> Process<X> for Major<[P; N]> {
    #[inline]
    fn process(&mut self, x: X) -> X {
        self.as_mut().process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        self.as_mut().block(x, y)
    }
}

impl<X: Copy, P: Inplace<X>, const N: usize> Inplace<X> for Major<[P; N]> {
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        self.as_mut().inplace(xy)
    }
}

/// A stateful processor
#[derive(Debug, Clone, Default)]
pub struct Processor<C, S = ()> {
    /// Processor configuration
    pub config: C,
    /// Processor state
    pub state: S,
}

impl<C, S> Processor<C, S> {
    /// Create a new stateful processor
    #[inline]
    pub fn new(config: C, state: S) -> Self {
        Self { config, state }
    }

    /// Obtain a borrowed processor
    ///
    /// Stateful `Process` is typically implemented on the borrowed processor.
    #[inline]
    pub fn as_mut(&mut self) -> Processor<&C, &mut S> {
        Processor {
            config: &self.config,
            state: &mut self.state,
        }
    }
}

/// Stateless filters
impl<X, Y, C> Process<X, Y> for Processor<C>
where
    X: Copy,
    C: Process<X, Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        self.config.process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.config.block(x, y)
    }
}

impl<X, C> Inplace<X> for Processor<C>
where
    X: Copy,
    C: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        self.config.inplace(xy)
    }
}

/// A chain of two small filters of different type.
///
/// This then automatically covers any nested tuple.
///
/// Iterations are sample-major, and filter-minor. This is good if at least
/// one of the filters with has no or small state and configuration.
///
/// For large filters/state, use [`Chain`].
impl<X, U, Y, C1, C2, S1, S2> Process<X, Y> for Processor<&Minor<(C1, C2), U>, &mut (S1, S2)>
where
    X: Copy,
    U: Copy,
    for<'a> Processor<&'a C1, &'a mut S1>: Process<X, U>,
    for<'a> Processor<&'a C2, &'a mut S2>: Process<U, Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        let u = Processor::new(&self.config.inner.0, &mut self.state.0).process(x);
        Processor::new(&self.config.inner.1, &mut self.state.1).process(u)
    }
}

impl<X, U, C1, C2, S1, S2> Inplace<X> for Processor<&Minor<(C1, C2), U>, &mut (S1, S2)>
where
    X: Copy,
    U: Copy,
    for<'a> Processor<&'a C1, &'a mut S1>: Process<X, U>,
    for<'a> Processor<&'a C2, &'a mut S2>: Process<U, X>,
{
}

/// Chain of two different large filters
impl<X, Y, C1, C2, S1, S2> Process<X, Y> for Processor<&Major<(C1, C2)>, &mut (S1, S2)>
where
    X: Copy,
    Y: Copy,
    for<'a> Processor<&'a C1, &'a mut S1>: Process<X, Y>,
    for<'a> Processor<&'a C2, &'a mut S2>: Inplace<Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        // TODO: defer to Minor
        let u = Processor::new(&self.config.inner.0, &mut self.state.0).process(x);
        Processor::new(&self.config.inner.1, &mut self.state.1).process(u)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        Processor::new(&self.config.inner.0, &mut self.state.0).block(x, y);
        Processor::new(&self.config.inner.1, &mut self.state.1).inplace(y);
    }
}

impl<X, C1, C2, S1, S2> Inplace<X> for Processor<&Major<(C1, C2)>, &mut (S1, S2)>
where
    X: Copy,
    for<'a> Processor<&'a C1, &'a mut S1>: Inplace<X>,
    for<'a> Processor<&'a C2, &'a mut S2>: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        Processor::new(&self.config.inner.0, &mut self.state.0).inplace(xy);
        Processor::new(&self.config.inner.1, &mut self.state.1).inplace(xy);
    }
}

/// A chain of multiple small filters of the same type
impl<X, C, S> Process<X> for Processor<Minor<&[C], X>, &mut [S]>
where
    X: Copy,
    for<'a> Processor<&'a C, &'a mut S>: Process<X>,
{
    #[inline]
    fn process(&mut self, x: X) -> X {
        debug_assert_eq!(self.config.inner.len(), self.state.len());
        self.config
            .inner
            .iter()
            .zip(self.state.iter_mut())
            .fold(x, |x, (f, s)| Processor::new(f, s).process(x))
    }
}

impl<X, C, S> Inplace<X> for Processor<Minor<&[C], X>, &mut [S]>
where
    X: Copy,
    for<'a> Processor<&'a C, &'a mut S>: Inplace<X>,
{
}

/// A chain of multiple large filters of the same type
impl<X, C, S> Process<X> for Processor<Major<&[C]>, &mut [S]>
where
    X: Copy,
    for<'a> Processor<&'a C, &'a mut S>: Inplace<X>,
{
    #[inline]
    fn process(&mut self, x: X) -> X {
        Processor::new(Minor::new(self.config.inner), &mut *self.state).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        debug_assert_eq!(self.config.inner.len(), self.state.len());
        if let Some(((c0, c), (s0, s))) = self
            .config
            .inner
            .split_first()
            .zip(self.state.split_first_mut())
        {
            Processor::new(c0, s0).block(x, y);
            for (c, s) in c.iter().zip(s) {
                Processor::new(c, s).inplace(y);
            }
        } else {
            y.copy_from_slice(x);
        }
    }
}

impl<X, C, S> Inplace<X> for Processor<Major<&[C]>, &mut [S]>
where
    X: Copy,
    for<'a> Processor<&'a C, &'a mut S>: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        debug_assert_eq!(self.config.inner.len(), self.state.len());
        for (c, s) in self.config.inner.iter().zip(self.state.iter_mut()) {
            Processor::new(c, s).inplace(xy);
        }
    }
}

/// A chain of multiple small filters of the same type
impl<X, C, S, const N: usize> Process<X> for Processor<&Minor<[C; N], X>, &mut [S; N]>
where
    X: Copy,
    for<'a> Processor<Minor<&'a [C], X>, &'a mut [S]>: Process<X>,
{
    #[inline]
    fn process(&mut self, x: X) -> X {
        Processor::new(self.config.as_ref(), &mut self.state[..]).process(x)
    }

    fn block(&mut self, x: &[X], y: &mut [X]) {
        Processor::new(self.config.as_ref(), &mut self.state[..]).block(x, y)
    }
}

impl<X, C, S, const N: usize> Inplace<X> for Processor<&Minor<[C; N], X>, &mut [S; N]>
where
    X: Copy,
    for<'a> Processor<Minor<&'a [C], X>, &'a mut [S]>: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [X]) {
        Processor::new(self.config.as_ref(), &mut self.state[..]).inplace(xy)
    }
}

/// A chain of multiple large filters of the same type
impl<X, C, S, const N: usize> Process<X> for Processor<&Major<[C; N]>, &mut [S; N]>
where
    X: Copy,
    for<'a> Processor<Major<&'a [C]>, &'a mut [S]>: Process<X>,
{
    #[inline]
    fn process(&mut self, x: X) -> X {
        Processor::new(self.config.as_ref(), &mut self.state[..]).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[X], y: &mut [X]) {
        Processor::new(self.config.as_ref(), &mut self.state[..]).block(x, y)
    }
}

impl<X, C, S, const N: usize> Inplace<X> for Processor<&Major<[C; N]>, &mut [S; N]>
where
    X: Copy,
    for<'a> Processor<Major<&'a [C]>, &'a mut [S]>: Inplace<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [X]) {
        Processor::new(self.config.as_ref(), &mut self.state[..]).inplace(xy)
    }
}

/// Parallel filter pair
///
/// This can be viewed as digital lattice filter or butterfly filter or complementary allpass pair
/// or polyphase interpolator.
/// Candidates for the branches are allpasses like Wdf or Ldi, polyphase banks for resampling or Hilbert filters.
#[derive(Clone, Debug, Default)]
pub struct Pair<C1, C2>(
    /// Top filter
    pub C1,
    /// Bottom filter
    pub C2,
);

/// Pair of parallel filter using the same input
///
/// `process` and `process_decimate` return the lowpass sum of the two branches.
/// Use an inverter in the bottom branch to form the highpass difference.
///
/// `process_interpolate` stores the (polyphase) branch outputs in the first two output samples.
/// Use this to obtain complementary/hilbert outputs (e.g. perform a `[y0 + y1, y0 - y1]` butterfly on it).
///
/// Potentially required scaling with 0.5 gain is to be performed ahead of the filter or within each branch.
///
/// `process_block` and `process_in_place` are use the default sample-major implementation
/// and may lead to suboptimal cashing and register thrashing. To alleviate this there are additional
/// branch-major inherent impls `process_{sum,diff}_{block,in_place}` for arrays.
impl<X, Y, C1, C2, S1, S2> Process<X, Y> for Processor<&Pair<C1, C2>, &mut (S1, S2)>
where
    Y: Add<Output = Y>,
    X: Copy,
    for<'a> Processor<&'a C1, &'a mut S1>: Process<X, Y>,
    for<'a> Processor<&'a C2, &'a mut S2>: Process<X, Y>,
{
    #[inline]
    fn process(&mut self, x: X) -> Y {
        Processor::new(&self.config.0, &mut self.state.0).process(x)
            + Processor::new(&self.config.1, &mut self.state.1).process(x)
    }
}

impl<X, C1, C2, S1, S2> Inplace<X> for Processor<&Pair<C1, C2>, &mut (S1, S2)>
where
    X: Copy + Add<Output = X>,
    for<'a> Processor<&'a C1, &'a mut S1>: Inplace<X>,
    for<'a> Processor<&'a C2, &'a mut S2>: Inplace<X>,
{
}

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
/// Note that this may result in suboptimal code because the data ordering is channel-minor.
/// The Process block loops are implemented channel-major to give better cache/register usage patterns
/// but they are not implemented configuration-major and require indexing/striding.
/// The layout still is transposed relative to channel-major blocks.
///
/// To avoid a potential penalty, implement the channel loop explicitly on channel-major data.
///
/// TODO: play with as_chunks::<N>()
impl<X, Y, C, S, const N: usize> Process<[X; N], [Y; N]> for Processor<&C, &mut Channels<S, N>>
where
    X: Copy,
    [Y; N]: Default,
    for<'a> Processor<&'a C, &'a mut S>: Process<X, Y>,
{
    #[inline]
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((x, y), state) in x.into_iter().zip(y.iter_mut()).zip(self.state.0.iter_mut()) {
            *y = Processor::new(self.config, state).process(x);
        }
        y
    }

    #[inline]
    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; N]]) {
        debug_assert_eq!(x.len(), y.len());
        for (i, state) in self.state.0.iter_mut().enumerate() {
            let mut p = Processor::new(self.config, state);
            for (x, y) in x.iter().zip(y.iter_mut()) {
                y[i] = p.process(x[i]);
            }
        }
    }
}

impl<X, C, S, const N: usize> Inplace<[X; N]> for Processor<&C, &mut Channels<S, N>>
where
    X: Copy,
    [X; N]: Default,
    for<'a> Processor<&'a C, &'a mut S>: Process<X>,
{
    #[inline]
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        for (i, state) in self.state.0.iter_mut().enumerate() {
            let mut p = Processor::new(self.config, state);
            for xy in xy.iter_mut() {
                xy[i] = p.process(xy[i]);
            }
        }
    }
}

/// Identity using [`Copy`]
#[derive(Debug, Clone, Default)]
pub struct Identity;
impl<T: Copy> Process<T> for Identity {
    #[inline]
    fn process(&mut self, x: T) -> T {
        x
    }

    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), y.len());
        y.copy_from_slice(x);
    }
}

impl<T: Copy> Inplace<T> for Identity {
    #[inline]
    fn inplace(&mut self, _xy: &mut [T]) {}
}

/// Inversion using [`Neg`].
#[derive(Debug, Clone, Default)]
pub struct Invert;
impl<T: Copy + Neg<Output = T>> Process<T> for Invert {
    #[inline]
    fn process(&mut self, x: T) -> T {
        x.neg()
    }
}

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
impl<T: Copy, G: Add<T, Output = T> + Copy> Process<T> for Adder<G> {
    #[inline]
    fn process(&mut self, x: T) -> T {
        self.0 + x
    }
}

/// Gain using `Mul`
impl<T, G, const F: u8> Process<T> for Q<G, F>
where
    Q<G, F>: Mul<T, Output = T> + Copy,
    T: Copy,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        *self * x
    }
}

/// Clamp between min and max using `Ord`
#[derive(Debug, Clone, Default)]
pub struct Clamp<T> {
    /// Lowest output value
    pub min: T,
    /// Highest output value
    pub max: T,
}

impl<T> Process<T> for Clamp<T>
where
    T: Copy + Ord,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        x.clamp(self.min, self.max)
    }
}

/// Decimate or zero stuff
pub struct Rate;
impl<X, const N: usize> Process<[X; N], X> for Rate
where
    [X; N]: Copy,
    X: Copy,
{
    #[inline]
    fn process(&mut self, x: [X; N]) -> X {
        x[N - 1]
    }
}

impl<X, const N: usize> Process<X, [X; N]> for Rate
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

// TODO: Delay, Nyquist, Integrator, Derivative

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn stateless() {
        assert_eq!((Identity).process(3), 3);
        assert_eq!(Processor::new(Invert, ()).process(9), -9);
        assert_eq!((Q::<i32, 3>(32)).process(9), 9 * 4);
        assert_eq!((Adder(7)).process(9), 7 + 9);
        assert_eq!(Minor::new((Adder(7), Adder(1))).process(9), 7 + 1 + 9);
    }
}
