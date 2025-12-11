//! Sample processing, filtering, combination of filters.
use core::ops::{Add, AddAssign, Mul, Neg, SubAssign};

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
/// through [`ProcessorRef<'_, C, S>`] (created ad-hoc from by borrowing configuration and state)
/// or [`Processor<C, S>`] (owned configuration and state).
/// Stateless filters should implement `Process` on `ProcessorRef<'_, C, S=()>` for composability.
///
/// Tuples, arrays, and Pairs, and Chain can be mixed and nested ad lib. The corresponding
/// [`Processor`]/[`ProcessorRef`] will implement `Process`.
///
/// And the same configuration they can be applied to multiple [`Channels`].
pub trait Process<T> {
    /// Update the state with a new input sample and obtain an output sample
    fn process(&mut self, x: &T) -> T;

    /// Process a block of samples
    ///
    /// Input and output should be of the same size.
    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(y) {
            *y = self.process(x);
        }
    }

    /// Process a block of samples in place
    #[inline]
    fn in_place(&mut self, xy: &mut [T]) {
        for xy in xy {
            *xy = self.process(xy);
        }
    }

    /// Process a block of samples and return only the last
    ///
    /// The input block size should match the design decimation
    /// ratio of the process.
    #[inline]
    fn decimate(&mut self, x: &[T]) -> Option<T> {
        x.iter().map(|x| self.process(x)).last()
    }

    /// Process one sample into a block of multiple samples
    ///
    /// The default implementation only stores a single output sample.
    /// The remaining output samples must explicitly be set.
    /// If the output slice is empty, the output sample is discarded.
    /// The output block size should match the design interpolation
    /// ratio of the process.
    #[inline]
    fn interpolate(&mut self, x: &T, y: &mut [T]) {
        let y0 = self.process(x);
        if let Some(y) = y.first_mut() {
            *y = y0;
        }
    }
}

/// Stateful processor by reference
#[derive(Debug)]
pub struct ProcessorRef<'a, C: ?Sized, S: ?Sized = ()> {
    /// Processor configuration
    pub config: &'a C,
    /// Processor state
    pub state: &'a mut S,
}

impl<'a, C: ?Sized, S: ?Sized> ProcessorRef<'a, C, S> {
    /// Create a new stateful processor by reference
    #[inline]
    pub fn new(config: &'a C, state: &'a mut S) -> Self {
        Self { config, state }
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
}

/// Owned configuration/state defers to [`ProcessorRef`]
impl<T, C, S> Process<T> for Processor<C, S>
where
    for<'a> ProcessorRef<'a, C, S>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        ProcessorRef::new(&self.config, &mut self.state).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        ProcessorRef::new(&self.config, &mut self.state).block(x, y)
    }

    #[inline]
    fn in_place(&mut self, xy: &mut [T]) {
        ProcessorRef::new(&self.config, &mut self.state).in_place(xy)
    }

    #[inline]
    fn decimate(&mut self, x: &[T]) -> Option<T> {
        ProcessorRef::new(&self.config, &mut self.state).decimate(x)
    }

    #[inline]
    fn interpolate(&mut self, x: &T, y: &mut [T]) {
        ProcessorRef::new(&self.config, &mut self.state).interpolate(x, y)
    }
}

/// Stateless filters
impl<C, T> Process<T> for ProcessorRef<'_, C>
where
    for<'a> &'a C: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        self.config.process(x)
    }

    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        self.config.block(x, y)
    }

    #[inline]
    fn in_place(&mut self, xy: &mut [T]) {
        self.config.in_place(xy)
    }

    #[inline]
    fn decimate(&mut self, x: &[T]) -> Option<T> {
        x.last().map(|x| self.config.process(x))
    }

    #[inline]
    fn interpolate(&mut self, x: &T, y: &mut [T]) {
        self.config.interpolate(x, y)
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
pub struct Chain<C: ?Sized>(pub C);

impl<C> Chain<C> {
    /// Create a new chain
    #[inline]
    pub fn new(c: C) -> Self {
        Self(c)
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
impl<T, C1, C2, S1, S2> Process<T> for ProcessorRef<'_, (C1, C2), (S1, S2)>
where
    for<'a> ProcessorRef<'a, C1, S1>: Process<T>,
    for<'a> ProcessorRef<'a, C2, S2>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        let u = ProcessorRef::new(&self.config.0, &mut self.state.0).process(x);
        ProcessorRef::new(&self.config.1, &mut self.state.1).process(&u)
    }
}

/// Chain of two different large filters
impl<T, C1, C2, S1, S2> Process<T> for ProcessorRef<'_, Chain<(C1, C2)>, (S1, S2)>
where
    for<'a> ProcessorRef<'a, C1, S1>: Process<T>,
    for<'a> ProcessorRef<'a, C2, S2>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        ProcessorRef::new(&self.config.0, self.state).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        ProcessorRef::new(&self.config.0.0, &mut self.state.0).block(x, y);
        ProcessorRef::new(&self.config.0.1, &mut self.state.1).in_place(y);
    }

    #[inline]
    fn in_place(&mut self, xy: &mut [T]) {
        ProcessorRef::new(&self.config.0.0, &mut self.state.0).in_place(xy);
        ProcessorRef::new(&self.config.0.1, &mut self.state.1).in_place(xy);
    }
}

impl<C1, C2, S1, S2> ProcessorRef<'_, Chain<(C1, C2)>, (S1, S2)> {
    /// Decimate using the input samples as scratch
    #[inline]
    pub fn decimate_in_place<T, const N: usize>(&mut self, xy: &mut [T; N]) -> Option<T>
    where
        for<'a> ProcessorRef<'a, C1, S1>: Process<T>,
        for<'a> ProcessorRef<'a, C2, S2>: Process<T>,
    {
        ProcessorRef::new(&self.config.0.0, &mut self.state.0).in_place(xy);
        ProcessorRef::new(&self.config.0.1, &mut self.state.1).decimate(xy)
    }
}

/// A chain of multiple small filters of the same type
impl<T, C, S> Process<T> for ProcessorRef<'_, [C], [S]>
where
    T: Copy,
    for<'a> ProcessorRef<'a, C, S>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        debug_assert_eq!(self.config.len(), self.state.len());
        self.config
            .iter()
            .zip(self.state.iter_mut())
            .fold(*x, |x, (f, s)| ProcessorRef::new(f, s).process(&x))
    }
}

/// A chain of multiple largs filters of the same type
impl<T, C, S> Process<T> for ProcessorRef<'_, Chain<[C]>, [S]>
where
    T: Copy,
    for<'a> ProcessorRef<'a, C, S>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        ProcessorRef::new(&self.config.0, self.state).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(self.config.0.len(), self.state.len());
        if let Some(((c0, c), (s0, s))) = self
            .config
            .0
            .split_first()
            .zip(self.state.split_first_mut())
        {
            ProcessorRef::new(c0, s0).block(x, y);
            for (c, s) in c.iter().zip(s) {
                ProcessorRef::new(c, s).in_place(y);
            }
        } else {
            y.copy_from_slice(x);
        }
    }

    #[inline]
    fn in_place(&mut self, xy: &mut [T]) {
        debug_assert_eq!(self.config.0.len(), self.state.len());
        for (c, s) in self.config.0.iter().zip(self.state.iter_mut()) {
            ProcessorRef::new(c, s).in_place(xy);
        }
    }
}

impl<C, S> ProcessorRef<'_, Chain<[C]>, [S]> {
    /// Decimate using the input samples as scratch
    #[inline]
    pub fn decimate_in_place<T, const N: usize>(&mut self, xy: &mut [T; N]) -> Option<T>
    where
        T: Copy,
        for<'a> ProcessorRef<'a, C, S>: Process<T>,
    {
        debug_assert_eq!(self.config.0.len(), self.state.len());
        if let Some(((c1, c), (s1, s))) =
            self.config.0.split_last().zip(self.state.split_last_mut())
        {
            for (c, s) in c.iter().zip(s) {
                ProcessorRef::new(c, s).in_place(xy);
            }
            ProcessorRef::new(c1, s1).decimate(xy)
        } else {
            xy.last().copied()
        }
    }
}

/// A chain of multiple small filters of the same type
impl<T, C, S, const N: usize> Process<T> for ProcessorRef<'_, [C; N], [S; N]>
where
    for<'a> ProcessorRef<'a, [C], [S]>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        ProcessorRef::new(&self.config[..], &mut self.state[..]).process(x)
    }
}

/// A chain of multiple large filters of the same type
impl<T, C, S, const N: usize> Process<T> for ProcessorRef<'_, Chain<[C; N]>, [S; N]>
where
    for<'a> ProcessorRef<'a, Chain<[C]>, [S]>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        let c: &Chain<[_]> = &*self.config;
        ProcessorRef::new(c, &mut self.state[..]).process(x)
    }

    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        let c: &Chain<[_]> = &*self.config;
        ProcessorRef::new(c, &mut self.state[..]).block(x, y)
    }

    #[inline]
    fn in_place(&mut self, xy: &mut [T]) {
        let c: &Chain<[_]> = &*self.config;
        ProcessorRef::new(c, &mut self.state[..]).in_place(xy)
    }

    #[inline]
    fn decimate(&mut self, x: &[T]) -> Option<T> {
        let c: &Chain<[_]> = &*self.config;
        ProcessorRef::new(c, &mut self.state[..]).decimate(x)
    }

    #[inline]
    fn interpolate(&mut self, x: &T, y: &mut [T]) {
        let c: &Chain<[_]> = &*self.config;
        ProcessorRef::new(c, &mut self.state[..]).interpolate(x, y)
    }
}

impl<C, S, const N: usize> ProcessorRef<'_, Chain<[C; N]>, [S; N]> {
    /// Decimate using the input samples as scratch
    #[inline]
    pub fn decimate_in_place<T, const M: usize>(&mut self, xy: &mut [T; M]) -> Option<T>
    where
        T: Copy,
        for<'a> ProcessorRef<'a, C, S>: Process<T>,
    {
        let c: &Chain<[_]> = &*self.config;
        ProcessorRef::new(c, &mut self.state[..]).decimate_in_place(xy)
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

/// Complementary allpass state
#[derive(Clone, Debug, Default)]
pub struct PairState<S1, S2>(
    /// Top filter state
    pub S1,
    /// Bottom filter state
    pub S2,
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
impl<T, C1, C2, S1, S2> Process<T> for ProcessorRef<'_, Pair<C1, C2>, PairState<S1, S2>>
where
    T: Add<Output = T>,
    for<'a> ProcessorRef<'a, C1, S1>: Process<T>,
    for<'a> ProcessorRef<'a, C2, S2>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        ProcessorRef::new(&self.config.0, &mut self.state.0).process(x)
            + ProcessorRef::new(&self.config.1, &mut self.state.1).process(x)
    }

    #[inline]
    fn decimate(&mut self, x: &[T]) -> Option<T> {
        ProcessorRef::new(&self.config.0, &mut self.state.0)
            .decimate(x)
            .zip(ProcessorRef::new(&self.config.1, &mut self.state.1).decimate(x))
            .map(|(y0, y1)| y0 + y1)
    }

    #[inline]
    fn interpolate(&mut self, x: &T, y: &mut [T]) {
        let y0 = ProcessorRef::new(&self.config.0, &mut self.state.0).process(x);
        let y1 = ProcessorRef::new(&self.config.1, &mut self.state.1).process(x);
        if let Some(y) = y.first_chunk_mut() {
            *y = [y0, y1];
        }
    }
}

impl<C1, C2, S1, S2> ProcessorRef<'_, Pair<C1, C2>, PairState<S1, S2>> {
    /// Process a block into the sum of the two branches
    pub fn process_sum_block<T, const N: usize>(&mut self, x: &[T; N], y: &mut [T; N])
    where
        T: AddAssign,
        [T; N]: Default,
        for<'a> ProcessorRef<'a, C1, S1>: Process<T>,
        for<'a> ProcessorRef<'a, C2, S2>: Process<T>,
    {
        ProcessorRef::new(&self.config.0, &mut self.state.0).block(x, y);
        let mut y1 = <[T; N]>::default();
        ProcessorRef::new(&self.config.1, &mut self.state.1).block(x, &mut y1);
        for (y0, y1) in y.iter_mut().zip(y1) {
            *y0 += y1;
        }
    }

    /// Process a block into the difference of the two branches
    pub fn process_diff_block<T, const N: usize>(&mut self, x: &[T; N], y: &mut [T; N])
    where
        T: SubAssign,
        [T; N]: Default,
        for<'a> ProcessorRef<'a, C1, S1>: Process<T>,
        for<'a> ProcessorRef<'a, C2, S2>: Process<T>,
    {
        ProcessorRef::new(&self.config.0, &mut self.state.0).block(x, y);
        let mut y1 = <[T; N]>::default();
        ProcessorRef::new(&self.config.1, &mut self.state.1).block(x, &mut y1);
        for (y0, y1) in y.iter_mut().zip(y1) {
            *y0 -= y1;
        }
    }

    /// Process a block into the sum of the two branches
    pub fn process_sum_in_place<T, const N: usize>(&mut self, xy: &mut [T; N])
    where
        T: AddAssign,
        [T; N]: Clone,
        for<'a> ProcessorRef<'a, C1, S1>: Process<T>,
        for<'a> ProcessorRef<'a, C2, S2>: Process<T>,
    {
        let mut xy1 = xy.clone();
        ProcessorRef::new(&self.config.1, &mut self.state.1).in_place(&mut xy1);
        ProcessorRef::new(&self.config.0, &mut self.state.0).in_place(xy);
        for (y0, y1) in xy.iter_mut().zip(xy1) {
            *y0 += y1;
        }
    }

    /// Process a block into the difference of the two branches
    pub fn process_diff_in_place<T, const N: usize>(&mut self, xy: &mut [T; N])
    where
        T: SubAssign,
        [T; N]: Clone,
        for<'a> ProcessorRef<'a, C1, S1>: Process<T>,
        for<'a> ProcessorRef<'a, C2, S2>: Process<T>,
    {
        let mut xy1 = xy.clone();
        ProcessorRef::new(&self.config.1, &mut self.state.1).in_place(&mut xy1);
        ProcessorRef::new(&self.config.0, &mut self.state.0).in_place(xy);
        for (y0, y1) in xy.iter_mut().zip(xy1) {
            *y0 -= y1;
        }
    }
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
/// but require indexing/striding and the layout still is transposed relative to channel-major blocks.
///
/// To avoid a potential penalty, implement the channel loop explicitly on channel-major data.
///
/// TODO: play with as_chunks::<N>()
impl<T, C, S, const N: usize> Process<[T; N]> for ProcessorRef<'_, C, Channels<S, N>>
where
    [T; N]: Default,
    for<'a> ProcessorRef<'a, C, S>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: &[T; N]) -> [T; N] {
        let mut y = <[T; N]>::default();
        for ((x, y), state) in x.iter().zip(y.iter_mut()).zip(self.state.0.iter_mut()) {
            *y = ProcessorRef::new(self.config, state).process(x);
        }
        y
    }

    #[inline]
    fn block(&mut self, x: &[[T; N]], y: &mut [[T; N]]) {
        debug_assert_eq!(x.len(), y.len());
        for (i, state) in self.state.0.iter_mut().enumerate() {
            for (x, y) in x.iter().zip(y.iter_mut()) {
                y[i] = ProcessorRef::new(self.config, state).process(&x[i]);
            }
        }
    }

    #[inline]
    fn in_place(&mut self, xy: &mut [[T; N]]) {
        for (i, state) in self.state.0.iter_mut().enumerate() {
            for xy in xy.iter_mut() {
                xy[i] = ProcessorRef::new(self.config, state).process(&xy[i]);
            }
        }
    }

    #[inline]
    fn decimate(&mut self, x: &[[T; N]]) -> Option<[T; N]> {
        // https://github.com/rust-lang/rust/issues/130828 and https://github.com/rust-lang/rust/issues/79711
        let mut y = <[T; N]>::default();
        for (i, (y, state)) in y.iter_mut().zip(self.state.0.iter_mut()).enumerate() {
            *y = x
                .iter()
                .map(|x| ProcessorRef::new(self.config, state).process(&x[i]))
                .last()?;
        }
        Some(y)
    }
}

/// Identity using [`Copy`]
#[derive(Debug, Clone, Default)]
pub struct Identity;
impl<T> Process<T> for &Identity
where
    T: Copy,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        *x
    }

    #[inline]
    fn block(&mut self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), y.len());
        y.copy_from_slice(x);
    }

    #[inline]
    fn in_place(&mut self, _xy: &mut [T]) {}
}

/// Inversion using [`Neg`].
#[derive(Debug, Clone, Default)]
pub struct Invert;
impl<T> Process<T> for &Invert
where
    T: Copy + Neg<Output = T>,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
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
impl<T, G> Process<T> for &Adder<G>
where
    G: Add<T, Output = T> + Copy,
    T: Copy,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        self.0 + *x
    }
}

/// Gain using `Mul`
impl<T, G, const F: u8> Process<T> for &Q<G, F>
where
    Q<G, F>: Mul<T, Output = T> + Copy,
    T: Copy,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        **self * *x
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

impl<T> Process<T> for &Clamp<T>
where
    T: Copy + Ord,
{
    #[inline]
    fn process(&mut self, x: &T) -> T {
        (*x).clamp(self.min, self.max)
    }
}

// TODO: Delay, Nyquist, Integrator, Derivative

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn stateless() {
        assert_eq!((&Identity).process(&3), 3);
        assert_eq!(Processor::new(Invert, ()).process(&9), -9);
        assert_eq!((&Q::<i32, 3>(32)).process(&9), 9 * 4);
        assert_eq!((&Adder(7)).process(&9), 7 + 9);
        assert_eq!(
            Processor::new((Adder(7), Adder(1)), ((), ())).process(&9),
            7 + 1 + 9
        );
    }
}
