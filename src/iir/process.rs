use core::ops::{Add, AddAssign, Sub, SubAssign};

/// Processing block
///
/// Single input, single output
pub trait Process<T: Copy> {
    /// Update the state with a new sample and obtain an output sample
    fn process(&mut self, x: T) -> T;

    /// Process a block of samples
    ///
    /// Input and output should be of the same size.
    #[inline]
    fn process_block(&mut self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(y) {
            *y = self.process(*x);
        }
    }

    /// Process a block of samples in place
    #[inline]
    fn process_in_place(&mut self, xy: &mut [T]) {
        for xy in xy {
            *xy = self.process(*xy);
        }
    }

    /// Process a block of samples and return only the last
    ///
    /// The input block size should match the design decimation
    /// ratio of the process.
    #[inline]
    fn process_decimate(&mut self, x: &[T]) -> Option<T> {
        x.iter().map(|x| self.process(*x)).last()
    }

    /// Process one sample into a block of multiple samples
    ///
    /// The default implementation only touches the first output sample.
    /// The remaining output samples must explicitly be set.
    /// If the output slice is empty, the output sample is discarded.
    /// The output block size should match the design intepolation
    /// ratio of the process.
    #[inline]
    fn process_interpolate(&mut self, x: T, y: &mut [T]) {
        let y0 = self.process(x);
        if let Some(y) = y.first_mut() {
            *y = y0;
        }
    }
}

/// Stateful processor by reference
#[derive(Debug)]
pub struct StatefulRef<'a, C: ?Sized, S: ?Sized> {
    /// Processor configuration
    pub config: &'a C,
    /// Processor state
    pub state: &'a mut S,
}

impl<'a, C: ?Sized, S: ?Sized> StatefulRef<'a, C, S> {
    /// Create a new stateful processor by reference
    #[inline]
    pub fn new(config: &'a C, state: &'a mut S) -> Self {
        Self { config, state }
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
}

impl<T, C, S> Process<T> for Stateful<C, S>
where
    T: Copy,
    for<'a> StatefulRef<'a, C, S>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        StatefulRef::new(&self.config, &mut self.state).process(x)
    }

    #[inline]
    fn process_block(&mut self, x: &[T], y: &mut [T]) {
        StatefulRef::new(&self.config, &mut self.state).process_block(x, y)
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [T]) {
        StatefulRef::new(&self.config, &mut self.state).process_in_place(xy)
    }

    #[inline]
    fn process_decimate(&mut self, x: &[T]) -> Option<T> {
        StatefulRef::new(&self.config, &mut self.state).process_decimate(x)
    }

    #[inline]
    fn process_interpolate(&mut self, x: T, y: &mut [T]) {
        StatefulRef::new(&self.config, &mut self.state).process_interpolate(x, y)
    }
}

/// A chain of multiple filters of the same type
impl<T, C, S> Process<T> for StatefulRef<'_, [C], [S]>
where
    T: Copy,
    for<'a> StatefulRef<'a, C, S>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        debug_assert_eq!(self.config.len(), self.state.len());
        self.config
            .iter()
            .zip(self.state.iter_mut())
            .fold(x, |x, (f, s)| StatefulRef::new(f, s).process(x))
    }

    #[inline]
    fn process_block(&mut self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(self.config.len(), self.state.len());
        if let Some(((f0, f1), (s0, s1))) =
            self.config.split_first().zip(self.state.split_first_mut())
        {
            StatefulRef::new(f0, s0).process_block(x, y);
            StatefulRef::new(f1, s1).process_in_place(y);
        } else {
            y.copy_from_slice(x);
        }
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [T]) {
        debug_assert_eq!(self.config.len(), self.state.len());
        for (f, s) in self.config.iter().zip(self.state.iter_mut()) {
            StatefulRef::new(f, s).process_in_place(xy);
        }
    }
}

/// A chain of multiple filters of the same type
impl<T, C, S, const N: usize> Process<T> for StatefulRef<'_, [C; N], [S; N]>
where
    T: Copy,
    for<'a> StatefulRef<'a, [C], [S]>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        StatefulRef::new(&self.config[..], &mut self.state[..]).process(x)
    }

    #[inline]
    fn process_block(&mut self, x: &[T], y: &mut [T]) {
        StatefulRef::new(&self.config[..], &mut self.state[..]).process_block(x, y)
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [T]) {
        StatefulRef::new(&self.config[..], &mut self.state[..]).process_in_place(xy)
    }

    #[inline]
    fn process_decimate(&mut self, x: &[T]) -> Option<T> {
        StatefulRef::new(&self.config[..], &mut self.state[..]).process_decimate(x)
    }

    #[inline]
    fn process_interpolate(&mut self, x: T, y: &mut [T]) {
        StatefulRef::new(&self.config[..], &mut self.state[..]).process_interpolate(x, y)
    }
}

/// A chain of two filters of different type.
///
/// This then automatically covers any nested tuple.
impl<T, C1, C2, S1, S2> Process<T> for StatefulRef<'_, (C1, C2), (S1, S2)>
where
    T: Copy,
    for<'a> StatefulRef<'a, C1, S1>: Process<T>,
    for<'a> StatefulRef<'a, C2, S2>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        let u = StatefulRef::new(&self.config.0, &mut self.state.0).process(x);
        StatefulRef::new(&self.config.1, &mut self.state.1).process(u)
    }

    #[inline]
    fn process_block(&mut self, x: &[T], y: &mut [T]) {
        StatefulRef::new(&self.config.0, &mut self.state.0).process_block(x, y);
        StatefulRef::new(&self.config.1, &mut self.state.1).process_in_place(y);
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [T]) {
        StatefulRef::new(&self.config.0, &mut self.state.0).process_in_place(xy);
        StatefulRef::new(&self.config.1, &mut self.state.1).process_in_place(xy);
    }
}

/// Parallel filter pair
///
/// Can be viewed as digital lattice filter or butterfly filter or complementary allpass pair.
/// Cadidates for the branches are allpasses like Wdf or Ldi or polyphase banks.
///
/// The [`Process::process`] and [`Process::process_decimate`] return the sum of the two branches
/// while intrinsic `process_complementary_in_place()` etc. yields both sum and difference.
///
/// [`Process::process_interpolate`] stores the (polyphase) branch outputs in the first two output samples.
///
/// Scaling with 0.5 gain is to be performed ahead of the filter or within both branches.
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

impl<T, C1, C2, S1, S2> Process<T> for StatefulRef<'_, Pair<C1, C2>, PairState<S1, S2>>
where
    T: Copy + Add<Output = T>,
    for<'a> StatefulRef<'a, C1, S1>: Process<T>,
    for<'a> StatefulRef<'a, C2, S2>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        StatefulRef::new(&self.config.0, &mut self.state.0).process(x)
            + StatefulRef::new(&self.config.1, &mut self.state.1).process(x)
    }

    #[inline]
    fn process_decimate(&mut self, x: &[T]) -> Option<T> {
        StatefulRef::new(&self.config.0, &mut self.state.0)
            .process_decimate(x)
            .zip(StatefulRef::new(&self.config.1, &mut self.state.1).process_decimate(x))
            .map(|(y0, y1)| y0 + y1)
    }

    #[inline]
    fn process_interpolate(&mut self, x: T, y: &mut [T]) {
        let y0 = StatefulRef::new(&self.config.0, &mut self.state.0).process(x);
        let y1 = StatefulRef::new(&self.config.1, &mut self.state.1).process(x);
        if let Some(y) = y.first_chunk_mut() {
            *y = [y0, y1];
        }
    }
}

impl<C1, C2, S1, S2> StatefulRef<'_, Pair<C1, C2>, PairState<S1, S2>> {
    /// Process inputs into complementary outputs in place
    ///
    /// Such that `[x0, x1]` inputs become `[y0 + y1, y0 - y1]` outputs (butterfly).
    /// For single channel input, copying and scaling by 0.5
    /// is to be done before the filter.
    #[inline]
    pub fn process_complementary_in_place<T>(&mut self, xy: [&mut [T]; 2])
    where
        T: Copy + Add<Output = T> + Sub<Output = T>,
        for<'a> StatefulRef<'a, C1, S1>: Process<T>,
        for<'a> StatefulRef<'a, C2, S2>: Process<T>,
    {
        let [xy0, xy1] = xy;
        debug_assert_eq!(xy0.len(), xy1.len());
        StatefulRef::new(&self.config.0, &mut self.state.0).process_in_place(xy0);
        StatefulRef::new(&self.config.1, &mut self.state.1).process_in_place(xy1);
        for (y0, y1) in xy0.iter_mut().zip(xy1) {
            // Butterfly
            [*y0, *y1] = [*y0 + *y1, *y0 - *y1];
        }
    }

    /// Process input as lowpass
    #[inline]
    pub fn process_sum_in_place<T, const N: usize>(&mut self, xy: &mut [T; N])
    where
        T: Copy + Default + AddAssign,
        for<'a> StatefulRef<'a, C1, S1>: Process<T>,
        for<'a> StatefulRef<'a, C2, S2>: Process<T>,
    {
        let mut xy1 = [T::default(); N];
        StatefulRef::new(&self.config.1, &mut self.state.1).process_block(xy, &mut xy1);
        StatefulRef::new(&self.config.0, &mut self.state.0).process_in_place(xy);
        for (y0, y1) in xy.iter_mut().zip(xy1) {
            *y0 += y1;
        }
    }

    /// Process input as highpass
    #[inline]
    pub fn process_difference_in_place<T, const N: usize>(&mut self, xy: &mut [T; N])
    where
        T: Copy + Default + SubAssign,
        for<'a> StatefulRef<'a, C1, S1>: Process<T>,
        for<'a> StatefulRef<'a, C2, S2>: Process<T>,
    {
        let mut xy1 = [T::default(); N];
        StatefulRef::new(&self.config.1, &mut self.state.1).process_block(xy, &mut xy1);
        StatefulRef::new(&self.config.0, &mut self.state.0).process_in_place(xy);
        for (y0, y1) in xy.iter_mut().zip(xy1) {
            *y0 -= y1;
        }
    }
}
