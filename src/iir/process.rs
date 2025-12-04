use core::ops::{Add, Sub};

/// Processing block
/// Single input, single output, one value
pub trait Process<T: Copy> {
    /// Update the state with a new sample and obtain an output sample
    fn process(&mut self, x: T) -> T;

    /// Process a block of samples
    #[inline]
    fn process_block(&mut self, x: &[T], y: &mut [T]) {
        for (x, y) in x.iter().zip(y) {
            *y = self.process(*x);
        }
    }

    /// Process a block of samples inplace
    #[inline]
    fn process_in_place(&mut self, xy: &mut [T]) {
        for xy in xy {
            *xy = self.process(*xy);
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
}

/// A chain of multiple filters of the same type
impl<T, C, S> Process<T> for StatefulRef<'_, [C], [S]>
where
    T: Copy,
    for<'a> StatefulRef<'a, C, S>: Process<T>,
{
    #[inline]
    fn process(&mut self, x: T) -> T {
        self.config
            .iter()
            .zip(self.state.iter_mut())
            .fold(x, |x, (f, s)| StatefulRef::new(f, s).process(x))
    }

    #[inline]
    fn process_block(&mut self, x: &[T], y: &mut [T]) {
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
    fn process(&mut self, mut x: T) -> T {
        x = StatefulRef::new(&self.config.0, &mut self.state.0).process(x);
        StatefulRef::new(&self.config.1, &mut self.state.1).process(x)
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

/// Complementary allpass filter pair
///
/// The [`Process`] implementation returns only the sum of the two filters
/// while [`Self::process_complementary_in_place`]
/// yields both sum and difference.
/// Scaling with 0.5 gain is to be performed ahead of the filter.
#[derive(Clone, Debug, Default)]
pub struct Butterfly<C1, C2>(
    /// Top filter
    pub C1,
    /// Bottom filter
    pub C2,
);

/// Complementary allpass state
#[derive(Clone, Debug, Default)]
pub struct ButterflyState<S1, S2>(
    /// Top filter state
    pub S1,
    /// Bottom filter state
    pub S2,
);

impl<T: Add<Output = T>, C1, C2, S1, S2> Process<T>
    for StatefulRef<'_, Butterfly<C1, C2>, ButterflyState<S1, S2>>
where
    T: Copy,
    for<'a> StatefulRef<'a, C1, S1>: Process<T>,
    for<'a> StatefulRef<'a, C2, S2>: Process<T>,
{
    fn process(&mut self, x: T) -> T {
        StatefulRef::new(&self.config.0, &mut self.state.0).process(x)
            + StatefulRef::new(&self.config.1, &mut self.state.1).process(x)
    }
}

impl<C1, C2, S1, S2> StatefulRef<'_, Butterfly<C1, C2>, ButterflyState<S1, S2>> {
    /// Process inputs into complementary outputs in place
    ///
    /// Such that `[x0, x1]` inputs become `[y0 + y1, y0 - y1]` outputs.
    /// For single channel input, copying and scaling by 0.5
    /// is to be done before the filter.
    pub fn process_complementary_in_place<T: Add<Output = T> + Sub<Output = T>>(
        &mut self,
        xy: [&mut [T]; 2],
    ) where
        T: Copy,
        for<'a> StatefulRef<'a, C1, S1>: Process<T>,
        for<'a> StatefulRef<'a, C2, S2>: Process<T>,
    {
        let [xy0, xy1] = xy;
        StatefulRef::new(&self.config.0, &mut self.state.0).process_in_place(xy0);
        StatefulRef::new(&self.config.1, &mut self.state.1).process_in_place(xy1);
        for (y0, y1) in xy0.iter_mut().zip(xy1) {
            // Butterfly
            [*y0, *y1] = [*y0 + *y1, *y0 - *y1];
        }
    }
}
