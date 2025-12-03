/// Processing block
/// Single input, single output, one value
pub trait Process {
    /// Update the state with a new sample and obtain an output sample
    fn process(&mut self, x: i32) -> i32;

    /// Process a block of samples
    #[inline]
    fn process_block(&mut self, x: &[i32], y: &mut [i32]) {
        for (x, y) in x.iter().zip(y) {
            *y = self.process(*x);
        }
    }

    /// Process a block of samples inplace
    #[inline]
    fn process_in_place(&mut self, xy: &mut [i32]) {
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

impl<C, S> Process for Stateful<C, S>
where
    for<'a> StatefulRef<'a, C, S>: Process,
{
    #[inline]
    fn process(&mut self, x: i32) -> i32 {
        StatefulRef::new(&self.config, &mut self.state).process(x)
    }

    #[inline]
    fn process_block(&mut self, x: &[i32], y: &mut [i32]) {
        StatefulRef::new(&self.config, &mut self.state).process_block(x, y)
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [i32]) {
        StatefulRef::new(&self.config, &mut self.state).process_in_place(xy)
    }
}

/// A chain of multiple filters of the same type
impl<C, S> Process for StatefulRef<'_, [C], [S]>
where
    for<'a> StatefulRef<'a, C, S>: Process,
{
    #[inline]
    fn process(&mut self, x: i32) -> i32 {
        self.config
            .iter()
            .zip(self.state.iter_mut())
            .fold(x, |x, (f, s)| StatefulRef::new(f, s).process(x))
    }

    #[inline]
    fn process_block(&mut self, x: &[i32], y: &mut [i32]) {
        if let Some((f, s)) = self.config.first().zip(self.state.first_mut()) {
            StatefulRef::new(f, s).process_block(x, y);
        } else {
            y.clone_from_slice(x);
        }
        for (f, s) in self.config.iter().zip(self.state.iter_mut()).skip(1) {
            StatefulRef::new(f, s).process_in_place(y);
        }
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [i32]) {
        for (f, s) in self.config.iter().zip(self.state.iter_mut()) {
            StatefulRef::new(f, s).process_in_place(xy);
        }
    }
}

/// A chain of multiple filters of the same type
impl<C, S, const N: usize> Process for StatefulRef<'_, [C; N], [S; N]>
where
    for<'a> StatefulRef<'a, [C], [S]>: Process,
{
    #[inline]
    fn process(&mut self, x: i32) -> i32 {
        StatefulRef::new(&self.config[..], &mut self.state[..]).process(x)
    }

    #[inline]
    fn process_block(&mut self, x: &[i32], y: &mut [i32]) {
        StatefulRef::new(&self.config[..], &mut self.state[..]).process_block(x, y)
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [i32]) {
        StatefulRef::new(&self.config[..], &mut self.state[..]).process_in_place(xy)
    }
}

/// A chain of two filters of different type.
///
/// This then automatically covers any nested tuple.
impl<C1, C2, S1, S2> Process for StatefulRef<'_, (C1, C2), (S1, S2)>
where
    for<'a> StatefulRef<'a, C1, S1>: Process,
    for<'a> StatefulRef<'a, C2, S2>: Process,
{
    #[inline]
    fn process(&mut self, mut x: i32) -> i32 {
        x = StatefulRef::new(&self.config.0, &mut self.state.0).process(x);
        StatefulRef::new(&self.config.1, &mut self.state.1).process(x)
    }

    #[inline]
    fn process_block(&mut self, x: &[i32], y: &mut [i32]) {
        StatefulRef::new(&self.config.0, &mut self.state.0).process_block(x, y);
        StatefulRef::new(&self.config.1, &mut self.state.1).process_in_place(y);
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [i32]) {
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

impl<C1, C2, S1, S2> Process for StatefulRef<'_, Butterfly<C1, C2>, ButterflyState<S1, S2>>
where
    for<'a> StatefulRef<'a, C1, S1>: Process,
    for<'a> StatefulRef<'a, C2, S2>: Process,
{
    fn process(&mut self, x: i32) -> i32 {
        StatefulRef::new(&self.config.0, &mut self.state.0).process(x)
            + StatefulRef::new(&self.config.1, &mut self.state.1).process(x)
    }
}

impl<C1, C2, S1, S2> StatefulRef<'_, Butterfly<C1, C2>, ButterflyState<S1, S2>>
where
    for<'a> StatefulRef<'a, C1, S1>: Process,
    for<'a> StatefulRef<'a, C2, S2>: Process,
{
    /// Process complementary inputs into complementary outputs in place
    ///
    /// Such that `[x0, x1]` inputs become `[y0 + y1, y0 - y1]` outputs.
    /// For single channel input, copying and scaling by 0.5
    /// is to be done before the filter.
    pub fn process_complementary_in_place(&mut self, xy: [&mut [i32]; 2]) {
        let [xy0, xy1] = xy;
        StatefulRef::new(&self.config.0, &mut self.state.0).process_in_place(xy0);
        StatefulRef::new(&self.config.1, &mut self.state.1).process_in_place(xy1);
        for (y0, y1) in xy0.iter_mut().zip(xy1) {
            // Butterfly
            [*y0, *y1] = [*y0 + *y1, *y0 - *y1];
        }
    }
}
