use core::array::{from_fn, repeat};

use crate::{
    Channels, Inplace, Major, Minor, Parallel, Process, SplitInplace, SplitProcess, Transpose,
};

//////////// SPLIT ////////////

/// A stateful processor with split state
#[derive(Debug, Copy, Clone, Default)]
pub struct Split<C, S> {
    /// Processor configuration
    pub config: C,
    /// Processor state
    pub state: S,
}

impl<X: Copy, Y, S: ?Sized, C: SplitProcess<X, Y, S> + ?Sized> Process<X, Y> for Split<&C, &mut S> {
    fn process(&mut self, x: X) -> Y {
        self.config.process(self.state, x)
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.config.block(self.state, x, y)
    }
}

impl<X: Copy, S: ?Sized, C: SplitInplace<X, S> + ?Sized> Inplace<X> for Split<&C, &mut S> {
    fn inplace(&mut self, xy: &mut [X]) {
        self.config.inplace(self.state, xy);
    }
}

impl<C, S> Split<C, S> {
    /// Create a new Split
    pub const fn new(config: C, state: S) -> Self {
        Self { config, state }
    }

    /// Statically assert that this implements Process<X, Y>
    pub const fn assert_process<X: Copy, Y>(&self)
    where
        Self: Process<X, Y>,
    {
    }

    /// Obtain a borrowing Split
    ///
    /// Stateful `Process` is implemented on the borrowing Split
    pub fn as_mut(&mut self) -> Split<&C, &mut S> {
        Split {
            config: &self.config,
            state: &mut self.state,
        }
    }
}

/// Stateless/stateful marker
///
/// To be used in `Split<Unsplit<P>, ()>` and `Split<(), Unsplit<P>>`
/// to mark processors requiring no state and no configuration respectively.
#[derive(Debug, Copy, Clone, Default)]
#[repr(transparent)]
pub struct Unsplit<P>(pub P);

impl<C> Split<Unsplit<C>, ()> {
    /// Create a stateless processor
    pub fn stateless(config: C) -> Self {
        Self::new(Unsplit(config), ())
    }
}

impl<S> Split<(), Unsplit<S>> {
    /// Create a state-only processor
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
        (self * rhs).parallel()
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
    /// Convert to a configuration-minor split
    pub fn minor<U>(self) -> Split<Minor<C, U>, S> {
        Split::new(Minor::new(self.config), self.state)
    }

    /// Convert to intermediate buffered processor-major
    pub fn major<U>(self) -> Split<Major<C, U>, S> {
        Split::new(Major::new(self.config), self.state)
    }

    /// Convert to parallel (MIMO)
    pub fn parallel(self) -> Split<Parallel<C>, S> {
        Split::new(Parallel(self.config), self.state)
    }

    /// Repeat by cloning configuration and current (!) state
    pub fn repeat<const N: usize>(self) -> Split<[C; N], [S; N]>
    where
        C: Clone,
        S: Clone,
    {
        Split::new(repeat(self.config), repeat(self.state))
    }

    /// Apply to multiple states by cloning the current (!) state
    pub fn channels<const N: usize>(self) -> Split<Channels<C>, [S; N]>
    where
        S: Clone,
    {
        Split::new(Channels(self.config), repeat(self.state))
    }

    /// Convert to parallel transpose operation on blocks/inplace of `[[x]; N]` instead of `[[x; N]]`
    pub fn transpose(self) -> Split<Transpose<C>, S> {
        Split::new(Transpose(self.config), self.state)
    }
}

impl<C, S, U> Split<Minor<C, U>, S> {
    /// Strip minor
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.inner, self.state)
    }
}

impl<C, S> Split<Parallel<C>, S> {
    /// Convert to serial
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.0, self.state)
    }
}

impl<C, S> Split<Transpose<C>, S> {
    /// Convert to non-transposing
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.0, self.state)
    }
}

impl<C, S, B> Split<Major<C, B>, S> {
    /// Remove major intermediate buffering
    pub fn inter(self) -> Split<C, S> {
        Split::new(self.config.inner, self.state)
    }
}

impl<C0, C1, S0, S1> Split<(C0, C1), (S0, S1)> {
    /// Zip up a split
    pub fn zip(self) -> (Split<C0, S0>, Split<C1, S1>) {
        (
            Split::new(self.config.0, self.state.0),
            Split::new(self.config.1, self.state.1),
        )
    }
}

impl<C, S, const N: usize> Split<[C; N], [S; N]> {
    /// Zip up a split
    pub fn zip(self) -> [Split<C, S>; N] {
        let mut it = self.config.into_iter().zip(self.state);
        from_fn(|_| {
            let (c, s) = it.next().unwrap();
            Split::new(c, s)
        })
    }
}

/// Stateless filters
impl<'a, X: Copy, Y, P> SplitProcess<X, Y> for Unsplit<&'a P>
where
    &'a P: Process<X, Y>,
{
    fn process(&self, _state: &mut (), x: X) -> Y {
        (&*self.0).process(x)
    }

    fn block(&self, _state: &mut (), x: &[X], y: &mut [Y]) {
        (&*self.0).block(x, y)
    }
}

impl<'a, X: Copy, P> SplitInplace<X> for Unsplit<&'a P>
where
    &'a P: Inplace<X>,
{
    fn inplace(&self, _state: &mut (), xy: &mut [X]) {
        (&*self.0).inplace(xy)
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
