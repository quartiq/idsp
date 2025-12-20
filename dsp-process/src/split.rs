use crate::{
    Assert, Channels, Inplace, Minor, Parallel, Process, SplitInplace, SplitProcess, Transpose,
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
    pub fn new(config: C, state: S) -> Self {
        Self { config, state }
    }

    /// Obtain a borrowing Split
    ///
    /// Stateful `Process` is typically implemented on the borrowing Split
    pub fn as_mut(&mut self) -> Split<&C, &mut S> {
        Split {
            config: &self.config,
            state: &mut self.state,
        }
    }
}

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

/// Unzip two splits into one (config major)
impl<C0, C1, S0, S1> core::ops::Mul<Split<C1, S1>> for Split<C0, S0> {
    type Output = Split<(C0, C1), (S0, S1)>;

    fn mul(self, rhs: Split<C1, S1>) -> Self::Output {
        (self, rhs).into()
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
            core::array::from_fn(|i| splits[i].0.take().unwrap()),
            core::array::from_fn(|i| splits[i].1.take().unwrap()),
        )
    }
}

impl<C, S> Split<C, S> {
    /// Convert to a configuration-minor split
    pub fn minor<U>(self) -> Split<Minor<C, U>, S> {
        Split::new(Minor::new(self.config), self.state)
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
        Split::new(
            core::array::repeat(self.config),
            core::array::repeat(self.state),
        )
    }

    /// Apply to multiple states by cloning the current (!) state
    pub fn channels<const N: usize>(self) -> Split<Channels<C>, [S; N]>
    where
        S: Clone,
    {
        Split::new(Channels(self.config), core::array::repeat(self.state))
    }

    /// Convert to parallel transpose operation on blocks/inplace of `[[x]; N]` instead of `[[x; N]]`
    pub fn transpose(self) -> Split<Transpose<C>, S> {
        Split::new(Transpose(self.config), self.state)
    }
}

impl<C, S, U> Split<Minor<C, U>, S> {
    /// Convert to major
    pub fn major(self) -> Split<C, S> {
        Split::new(self.config.inner, self.state)
    }
}

impl<C, S> Split<Parallel<C>, S> {
    /// Convert to serial
    pub fn major(self) -> Split<C, S> {
        Split::new(self.config.0, self.state)
    }
}

impl<C, S> Split<Transpose<C>, S> {
    /// Convert to non-transposing
    pub fn major(self) -> Split<C, S> {
        Split::new(self.config.0, self.state)
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
        core::array::from_fn(|_| {
            let (c, s) = it.next().unwrap();
            Split::new(c, s)
        })
    }
}

/// Stateless/stateful marker
///
/// To be used in `Split<Unsplit<P>, ()>` and `Split<(), Unsplit<P>>`
/// to mark processors requiring no state and no configuration respectively.
#[derive(Debug, Copy, Clone, Default)]
#[repr(transparent)]
pub struct Unsplit<P>(pub P);

/// Stateless filters
impl<X: Copy, Y, P> SplitProcess<X, Y> for Unsplit<P>
where
    for<'a> &'a P: Process<X, Y>,
{
    fn process(&self, _state: &mut (), x: X) -> Y {
        (&self.0).process(x)
    }

    fn block(&self, _state: &mut (), x: &[X], y: &mut [Y]) {
        (&self.0).block(x, y)
    }
}

impl<X: Copy, P> SplitInplace<X> for Unsplit<P>
where
    for<'a> &'a P: Inplace<X>,
{
    fn inplace(&self, _state: &mut (), xy: &mut [X]) {
        (&self.0).inplace(xy)
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

//////////// SPLIT MAJOR ////////////

/// Chain of two different large filters
///
/// X->Y
impl<X: Copy, Y: Copy, C0, C1, S0, S1> SplitProcess<X, Y, (S0, S1)> for (C0, C1)
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitInplace<Y, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: X) -> Y {
        // TODO: defer to Minor
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

/// A chain of multiple large filters of the same type
///
/// slice can be empty
/// X->X->X...
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

/// A chain of multiple large filters of the same type
///
/// X->Y->Y...
impl<X: Copy, Y: Copy, C, S, const N: usize> SplitProcess<X, Y, [S; N]> for [C; N]
where
    C: SplitProcess<X, Y, S> + SplitInplace<Y, S>,
{
    fn process(&self, state: &mut [S; N], x: X) -> Y {
        let () = Assert::<N, 0>::GREATER;
        let ((c0, c), (s0, s)) = self.split_first().zip(state.split_first_mut()).unwrap();
        c.iter()
            .zip(s.iter_mut())
            .fold(c0.process(s0, x), |x, (c, s)| c.process(s, x))
    }

    fn block(&self, state: &mut [S; N], x: &[X], y: &mut [Y]) {
        let () = Assert::<N, 0>::GREATER;
        let ((c0, c), (s0, s)) = self.split_first().zip(state.split_first_mut()).unwrap();
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
        let () = Assert::<N, 0>::GREATER;
        let ((c0, c), (s0, s)) = self
            .inner
            .split_first()
            .zip(state.split_first_mut())
            .unwrap();
        c.iter()
            .zip(s.iter_mut())
            .fold(c0.process(s0, x), |x, (c, s)| c.process(s, x))
    }
}

impl<X: Copy, U, C, S> SplitInplace<X, S> for Minor<C, U> where Self: SplitProcess<X, X, S> {}

//////////// SPLIT PARALLEL ////////////

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
    [Y; N]: Default,
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((c, s), (x, y)) in self
            .0
            .iter()
            .zip(state.iter_mut())
            .zip(x.into_iter().zip(y.iter_mut()))
        {
            *y = c.process(s, x);
        }
        y
    }
}

impl<X: Copy, C, S> SplitInplace<X, S> for Parallel<C> where Self: SplitProcess<X, X, S> {}

//////////// TRANSPOSE ////////////

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

    fn block(&self, state: &mut (S0, S1), x: &[[X; 2]], y: &mut [[Y; 2]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        let (x0, x1) = x.as_flattened().split_at(n);
        let (y0, y1) = y.as_flattened_mut().split_at_mut(n);
        self.0.0.block(&mut state.0, x0, y0);
        self.0.1.block(&mut state.1, x1, y1);
    }
}

impl<X: Copy, C0, C1, S0, S1> SplitInplace<[X; 2], (S0, S1)> for Transpose<(C0, C1)>
where
    C0: SplitInplace<X, S0>,
    C1: SplitInplace<X, S1>,
{
    fn inplace(&self, state: &mut (S0, S1), xy: &mut [[X; 2]]) {
        let n = xy.len();
        let (xy0, xy1) = xy.as_flattened_mut().split_at_mut(n);
        self.0.0.inplace(&mut state.0, xy0);
        self.0.1.inplace(&mut state.1, xy1);
    }
}

impl<X: Copy, Y, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]> for Transpose<[C; N]>
where
    [Y; N]: Default,
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((c, s), (x, y)) in self
            .0
            .iter()
            .zip(state.iter_mut())
            .zip(x.into_iter().zip(y.iter_mut()))
        {
            *y = c.process(s, x);
        }
        y
    }

    fn block(&self, state: &mut [S; N], x: &[[X; N]], y: &mut [[Y; N]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        for ((c, s), (x, y)) in self.0.iter().zip(state.iter_mut()).zip(
            x.as_flattened()
                .chunks_exact(n)
                .zip(y.as_flattened_mut().chunks_exact_mut(n)),
        ) {
            c.block(s, x, y)
        }
    }
}

impl<X: Copy, C, S, const N: usize> SplitInplace<[X; N], [S; N]> for Transpose<[C; N]>
where
    [X; N]: Default,
    C: SplitInplace<X, S>,
{
    fn inplace(&self, state: &mut [S; N], xy: &mut [[X; N]]) {
        let n = xy.len();
        for ((c, s), xy) in self
            .0
            .iter()
            .zip(state.iter_mut())
            .zip(xy.as_flattened_mut().chunks_exact_mut(n))
        {
            c.inplace(s, xy)
        }
    }
}

//////////// CHANNELS ////////////

/// Process data from multiple channels with a common configuration
///
/// Note that block() and inplace() reinterpret the data as [`Transpose`]: __not__ as `[[X; N]]` but as `[[X]; N]`.
/// Use `x.as_flattened().chunks_exact(x.len())`/`x.as_chunks<N>().0` etc. to match that.
impl<X: Copy, Y, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]> for Channels<C>
where
    [Y; N]: Default,
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((x, y), state) in x.into_iter().zip(y.iter_mut()).zip(state.iter_mut()) {
            *y = self.0.process(state, x);
        }
        y
    }

    fn block(&self, state: &mut [S; N], x: &[[X; N]], y: &mut [[Y; N]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        for ((x, y), state) in x
            .as_flattened()
            .chunks_exact(n)
            .zip(y.as_flattened_mut().chunks_exact_mut(n))
            .zip(state.iter_mut())
        {
            self.0.block(state, x, y)
        }
    }
}

impl<X: Copy, C, S, const N: usize> SplitInplace<[X; N], [S; N]> for Channels<C>
where
    [X; N]: Default,
    C: SplitInplace<X, S>,
{
    fn inplace(&self, state: &mut [S; N], xy: &mut [[X; N]]) {
        let n = xy.len();
        for (xy, state) in xy
            .as_flattened_mut()
            .chunks_exact_mut(n)
            .zip(state.iter_mut())
        {
            self.0.inplace(state, xy)
        }
    }
}
