use crate::{Assert, Inplace, Minor, Parallel, Process};

//////////// SPLIT ////////////

/// A stateful processor with split state
///
/// Splitting configuration (the part of the filter that is unaffected
/// by processing inputs, e.g. "coefficients"), from state (the part
/// that is modified by processing) allows:
///
/// * Separating mutable from immutable state guarantees consistency
///   (configuration can not change state and processing can
///   not change configuration)
/// * Reduces memory traffic when swapping configuration
/// * Allows the same filter to be applied to multiple states
///   (e.g. IQ data, multiple channels) guaranteeing consistency,
///   reducing memory usage, and improving caching.
#[derive(Debug, Clone, Default)]
pub struct Split<C, S> {
    /// Processor configuration
    pub config: C,
    /// Processor state
    pub state: S,
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

impl<C> Split<Stateless<C>, ()> {
    /// Create a stateless processor
    pub fn stateless(config: C) -> Self {
        Self::new(Stateless(config), ())
    }
}

impl<S> Split<(), Stateful<S>> {
    /// Create a state-only processor
    pub fn stateful(state: S) -> Self {
        Self::new((), Stateful(state))
    }
}

/// Unzip two splits into one (config major)
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

    /// Repeat by cloning configuration and state
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

    /// Apply to multiple states by clonig state
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

/// Stateless marker
///
/// To be used in `Split<Stateless<C>, ()>`
#[derive(Debug, Clone, Default)]
#[repr(transparent)]
pub struct Stateless<C>(pub C);

/// Stateless filters
impl<X: Copy, Y, C> Process<X, Y> for Split<&Stateless<C>, &mut ()>
where
    for<'a> &'a C: Process<X, Y>,
{
    fn process(&mut self, x: X) -> Y {
        (&self.config.0).process(x)
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        (&self.config.0).block(x, y)
    }
}

impl<X: Copy, C> Inplace<X> for Split<&Stateless<C>, &mut ()>
where
    for<'a> &'a C: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [X]) {
        (&self.config.0).inplace(xy)
    }
}

/// State-only marker
///
/// To be used in `Split<(), Stateful<S>>`
#[derive(Debug, Clone, Default)]
#[repr(transparent)]
pub struct Stateful<S>(pub S);

/// Configuration-less filters
impl<X: Copy, Y, S: Process<X, Y>> Process<X, Y> for Split<&(), &mut Stateful<S>> {
    fn process(&mut self, x: X) -> Y {
        self.state.0.process(x)
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.state.0.block(x, y)
    }
}

impl<X: Copy, S: Inplace<X>> Inplace<X> for Split<&(), &mut Stateful<S>> {
    fn inplace(&mut self, xy: &mut [X]) {
        self.state.0.inplace(xy)
    }
}

//////////// SPLIT MAJOR ////////////

/// Chain of two different large filters
///
/// X->Y
impl<X: Copy, Y: Copy, C0, C1, S0, S1> Process<X, Y> for Split<&(C0, C1), &mut (S0, S1)>
where
    for<'a> Split<&'a C0, &'a mut S0>: Process<X, Y>,
    for<'a> Split<&'a C1, &'a mut S1>: Inplace<Y>,
{
    fn process(&mut self, x: X) -> Y {
        // TODO: defer to Minor
        let u = Split::new(&self.config.0, &mut self.state.0).process(x);
        Split::new(&self.config.1, &mut self.state.1).process(u)
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        Split::new(&self.config.0, &mut self.state.0).block(x, y);
        Split::new(&self.config.1, &mut self.state.1).inplace(y);
    }
}

impl<X: Copy, C0, C1, S0, S1> Inplace<X> for Split<&(C0, C1), &mut (S0, S1)>
where
    for<'a> Split<&'a C0, &'a mut S0>: Inplace<X>,
    for<'a> Split<&'a C1, &'a mut S1>: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [X]) {
        Split::new(&self.config.0, &mut self.state.0).inplace(xy);
        Split::new(&self.config.1, &mut self.state.1).inplace(xy);
    }
}

/// A chain of multiple large filters of the same type
///
/// slice can be empty
/// X->X->X...
impl<X: Copy, C, S> Process<X> for Split<&[C], &mut [S]>
where
    for<'a> Split<&'a C, &'a mut S>: Inplace<X>,
{
    fn process(&mut self, x: X) -> X {
        Split::new(Minor::new(self.config), &mut *self.state).process(x)
    }

    fn block(&mut self, x: &[X], y: &mut [X]) {
        debug_assert_eq!(self.config.len(), self.state.len());
        if let Some(((c0, c), (s0, s))) =
            self.config.split_first().zip(self.state.split_first_mut())
        {
            Split::new(c0, s0).block(x, y);
            for (c, s) in c.iter().zip(s) {
                Split::new(c, s).inplace(y);
            }
        } else {
            y.copy_from_slice(x);
        }
    }
}

impl<X: Copy, C, S> Inplace<X> for Split<&[C], &mut [S]>
where
    for<'a> Split<&'a C, &'a mut S>: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [X]) {
        debug_assert_eq!(self.config.len(), self.state.len());
        for (c, s) in self.config.iter().zip(self.state.iter_mut()) {
            Split::new(c, s).inplace(xy);
        }
    }
}

/// A chain of multiple large filters of the same type
///
/// X->Y->Y...
impl<X: Copy, Y: Copy, C, S, const N: usize> Process<X, Y> for Split<&[C; N], &mut [S; N]>
where
    for<'a> Split<&'a C, &'a mut S>: Process<X, Y> + Inplace<Y>,
{
    fn process(&mut self, x: X) -> Y {
        let () = Assert::<N, 0>::GREATER;
        let ((c0, c), (s0, s)) = self
            .config
            .split_first()
            .zip(self.state.split_first_mut())
            .unwrap();
        c.iter()
            .zip(s.iter_mut())
            .fold(Split::new(c0, s0).process(x), |x, (c, s)| {
                Split::new(c, s).process(x)
            })
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        let () = Assert::<N, 0>::GREATER;
        let ((c0, c), (s0, s)) = self
            .config
            .split_first()
            .zip(self.state.split_first_mut())
            .unwrap();
        Split::new(c0, s0).block(x, y);
        for (c, s) in c.iter().zip(s.iter_mut()) {
            Split::new(c, s).inplace(y)
        }
    }
}

impl<X: Copy, C, S, const N: usize> Inplace<X> for Split<&[C; N], &mut [S; N]>
where
    for<'a> Split<&'a C, &'a mut S>: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [X]) {
        Split::new(self.config.as_ref(), self.state.as_mut()).inplace(xy)
    }
}

//////////// SPLIT MINOR ////////////

impl<X: Copy, U: Copy, Y, C0, C1, S0, S1> Process<X, Y>
    for Split<&Minor<(C0, C1), U>, &mut (S0, S1)>
where
    for<'a> Split<&'a C0, &'a mut S0>: Process<X, U>,
    for<'a> Split<&'a C1, &'a mut S1>: Process<U, Y>,
{
    fn process(&mut self, x: X) -> Y {
        let u = Split::new(&self.config.inner.0, &mut self.state.0).process(x);
        Split::new(&self.config.inner.1, &mut self.state.1).process(u)
    }
}

/// A chain of multiple small filters of the same type
impl<X: Copy, C, S> Process<X> for Split<Minor<&[C], X>, &mut [S]>
where
    for<'a> Split<&'a C, &'a mut S>: Process<X>,
{
    fn process(&mut self, x: X) -> X {
        debug_assert_eq!(self.config.inner.len(), self.state.len());
        self.config
            .inner
            .iter()
            .zip(self.state.iter_mut())
            .fold(x, |x, (f, s)| Split::new(f, s).process(x))
    }
}

/// A chain of multiple small filters of the same type
impl<X: Copy, Y: Copy, C, S, const N: usize> Process<X, Y> for Split<&Minor<[C; N], Y>, &mut [S; N]>
where
    for<'a> Split<&'a C, &'a mut S>: Process<X, Y> + Process<Y>,
{
    fn process(&mut self, x: X) -> Y {
        let () = Assert::<N, 0>::GREATER;
        let ((c0, c), (s0, s)) = self
            .config
            .inner
            .split_first()
            .zip(self.state.split_first_mut())
            .unwrap();
        c.iter()
            .zip(s.iter_mut())
            .fold(Split::new(c0, s0).process(x), |x, (c, s)| {
                Split::new(c, s).process(x)
            })
    }
}

impl<X: Copy, U, C, S: ?Sized> Inplace<X> for Split<&Minor<C, U>, &mut S> where Self: Process<X> {}

//////////// SPLIT PARALLEL ////////////

impl<X0: Copy, X1: Copy, Y0, Y1, C0, C1, S0, S1> Process<(X0, X1), (Y0, Y1)>
    for Split<&Parallel<(C0, C1)>, &mut (S0, S1)>
where
    for<'a> Split<&'a C0, &'a mut S0>: Process<X0, Y0>,
    for<'a> Split<&'a C1, &'a mut S1>: Process<X1, Y1>,
{
    fn process(&mut self, x: (X0, X1)) -> (Y0, Y1) {
        (
            Split::new(&self.config.0.0, &mut self.state.0).process(x.0),
            Split::new(&self.config.0.1, &mut self.state.1).process(x.1),
        )
    }
}

impl<X: Copy, Y, C0, C1, S0, S1> Process<[X; 2], [Y; 2]>
    for Split<&Parallel<(C0, C1)>, &mut (S0, S1)>
where
    for<'a> Split<&'a C0, &'a mut S0>: Process<X, Y>,
    for<'a> Split<&'a C1, &'a mut S1>: Process<X, Y>,
{
    fn process(&mut self, x: [X; 2]) -> [Y; 2] {
        [
            Split::new(&self.config.0.0, &mut self.state.0).process(x[0]),
            Split::new(&self.config.0.1, &mut self.state.1).process(x[1]),
        ]
    }
}

impl<X: Copy, Y, C, S, const N: usize> Process<[X; N], [Y; N]>
    for Split<&Parallel<[C; N]>, &mut [S; N]>
where
    [Y; N]: Default,
    for<'a> Split<&'a C, &'a mut S>: Process<X, Y>,
{
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((c, s), (x, y)) in self
            .config
            .0
            .iter()
            .zip(self.state.iter_mut())
            .zip(x.into_iter().zip(y.iter_mut()))
        {
            *y = Split::new(c, s).process(x);
        }
        y
    }
}

impl<X: Copy, C, S: ?Sized> Inplace<X> for Split<&Parallel<C>, &mut S> where Self: Process<X> {}

//////////// TRANSPOSE ////////////

/// Like [`Parallel`] but transposing `[[X;N]] <-> [[X];N]`
#[derive(Clone, Debug, Default)]
pub struct Transpose<C>(pub C);

impl<X: Copy, Y, C0, C1, S0, S1> Process<[X; 2], [Y; 2]>
    for Split<&Transpose<(C0, C1)>, &mut (S0, S1)>
where
    for<'a> Split<&'a C0, &'a mut S0>: Process<X, Y>,
    for<'a> Split<&'a C1, &'a mut S1>: Process<X, Y>,
{
    fn process(&mut self, x: [X; 2]) -> [Y; 2] {
        [
            Split::new(&self.config.0.0, &mut self.state.0).process(x[0]),
            Split::new(&self.config.0.1, &mut self.state.1).process(x[1]),
        ]
    }

    fn block(&mut self, x: &[[X; 2]], y: &mut [[Y; 2]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        let (x0, x1) = x.as_flattened().split_at(n);
        let (y0, y1) = y.as_flattened_mut().split_at_mut(n);
        Split::new(&self.config.0.0, &mut self.state.0).block(x0, y0);
        Split::new(&self.config.0.1, &mut self.state.1).block(x1, y1);
    }
}

impl<X: Copy, C0, C1, S0, S1> Inplace<[X; 2]> for Split<&Transpose<(C0, C1)>, &mut (S0, S1)>
where
    for<'a> Split<&'a C0, &'a mut S0>: Inplace<X>,
    for<'a> Split<&'a C1, &'a mut S1>: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [[X; 2]]) {
        let n = xy.len();
        let (xy0, xy1) = xy.as_flattened_mut().split_at_mut(n);
        Split::new(&self.config.0.0, &mut self.state.0).inplace(xy0);
        Split::new(&self.config.0.1, &mut self.state.1).inplace(xy1);
    }
}

impl<X: Copy, Y, C, S, const N: usize> Process<[X; N], [Y; N]>
    for Split<&Transpose<[C; N]>, &mut [S; N]>
where
    [Y; N]: Default,
    for<'a> Split<&'a C, &'a mut S>: Process<X, Y>,
{
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((c, s), (x, y)) in self
            .config
            .0
            .iter()
            .zip(self.state.iter_mut())
            .zip(x.into_iter().zip(y.iter_mut()))
        {
            *y = Split::new(c, s).process(x);
        }
        y
    }

    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; N]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        for ((c, s), (x, y)) in self.config.0.iter().zip(self.state.iter_mut()).zip(
            x.as_flattened()
                .chunks_exact(n)
                .zip(y.as_flattened_mut().chunks_exact_mut(n)),
        ) {
            Split::new(c, s).block(x, y)
        }
    }
}

impl<X: Copy, C, S, const N: usize> Inplace<[X; N]> for Split<&Transpose<[C; N]>, &mut [S; N]>
where
    [X; N]: Default,
    for<'a> Split<&'a C, &'a mut S>: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        let n = xy.len();
        for ((c, s), xy) in self
            .config
            .0
            .iter()
            .zip(self.state.iter_mut())
            .zip(xy.as_flattened_mut().chunks_exact_mut(n))
        {
            Split::new(c, s).inplace(xy)
        }
    }
}

//////////// CHANNELS ////////////

/// Multiple channels to be processed with the same configuration
#[derive(Clone, Debug, Default)]
pub struct Channels<C>(pub C);

/// Process data from multiple channels with a common configuration
///
/// Note that block() and inplace() reinterpret the data as transposed: __not__ as `[[X; N]]` but as `[[X]; N]`.
/// Use `x.as_flattened().chunks_exact(x.len())`/`x.as_chunks<N>().0` etc. to match that.
impl<X: Copy, Y, C, S, const N: usize> Process<[X; N], [Y; N]> for Split<&Channels<C>, &mut [S; N]>
where
    [Y; N]: Default,
    for<'a> Split<&'a C, &'a mut S>: Process<X, Y>,
{
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for ((x, y), state) in x.into_iter().zip(y.iter_mut()).zip(self.state.iter_mut()) {
            *y = Split::new(&self.config.0, state).process(x);
        }
        y
    }

    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; N]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        for ((x, y), state) in x
            .as_flattened()
            .chunks_exact(n)
            .zip(y.as_flattened_mut().chunks_exact_mut(n))
            .zip(self.state.iter_mut())
        {
            Split::new(&self.config.0, state).block(x, y)
        }
    }
}

impl<X: Copy, C, S, const N: usize> Inplace<[X; N]> for Split<&Channels<C>, &mut [S; N]>
where
    [X; N]: Default,
    for<'a> Split<&'a C, &'a mut S>: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        let n = xy.len();
        for (xy, state) in xy
            .as_flattened_mut()
            .chunks_exact_mut(n)
            .zip(self.state.iter_mut())
        {
            Split::new(&self.config.0, state).inplace(xy)
        }
    }
}
