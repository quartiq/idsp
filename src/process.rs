//! Sample processing, filtering, combination of filters.
use core::marker::PhantomData;

/// Binary const assertions
struct Assert<const A: usize, const B: usize>;
impl<const A: usize, const B: usize> Assert<A, B> {
    /// Assert A>B
    const GREATER: () = assert!(A > B);
}

//////////// TRAITS ////////////

/// Processing block
///
/// Single input, single output
///
/// Process impls can be cascaded in (homogeneous) `[C; N]` arrays/`[C]` slices, and heterogeneous
/// `(C0, C1)` tuples. They can be used as configuration-major or
/// configuration-minor (through [`Minor`]) or in [`Add`]s on complementary allpasses and polyphase banks.
/// Tuples, arrays, and Pairs, and Minor can be mixed and nested ad lib.
///
/// For a given filter configuration `C` and state `S` pair the trait is usually implemented
/// through [`Split<&'a C, &mut S>`] (created ad-hoc from by borrowing configuration and state)
/// or [`Split<C, S>`] (owned configuration and state).
/// Stateless filters should implement `Process for &Self` for composability through
/// [`Split<Stateless<Self>, ()>`].
/// Configuration-less filters or filters that include their configuration should implement
/// `Process for Self` and can be used in split configurations through [`Split<(), Stateful<Self>>`].
pub trait Process<X: Copy, Y = X> {
    /// Update the state with a new input and obtain an output
    fn process(&mut self, x: X) -> Y;

    /// Process a block of inputs into a block of outputs
    ///
    /// Input and output must be of the same size.
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
    fn inplace(&mut self, xy: &mut [X]) {
        for xy in xy.iter_mut() {
            *xy = self.process(*xy);
        }
    }
}

//////////// BLANKET ////////////

impl<X: Copy, Y, T: Process<X, Y>> Process<X, Y> for &mut T {
    fn process(&mut self, x: X) -> Y {
        (*self).process(x)
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        (*self).block(x, y)
    }
}

impl<X: Copy, T: Inplace<X>> Inplace<X> for &mut T {
    fn inplace(&mut self, xy: &mut [X]) {
        (*self).inplace(xy)
    }
}

//////////// MAJOR ////////////

/// Type sequence must be X->Y->Y
impl<X: Copy, Y: Copy, P0: Process<X, Y>, P1: Inplace<Y>> Process<X, Y> for (P0, P1) {
    fn process(&mut self, x: X) -> Y {
        self.1.process(self.0.process(x))
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.0.block(x, y);
        self.1.inplace(y);
    }
}

impl<X: Copy, P0: Inplace<X>, P1: Inplace<X>> Inplace<X> for (P0, P1) {
    fn inplace(&mut self, xy: &mut [X]) {
        self.0.inplace(xy);
        self.1.inplace(xy);
    }
}

/// Slices can be zero length but must have input and output types the same
impl<X: Copy, P: Inplace<X>> Process<X> for [P] {
    fn process(&mut self, x: X) -> X {
        Minor::new(self).process(x)
    }

    fn block(&mut self, x: &[X], y: &mut [X]) {
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
    fn inplace(&mut self, xy: &mut [X]) {
        for p in self.iter_mut() {
            p.inplace(xy)
        }
    }
}

/// Arrays must be non-empty but that first item can transform X->Y
impl<X: Copy, Y: Copy, P: Process<X, Y> + Inplace<Y>, const N: usize> Process<X, Y> for [P; N] {
    fn process(&mut self, x: X) -> Y {
        let () = Assert::<N, 0>::GREATER;
        let (p0, p) = self.split_first_mut().unwrap();
        p.iter_mut().fold(p0.process(x), |x, p| p.process(x))
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        let () = Assert::<N, 0>::GREATER;
        let (p0, p) = self.split_first_mut().unwrap();
        p0.block(x, y);
        for p in p.iter_mut() {
            p.inplace(y);
        }
    }
}

impl<X: Copy, P: Inplace<X>, const N: usize> Inplace<X> for [P; N] {
    fn inplace(&mut self, xy: &mut [X]) {
        self.as_mut().inplace(xy)
    }
}

//////////// MINOR ////////////

/// Processor-minor, data-major
///
/// The various Process tooling implementations for `Minor`
/// place the data loop as the outer-most loop (processor-minor, data-major).
/// This is optimal for processors with small or no state and configuration.
///
/// Chain of large processors are implemented through tuples and slices/arrays.
/// Those optimize well if the sizes obey configuration ~ state > data.
/// If they do not, use `Minor`.
///
/// Note that the major implementations only override the behavior
/// for `block()` and `inplace()`. `process()` is unaffected.
#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Minor<C, U> {
    /// The inner configurations
    pub inner: C,
    /// An intermediate data type
    _intermediate: PhantomData<U>,
}

impl<C, U> Minor<C, U> {
    /// Create a new chain
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            _intermediate: PhantomData,
        }
    }
}

impl<C, U, const N: usize> Minor<[C; N], U> {
    /// Borrowed Self
    pub fn as_ref(&self) -> Minor<&[C], U> {
        Minor::new(self.inner.as_ref())
    }

    /// Mutably borrowed Self
    pub fn as_mut(&mut self) -> Minor<&mut [C], U> {
        Minor::new(self.inner.as_mut())
    }
}

/// X->U->X
impl<X: Copy, U: Copy, Y, P0: Process<X, U>, P1: Process<U, Y>> Process<X, Y>
    for Minor<(P0, P1), U>
{
    fn process(&mut self, x: X) -> Y {
        self.inner.1.process(self.inner.0.process(x))
    }
}

/// X->X->X...
impl<X: Copy, P: Process<X>> Process<X> for Minor<&mut [P], X> {
    fn process(&mut self, x: X) -> X {
        self.inner.iter_mut().fold(x, |x, p| p.process(x))
    }
}

/// X->Y->Y...
impl<X: Copy, Y: Copy, P: Process<X, Y> + Process<Y>, const N: usize> Process<X, Y>
    for Minor<[P; N], Y>
{
    fn process(&mut self, x: X) -> Y {
        let () = Assert::<N, 0>::GREATER;
        let (p0, p) = self.inner.split_first_mut().unwrap();
        p.iter_mut().fold(p0.process(x), |x, p| p.process(x))
    }
}

impl<X: Copy, U, P> Inplace<X> for Minor<P, U> where Self: Process<X> {}

//////////// PARALLEL ////////////

/// Fan out parallel input to parallel processors
#[derive(Clone, Debug, Default)]
pub struct Parallel<P>(pub P);

impl<X0: Copy, X1: Copy, Y0, Y1, P0: Process<X0, Y0>, P1: Process<X1, Y1>>
    Process<(X0, X1), (Y0, Y1)> for Parallel<(P0, P1)>
{
    fn process(&mut self, x: (X0, X1)) -> (Y0, Y1) {
        (self.0.0.process(x.0), self.0.1.process(x.1))
    }
}

impl<X: Copy, Y, P0: Process<X, Y>, P1: Process<X, Y>> Process<[X; 2], [Y; 2]>
    for Parallel<(P0, P1)>
{
    fn process(&mut self, x: [X; 2]) -> [Y; 2] {
        [self.0.0.process(x[0]), self.0.1.process(x[1])]
    }
}

impl<X: Copy, Y, P: Process<X, Y>, const N: usize> Process<[X; N], [Y; N]> for Parallel<[P; N]>
where
    [Y; N]: Default,
{
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        for (c, (x, y)) in self.0.iter_mut().zip(x.into_iter().zip(y.iter_mut())) {
            *y = c.process(x);
        }
        y
    }
}

impl<X: Copy, P> Inplace<X> for Parallel<P> where Self: Process<X> {}

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

/// `[[X;N]] <-> [[X];N]`
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

/// Process samples from multiple channels with a common configuration
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

//////////// ELEMENTARY PROCESSORS ////////////

/// Sum outputs of filters
///
/// Fan in.
#[derive(Debug, Clone, Default)]
pub struct Add;
impl<X: Copy, Y: core::iter::Sum<X>, const N: usize> Process<[X; N], Y> for &Add {
    fn process(&mut self, x: [X; N]) -> Y {
        x.into_iter().sum()
    }
}
impl<X0: Copy + core::ops::Add<X1, Output = Y>, X1: Copy, Y> Process<(X0, X1), Y> for &Add {
    fn process(&mut self, x: (X0, X1)) -> Y {
        x.0 + x.1
    }
}
impl<X: Copy> Inplace<X> for &Add where Self: Process<X> {}

/// Product of outputs of filters
///
/// Fan in.
#[derive(Debug, Clone, Default)]
pub struct Mul;
impl<X: Copy, Y: core::iter::Product<X>, const N: usize> Process<[X; N], Y> for &Mul {
    fn process(&mut self, x: [X; N]) -> Y {
        x.into_iter().product()
    }
}
impl<X0: Copy + core::ops::Mul<X1, Output = Y>, X1: Copy, Y> Process<(X0, X1), Y> for &Mul {
    fn process(&mut self, x: (X0, X1)) -> Y {
        x.0 * x.1
    }
}
impl<X: Copy> Inplace<X> for &Mul where Self: Process<X> {}

/// Difference of outputs of two filters
///
/// Fan in.
#[derive(Debug, Clone, Default)]
pub struct Sub;
impl<X: Copy + core::ops::Sub<Output = Y>, Y> Process<[X; 2], Y> for &Sub {
    fn process(&mut self, x: [X; 2]) -> Y {
        x[0] - x[1]
    }
}
impl<X0: Copy + core::ops::Sub<X1, Output = Y>, X1: Copy, Y> Process<(X0, X1), Y> for &Sub {
    fn process(&mut self, x: (X0, X1)) -> Y {
        x.0 - x.1
    }
}
impl<X: Copy> Inplace<X> for &Sub where Self: Process<X> {}

/// Sum and difference of outputs of two filters
#[derive(Debug, Clone, Default)]
pub struct Butterfly;
impl<X: Copy + core::ops::Add<Output = Y> + core::ops::Sub<Output = Y>, Y> Process<[X; 2], [Y; 2]>
    for &Butterfly
{
    fn process(&mut self, x: [X; 2]) -> [Y; 2] {
        [x[0] + x[1], x[0] - x[1]]
    }
}

impl<X: Copy> Inplace<X> for &Butterfly where Self: Process<X> {}

/// Identity using [`Copy`]
#[derive(Debug, Clone, Default)]
pub struct Identity;
impl<T: Copy> Process<T> for &Identity {
    fn process(&mut self, x: T) -> T {
        x
    }

    fn block(&mut self, x: &[T], y: &mut [T]) {
        y.copy_from_slice(x);
    }
}

impl<T: Copy> Inplace<T> for &Identity {
    fn inplace(&mut self, _xy: &mut [T]) {}
}

impl<X: Copy> Process<X, (X, X)> for &Identity {
    fn process(&mut self, x: X) -> (X, X) {
        (x, x)
    }
}

impl<X: Copy, const N: usize> Process<X, [X; N]> for &Identity {
    fn process(&mut self, x: X) -> [X; N] {
        core::array::repeat(x)
    }
}

/// Inversion using `Neg`.
#[derive(Debug, Clone, Default)]
pub struct Invert;
impl<T: Copy + core::ops::Neg<Output = T>> Process<T> for &Invert {
    fn process(&mut self, x: T) -> T {
        x.neg()
    }
}

impl<T: Copy> Inplace<T> for &Invert where Self: Process<T> {}

/// Addition of a constant
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct Offset<T>(pub T);

/// Offset using `Add`
impl<X: Copy, Y, T: core::ops::Add<X, Output = Y> + Copy> Process<X, Y> for &Offset<T> {
    fn process(&mut self, x: X) -> Y {
        self.0 + x
    }
}

impl<X: Copy, T> Inplace<X> for &Offset<T> where Self: Process<X> {}

/// Multiply by constant
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct Gain<T>(pub T);

/// Gain using `Mul`
impl<X: Copy, Y, T: core::ops::Mul<X, Output = Y> + Copy> Process<X, Y> for &Gain<T> {
    fn process(&mut self, x: X) -> Y {
        self.0 * x
    }
}

impl<X: Copy, T> Inplace<X> for &Gain<T> where Self: Process<X> {}

/// Clamp between min and max using `Ord`
#[derive(Debug, Clone, Default)]
pub struct Clamp<T> {
    /// Lowest output value
    pub min: T,
    /// Highest output value
    pub max: T,
}

impl<T: Copy + Ord> Process<T> for &Clamp<T> {
    fn process(&mut self, x: T) -> T {
        x.clamp(self.min, self.max)
    }
}

impl<T: Copy> Inplace<T> for &Clamp<T> where Self: Process<T> {}

/// Decimate or zero stuff
pub struct Rate;
impl<X: Copy, const N: usize> Process<[X; N], X> for &Rate {
    fn process(&mut self, x: [X; N]) -> X {
        x[N - 1]
    }
}

impl<X: Copy, const N: usize> Process<X, [X; N]> for &Rate
where
    [X; N]: Default,
{
    fn process(&mut self, x: X) -> [X; N] {
        let mut y = <[X; N]>::default();
        y[0] = x;
        y
    }
}
impl<X: Copy> Inplace<X> for &Rate where Self: Process<X> {}

/// Buffer input or output, or fixed delay line
#[derive(Debug, Clone)]
pub struct Buffer<X, const N: usize> {
    buffer: [X; N],
    idx: usize,
}

impl<X, const N: usize> Buffer<X, N> {
    /// The buffer is empty
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
    fn process(&mut self, x: X) -> X {
        self.idx = (self.idx + 1) % N;
        core::mem::replace(&mut self.buffer[self.idx], x)
    }

    // TODO: block(), inplace(), Process<[X; M]>
}

impl<X: Copy, const N: usize> Inplace<X> for Buffer<X, N> where Self: Process<X> {}

impl<X: Copy, const N: usize> Process<X, Option<[X; N]>> for Buffer<X, N> {
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

/// Panics on underflow
impl<X: Copy, const N: usize> Process<Option<[X; N]>, X> for Buffer<X, N> {
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

/// Adapts an interpolator to output chunk mode
pub struct Interpolator<P>(pub P);
impl<X: Copy, Y, P: Process<Option<X>, Y>, const N: usize> Process<X, [Y; N]> for Interpolator<P>
where
    [Y; N]: Default,
{
    fn process(&mut self, x: X) -> [Y; N] {
        let mut y = <[Y; N]>::default();
        if let Some((y0, y)) = y.split_first_mut() {
            *y0 = self.0.process(Some(x));
            for y in y.iter_mut() {
                *y = self.0.process(None);
            }
        }
        y
    }
}
impl<X: Copy, P> Inplace<X> for Interpolator<P> where Self: Process<X> {}

/// Adapts a decimator to input chunk mode
///
/// Synchronizes to the inner tick bu discarding samples after tick.
/// Panics if tick does not match N
pub struct Decimator<P>(pub P);
impl<X: Copy, Y, P: Process<X, Option<Y>>, const N: usize> Process<[X; N], Y> for Decimator<P> {
    fn process(&mut self, x: [X; N]) -> Y {
        x.into_iter().find_map(|x| self.0.process(x)).unwrap()
    }
}
impl<X: Copy, P> Inplace<X> for Decimator<P> where Self: Process<X> {}

// TODO: Nyquist, Integrator, Derivative

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
///
/// The corresponding state for this is `(((), (S0, S1)), ())`.
pub type Pair<C0, C1, X, I = Stateless<Identity>, J = Stateless<Add>> =
    Minor<((I, Parallel<(C0, C1)>), J), [X; 2]>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::Q32;

    #[allow(unused)]
    const fn assert_process<T: Process<X, Y>, X: Copy, Y>() {}

    #[test]
    fn stateless() {
        let y: i32 = (&Identity).process(3);
        assert_eq!(y, 3);
        assert_eq!(Split::stateless(Invert).as_mut().process(9), -9);
        assert_eq!((&Gain(Q32::<3>::new(32))).process(9), 9 * 4);
        assert_eq!((&Offset(7)).process(9), 7 + 9);
        assert_eq!(Minor::new((&Offset(7), &Offset(1))).process(9), 7 + 1 + 9);
        let mut xy = [3, 0, 0];
        let mut dly = Buffer::<_, 2>::default();
        dly.inplace(&mut xy);
        assert_eq!(xy, [0, 0, 3]);
        let y: i32 = Split::stateful(dly).as_mut().process(4);
        assert_eq!(y, 0);
        let f = Pair::<_, _, i32>::new((
            (
                Default::default(),
                Parallel((Stateless(Offset(3)), Stateless(Gain(Q32::<1>::new(4))))),
            ),
            Default::default(),
        ));
        let y: i32 = Split::new(&f, &mut Default::default()).process(5);
        assert_eq!(y, (5 + 3) + ((5 * 4) >> 1));
        let y: [i32; 5] = Split::new(Channels(f), Default::default())
            .as_mut()
            .process([5; _]);
        assert_eq!(y, [(5 + 3) + ((5 * 4) >> 1); 5]);
    }
}
