use crate::{SplitInplace, SplitProcess};
use core::marker::PhantomData;

//////////// SPLIT COMPOSE ////////////

/// Chain of two different large filters
///
/// `X->Y->Y`
impl<X: Copy, Y: Copy, C0, C1, S0, S1> SplitProcess<X, Y, (S0, S1)> for (C0, C1)
where
    C0: SplitProcess<X, Y, S0>,
    C1: SplitInplace<Y, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: X) -> Y {
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

/// Chain of multiple large filters of the same type
///
///
/// `X->X->X...`
///
/// * Clice can be empty
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
/// `X->Y->Y...`
impl<X: Copy, Y: Copy, C, S, const N: usize> SplitProcess<X, Y, [S; N]> for [C; N]
where
    C: SplitProcess<X, Y, S> + SplitInplace<Y, S>,
{
    fn process(&self, state: &mut [S; N], x: X) -> Y {
        const { assert!(N > 0) }
        let Some(((c0, c), (s0, s))) = self.split_first().zip(state.split_first_mut()) else {
            unreachable!()
        };
        c.iter()
            .zip(s.iter_mut())
            .fold(c0.process(s0, x), |x, (c, s)| c.process(s, x))
    }

    fn block(&self, state: &mut [S; N], x: &[X], y: &mut [Y]) {
        const { assert!(N > 0) }
        let Some(((c0, c), (s0, s))) = self.split_first().zip(state.split_first_mut()) else {
            unreachable!()
        };
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

/// Processor-minor, data-major
///
/// The various Process tooling implementations for `Minor`
/// place the data loop as the outer-most loop (processor-minor, data-major).
/// This is optimal for processors with small or no state and configuration.
///
/// Chain of large processors are implemented through native
/// tuples and slices/arrays or [`Major`].
/// Those optimize well if the sizes obey configuration ~ state > data.
/// If they do not, use `Minor`.
///
/// Note that the major implementations only override the behavior
/// for `block()` and `inplace()`. `process()` is unaffected and the same for all.
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct Minor<C: ?Sized, U> {
    /// An intermediate data type
    _intermediate: PhantomData<U>,
    /// The inner configurations
    pub inner: C,
}

impl<C, U> Minor<C, U> {
    /// Create a new chain
    pub const fn new(inner: C) -> Self {
        Self {
            inner,
            _intermediate: PhantomData,
        }
    }
}

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
        const { assert!(N > 0) }
        let Some(((c0, c), (s0, s))) = self.inner.split_first().zip(state.split_first_mut()) else {
            unreachable!()
        };
        c.iter()
            .zip(s.iter_mut())
            .fold(c0.process(s0, x), |x, (c, s)| c.process(s, x))
    }
}

impl<X: Copy, U, C, S> SplitInplace<X, S> for Minor<C, U> where Self: SplitProcess<X, X, S> {}

//////////// SPLIT PARALLEL ////////////

/// Fan out parallel input to parallel processors
#[derive(Clone, Copy, Debug, Default)]
pub struct Parallel<P>(pub P);

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

impl<X: Copy, Y: Copy + Default, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]>
    for Parallel<[C; N]>
where
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        let mut y = [Y::default(); N];
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

/// Data block transposition wrapper
///
/// Like [`Parallel`] but reinterpreting data as transpose `[[X; N]] <-> [[X]; N]`
/// such that `block()` and `inplace()` are lowered.
#[derive(Clone, Copy, Debug, Default)]
pub struct Transpose<C>(pub C);

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

impl<X: Copy, Y: Copy + Default, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]>
    for Transpose<[C; N]>
where
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        let mut y = [Y::default(); N];
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

impl<X: Copy + Default, C, S, const N: usize> SplitInplace<[X; N], [S; N]> for Transpose<[C; N]>
where
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

/// Multiple channels to be processed with the same configuration
#[derive(Clone, Copy, Debug, Default)]
pub struct Channels<C>(pub C);

/// Process data from multiple channels with a common configuration
///
/// Note that block() and inplace() reinterpret the data as [`Transpose`]: __not__ as `[[X; N]]` but as `[[X]; N]`.
/// Use `x.as_flattened().chunks_exact(x.len())`/`x.as_chunks<N>().0` etc. to match that.
impl<X: Copy, Y: Copy + Default, C, S, const N: usize> SplitProcess<[X; N], [Y; N], [S; N]>
    for Channels<C>
where
    C: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut [S; N], x: [X; N]) -> [Y; N] {
        let mut y = [Y::default(); N];
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

impl<X: Copy + Default, C, S, const N: usize> SplitInplace<[X; N], [S; N]> for Channels<C>
where
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

//////////// SPLIT MAJOR ////////////

/// Chain of major processors with intermediate buffer supporting block() processing
/// from individual block()s
///
/// Prefer default composition for X->X->X, arrays/slices where inplace is possible
#[derive(Debug, Clone, Copy, Default)]
pub struct Major<P: ?Sized, U> {
    /// Intermediate buffer
    _buf: PhantomData<U>,
    /// The inner processors
    pub inner: P,
}
impl<P, U> Major<P, U> {
    /// Create a new chain of processors
    pub const fn new(inner: P) -> Self {
        Self {
            inner,
            _buf: PhantomData,
        }
    }
}

impl<X: Copy, U: Copy + Default, Y, C0, C1, S0, S1, const N: usize> SplitProcess<X, Y, (S0, S1)>
    for Major<(C0, C1), [U; N]>
where
    C0: SplitProcess<X, U, S0>,
    C1: SplitProcess<U, Y, S1>,
{
    fn process(&self, state: &mut (S0, S1), x: X) -> Y {
        self.inner
            .1
            .process(&mut state.1, self.inner.0.process(&mut state.0, x))
    }

    fn block(&self, state: &mut (S0, S1), x: &[X], y: &mut [Y]) {
        let mut u = [U::default(); N];
        let (x, xr) = x.as_chunks::<N>();
        let (y, yr) = y.as_chunks_mut::<N>();
        for (x, y) in x.iter().zip(y) {
            self.inner.0.block(&mut state.0, x, &mut u);
            self.inner.1.block(&mut state.1, &u, y);
        }
        debug_assert_eq!(xr.len(), yr.len());
        let ur = &mut u[..xr.len()];
        self.inner.0.block(&mut state.0, xr, ur);
        self.inner.1.block(&mut state.1, ur, yr);
    }
}

impl<X: Copy, U: Copy + Default, C0, C1, S0, S1, const N: usize> SplitInplace<X, (S0, S1)>
    for Major<(C0, C1), [U; N]>
where
    C0: SplitProcess<X, U, S0>,
    C1: SplitProcess<U, X, S1>,
{
    fn inplace(&self, state: &mut (S0, S1), xy: &mut [X]) {
        let mut u = [U::default(); N];
        let (xy, xyr) = xy.as_chunks_mut::<N>();
        for xy in xy {
            self.inner.0.block(&mut state.0, xy, &mut u);
            self.inner.1.block(&mut state.1, &u, xy);
        }
        let ur = &mut u[..xyr.len()];
        self.inner.0.block(&mut state.0, xyr, ur);
        self.inner.1.block(&mut state.1, ur, xyr);
    }
}
