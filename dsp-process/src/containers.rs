use core::marker::PhantomData;
use crate::{Process, Inplace, Assert};

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

