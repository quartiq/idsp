use crate::{Inplace, Process};

//////////// ELEMENTARY PROCESSORS ////////////

/// Sum outputs of filters
///
/// Fan in.
#[derive(Debug, Copy, Clone, Default)]
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
#[derive(Debug, Copy, Clone, Default)]
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
#[derive(Debug, Copy, Clone, Default)]
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
#[derive(Debug, Copy, Clone, Default)]
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
#[derive(Debug, Copy, Clone, Default)]
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

/// Fan out
impl<X: Copy> Process<X, (X, X)> for &Identity {
    fn process(&mut self, x: X) -> (X, X) {
        (x, x)
    }
}

/// Fan out
impl<X: Copy, const N: usize> Process<X, [X; N]> for &Identity {
    fn process(&mut self, x: X) -> [X; N] {
        core::array::repeat(x)
    }
}

impl<X: Copy> Process<[X; 1], X> for &Identity {
    fn process(&mut self, x: [X; 1]) -> X {
        x[0]
    }
}

/// Inversion using `Neg`.
#[derive(Debug, Copy, Clone, Default)]
pub struct Neg;
impl<T: Copy + core::ops::Neg<Output = T>> Process<T> for &Neg {
    fn process(&mut self, x: T) -> T {
        x.neg()
    }
}

impl<T: Copy> Inplace<T> for &Neg where Self: Process<T> {}

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
#[derive(Debug, Copy, Clone, Default)]
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
#[derive(Debug, Copy, Clone, Default)]
pub struct Rate;
impl<X: Copy, const N: usize> Process<[X; N], X> for &Rate {
    fn process(&mut self, x: [X; N]) -> X {
        x[N - 1]
    }
}

impl<X: Copy + Default, const N: usize> Process<X, [X; N]> for &Rate {
    fn process(&mut self, x: X) -> [X; N] {
        let mut y = [X::default(); N];
        y[0] = x;
        y
    }
}
impl<X: Copy> Inplace<X> for &Rate where Self: Process<X> {}

/// Buffer input or output, or fixed delay line
#[derive(Debug, Copy, Clone, Default)]
pub struct Buffer<B> {
    buffer: B,
    idx: usize,
}

impl<X, const N: usize> Buffer<[X; N]> {
    /// The buffer is empty
    pub fn is_empty(&self) -> bool {
        self.idx == 0
    }
}

/// Delay line
impl<X: Copy, const N: usize> Process<X> for Buffer<[X; N]> {
    fn process(&mut self, x: X) -> X {
        let y = core::mem::replace(&mut self.buffer[self.idx], x);
        self.idx = (self.idx + 1) % N;
        y
    }

    // TODO: block(), inplace(), Process<[X; M]>
}

impl<X: Copy, const N: usize> Inplace<X> for Buffer<[X; N]> where Self: Process<X> {}

/// Buffer into chunks
impl<X: Copy, const N: usize> Process<X, Option<[X; N]>> for Buffer<[X; N]> {
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

/// Buffer out of chunks
///
/// Panics on underflow
impl<X: Copy, const N: usize> Process<Option<[X; N]>, X> for Buffer<[X; N]> {
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

/// Nyquist zero with gain 2
///
/// Bad for large differential delays
#[derive(Debug, Copy, Clone, Default)]
pub struct Nyquist<X>(
    /// Previous input
    pub X,
);
impl<X: Copy + core::ops::Add<X, Output = Y>, Y, const N: usize> Process<X, Y> for Nyquist<[X; N]> {
    fn process(&mut self, x: X) -> Y {
        const { assert!(N > 0) }
        let y = x + self.0[N - 1];
        self.0.copy_within(..N - 1, 1);
        self.0[0] = x;
        y
    }

    // TODO: block, inplace
}
impl<X: Copy> Inplace<X> for Nyquist<X> where Self: Process<X> {}

/// Integrator
#[derive(Debug, Copy, Clone, Default)]
pub struct Integrator<Y>(
    /// Current integrator value
    pub Y,
);
impl<X: Copy, Y: core::ops::AddAssign<X> + Copy> Process<X, Y> for Integrator<Y> {
    fn process(&mut self, x: X) -> Y {
        self.0 += x;
        self.0
    }
}
impl<X: Copy> Inplace<X> for Integrator<X> where Self: Process<X> {}

/// Comb (derivative)
///
/// Bad for large delays
#[derive(Debug, Copy, Clone, Default)]
pub struct Comb<X>(
    /// Delay line
    pub X,
);
impl<X: Copy + core::ops::Sub<X, Output = Y>, Y, const N: usize> Process<X, Y> for Comb<[X; N]> {
    fn process(&mut self, x: X) -> Y {
        const { assert!(N > 0) }
        let y = x - self.0[N - 1];
        self.0.copy_within(..N - 1, 1);
        self.0[0] = x;
        y
    }

    // TODO: block, inplace
}
impl<X: Copy> Inplace<X> for Comb<X> where Self: Process<X> {}
