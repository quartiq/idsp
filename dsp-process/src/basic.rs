use crate::{Inplace, Process, SplitInplace, SplitProcess};

//////////// ELEMENTARY PROCESSORS ////////////

/// Summation
///
/// Fan-in addition over tuples or arrays.
///
/// This is the standard reducer used after [`Parallel`](crate::Parallel) or
/// [`Pair`](crate::Pair) style branching.
#[derive(Debug, Copy, Clone, Default)]
pub struct Add;
impl<X: Copy, Y: core::iter::Sum<X>, const N: usize> Process<[X; N], Y> for Add {
    fn process(&mut self, x: [X; N]) -> Y {
        x.into_iter().sum()
    }
}
impl<X0: Copy + core::ops::Add<X1, Output = Y>, X1: Copy, Y> Process<(X0, X1), Y> for Add {
    fn process(&mut self, x: (X0, X1)) -> Y {
        x.0 + x.1
    }
}
impl<X: Copy> Inplace<X> for Add where Self: Process<X> {}

/// Product
///
/// Fan-in multiplication over tuples or arrays.
#[derive(Debug, Copy, Clone, Default)]
pub struct Mul;
impl<X: Copy, Y: core::iter::Product<X>, const N: usize> Process<[X; N], Y> for Mul {
    fn process(&mut self, x: [X; N]) -> Y {
        x.into_iter().product()
    }
}
impl<X0: Copy + core::ops::Mul<X1, Output = Y>, X1: Copy, Y> Process<(X0, X1), Y> for Mul {
    fn process(&mut self, x: (X0, X1)) -> Y {
        x.0 * x.1
    }
}
impl<X: Copy> Inplace<X> for Mul where Self: Process<X> {}

/// Difference
///
/// Fan-in subtraction for pairs.
#[derive(Debug, Copy, Clone, Default)]
pub struct Sub;
impl<X: Copy + core::ops::Sub<Output = Y>, Y> Process<[X; 2], Y> for Sub {
    fn process(&mut self, x: [X; 2]) -> Y {
        x[0] - x[1]
    }
}
impl<X0: Copy + core::ops::Sub<X1, Output = Y>, X1: Copy, Y> Process<(X0, X1), Y> for Sub {
    fn process(&mut self, x: (X0, X1)) -> Y {
        x.0 - x.1
    }
}
impl<X: Copy> Inplace<X> for Sub where Self: Process<X> {}

/// Sum and difference of a two-element input.
///
/// This is the classic butterfly primitive used in lattice, complementary, and
/// polyphase-style constructions.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Butterfly, Process};
///
/// assert_eq!(Butterfly.process([4, 1]), [5, 3]);
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct Butterfly;
impl<X: Copy + core::ops::Add<Output = Y> + core::ops::Sub<Output = Y>, Y> Process<[X; 2], [Y; 2]>
    for Butterfly
{
    fn process(&mut self, x: [X; 2]) -> [Y; 2] {
        [x[0] + x[1], x[0] - x[1]]
    }
}

impl<X: Copy> Inplace<X> for Butterfly where Self: Process<X> {}

/// Identity stage and simple fan-out/fan-in adapter.
///
/// Besides the usual `T -> T` identity, this type also provides several useful
/// shape conversions such as scalar fan-out to arrays or tuples.
#[derive(Debug, Copy, Clone, Default)]
pub struct Identity;
impl<T: Copy> Process<T> for Identity {
    fn process(&mut self, x: T) -> T {
        x
    }

    fn block(&mut self, x: &[T], y: &mut [T]) {
        y.copy_from_slice(x);
    }
}

/// NOP
impl<T: Copy> Inplace<T> for Identity {
    fn inplace(&mut self, _xy: &mut [T]) {}
}

/// Fan out
impl<X: Copy> Process<X, (X, X)> for Identity {
    fn process(&mut self, x: X) -> (X, X) {
        (x, x)
    }
}

/// Fan out
impl<X: Copy, const N: usize> Process<X, [X; N]> for Identity {
    fn process(&mut self, x: X) -> [X; N] {
        core::array::repeat(x)
    }
}

/// Flatten
impl<X: Copy> Process<[X; 1], X> for Identity {
    fn process(&mut self, x: [X; 1]) -> X {
        x[0]
    }
}

/// Inversion using `Neg`.
#[derive(Debug, Copy, Clone, Default)]
pub struct Neg;
impl<T: Copy + core::ops::Neg<Output = T>> Process<T> for Neg {
    fn process(&mut self, x: T) -> T {
        x.neg()
    }
}

impl<T: Copy> Inplace<T> for Neg where Self: Process<T> {}

/// Addition of a constant in split form.
///
/// See also [`Gain`] and [`Clamp`].
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct Offset<T>(pub T);

/// Offset using `Add`
impl<X: Copy + core::ops::Add<T, Output = Y>, Y, T: Copy> SplitProcess<X, Y> for Offset<T> {
    fn process(&self, _state: &mut (), x: X) -> Y {
        x + self.0
    }
}

impl<X: Copy, T> SplitInplace<X> for Offset<T> where Self: SplitProcess<X> {}

/// Multiplication by a constant in split form.
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct Gain<T>(pub T);

/// Gain using `Mul`
impl<X: Copy + core::ops::Mul<T, Output = Y>, Y, T: Copy> SplitProcess<X, Y> for Gain<T> {
    fn process(&self, _state: &mut (), x: X) -> Y {
        x * self.0
    }
}

impl<X: Copy, T> SplitInplace<X> for Gain<T> where Self: SplitProcess<X> {}

/// Clamp between min and max using `Ord`
///
/// This is a split-state combinator because the bounds are immutable
/// configuration.
#[derive(Debug, Copy, Clone, Default)]
pub struct Clamp<T> {
    /// Lowest output value
    pub min: T,
    /// Highest output value
    pub max: T,
}

impl<T: Copy + Ord> SplitProcess<T> for Clamp<T> {
    fn process(&self, _state: &mut (), x: T) -> T {
        x.clamp(self.min, self.max)
    }
}

impl<T: Copy> SplitInplace<T> for Clamp<T> where Self: SplitProcess<T> {}

/// Select or place one sample in a fixed-size rate-conversion slot.
///
/// As `[X; N] -> X`, it keeps the last sample. As `X -> [X; N]`, it emits the
/// sample in slot `0` and fills the rest with `Default::default()`.
#[derive(Debug, Copy, Clone, Default)]
pub struct Rate;
impl<X: Copy, const N: usize> Process<[X; N], X> for Rate {
    fn process(&mut self, x: [X; N]) -> X {
        x[N - 1]
    }
}

impl<X: Copy + Default, const N: usize> Process<X, [X; N]> for Rate {
    fn process(&mut self, x: X) -> [X; N] {
        let mut y = [X::default(); N];
        y[0] = x;
        y
    }
}
impl<X: Copy> Inplace<X> for Rate where Self: Process<X> {}

/// Fixed-size sample buffer used as a delay line or chunk accumulator.
///
/// The exact behavior depends on the chosen `Process` implementation:
///
/// * `X -> X`: delay line
/// * `X -> Option<[X; N]>`: buffer into chunks
/// * `Option<[X; N]> -> X`: stream samples out of a chunk buffer
#[derive(Debug, Copy, Clone, Default)]
pub struct Buffer<B> {
    buffer: B,
    idx: usize,
}

impl<X, const N: usize> Buffer<[X; N]> {
    /// Whether the chunk buffer is currently empty.
    ///
    /// For delay-line use this only reports whether the write index is at zero;
    /// it does not indicate whether previous samples are all defaults.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.idx == 0
    }
}

/// Delay line
///
/// This is the simplest stateful FIFO in the crate.
impl<X: Copy, const N: usize> Process<X> for Buffer<[X; N]> {
    fn process(&mut self, x: X) -> X {
        const { assert!(N > 0) }
        let y = core::mem::replace(&mut self.buffer[self.idx], x);
        self.idx = (self.idx + 1) % N;
        y
    }

    fn block(&mut self, x: &[X], y: &mut [X]) {
        const { assert!(N > 0) }
        debug_assert_eq!(x.len(), y.len());
        let mut x = x;
        let mut y = y;

        if self.idx != 0 {
            let n = x.len().min(N - self.idx);
            let (xh, xr) = x.split_at(n);
            let (yh, yr) = y.split_at_mut(n);
            yh.copy_from_slice(&self.buffer[self.idx..self.idx + n]);
            self.buffer[self.idx..self.idx + n].copy_from_slice(xh);
            self.idx = (self.idx + n) % N;
            x = xr;
            y = yr;
        }

        let (xc, xt) = x.as_chunks::<N>();
        let (yc, yt) = y.as_chunks_mut::<N>();
        for (xc, yc) in xc.iter().zip(yc) {
            *yc = self.buffer;
            self.buffer = *xc;
        }

        yt.copy_from_slice(&self.buffer[..xt.len()]);
        self.buffer[..xt.len()].copy_from_slice(xt);
        self.idx = xt.len();
    }
}

impl<X: Copy, const N: usize> Inplace<X> for Buffer<[X; N]> {
    fn inplace(&mut self, xy: &mut [X]) {
        const { assert!(N > 0) }
        let mut xy = xy;

        if self.idx != 0 {
            let n = xy.len().min(N - self.idx);
            let (head, rest) = xy.split_at_mut(n);
            for (xy, buf) in head
                .iter_mut()
                .zip(self.buffer[self.idx..self.idx + n].iter_mut())
            {
                core::mem::swap(xy, buf);
            }
            self.idx = (self.idx + n) % N;
            xy = rest;
        }

        let (chunks, tail) = xy.as_chunks_mut::<N>();
        for chunk in chunks {
            core::mem::swap(chunk, &mut self.buffer);
        }

        let n = tail.len();
        for (xy, buf) in tail.iter_mut().zip(self.buffer[..n].iter_mut()) {
            core::mem::swap(xy, buf);
        }
        self.idx = n;
    }
}

impl<X: Copy, const N: usize, const M: usize> Process<[X; M]> for Buffer<[X; N]> {
    fn process(&mut self, x: [X; M]) -> [X; M] {
        const { assert!(N > 0) }
        let mut y = x;
        <Self as Process<X>>::block(self, &x, &mut y);
        y
    }
}

/// Buffer into chunks
///
/// Returns `Some(chunk)` every `N` samples and `None` otherwise.
impl<X: Copy, const N: usize> Process<X, Option<[X; N]>> for Buffer<[X; N]> {
    fn process(&mut self, x: X) -> Option<[X; N]> {
        const { assert!(N > 0) }
        self.buffer[self.idx] = x;
        self.idx += 1;
        (self.idx == N).then(|| {
            self.idx = 0;
            self.buffer
        })
    }

    fn block(&mut self, x: &[X], y: &mut [Option<[X; N]>]) {
        const { assert!(N > 0) }
        debug_assert_eq!(x.len(), y.len());
        let mut x = x;
        let mut y = y;

        if self.idx != 0 {
            let n = x.len().min(N - self.idx);
            let (xh, xr) = x.split_at(n);
            let (yh, yr) = y.split_at_mut(n);
            self.buffer[self.idx..self.idx + n].copy_from_slice(xh);
            yh.fill(None);
            self.idx += n;
            if self.idx == N {
                self.idx = 0;
                yh[n - 1] = Some(self.buffer);
            }
            x = xr;
            y = yr;
        }

        let (xc, xt) = x.as_chunks::<N>();
        let (yc, yt) = y.as_chunks_mut::<N>();
        for (xc, yc) in xc.iter().zip(yc) {
            let (yl, yr) = yc.split_last_mut().unwrap();
            yr.fill(None);
            *yl = Some(*xc);
        }

        self.buffer[..xt.len()].copy_from_slice(xt);
        yt.fill(None);
        self.idx = xt.len();
    }
}

impl<X: Copy, const N: usize> Process<Option<[X; N]>, X> for Buffer<[X; N]> {
    fn process(&mut self, x: Option<[X; N]>) -> X {
        const { assert!(N > 0) }
        if let Some(x) = x {
            self.buffer = x;
            self.idx = 0;
        } else {
            self.idx += 1;
        }
        self.buffer[self.idx]
    }

    fn block(&mut self, x: &[Option<[X; N]>], y: &mut [X]) {
        const { assert!(N > 0) }
        debug_assert_eq!(x.len(), y.len());
        let mut i = 0;
        while i < x.len() {
            if let Some(buf) = x[i] {
                self.buffer = buf;
                self.idx = 0;
                y[i] = self.buffer[0];
                i += 1;
                continue;
            }

            let run = x[i..]
                .iter()
                .position(Option::is_some)
                .unwrap_or(x.len() - i);
            y[i..i + run].copy_from_slice(&self.buffer[self.idx + 1..self.idx + 1 + run]);
            self.idx += run;
            i += run;
        }
    }
}

/// Nyquist zero with gain 2
///
/// This is inefficient for large differential delays
///
/// See also [`Comb`] for the corresponding difference operator.
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

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len().min(N);
        let (xh, xt) = x.split_at(n);
        let (yh, yt) = y.split_at_mut(n);

        for ((xi, yi), s) in xh.iter().zip(yh.iter_mut()).zip(self.0[..n].iter().rev()) {
            *yi = *xi + *s;
        }
        for ((xi, yi), xp) in xt.iter().zip(yt.iter_mut()).zip(x.iter()) {
            *yi = *xi + *xp;
        }

        if x.len() >= N {
            for (dst, src) in self.0.iter_mut().zip(x[x.len() - N..].iter().rev()) {
                *dst = *src;
            }
        } else {
            self.0.copy_within(..N - x.len(), x.len());
            for (dst, src) in self.0.iter_mut().zip(x.iter().rev()) {
                *dst = *src;
            }
        }
    }

    // TODO: inplace()
}
impl<X: Copy, const N: usize> Inplace<X> for Nyquist<[X; N]> where Self: Process<X> {}

/// Running sum / discrete-time integrator.
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
///
/// See also [`Nyquist`] for the sum form.
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

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len().min(N);
        let (xh, xt) = x.split_at(n);
        let (yh, yt) = y.split_at_mut(n);

        for ((xi, yi), s) in xh.iter().zip(yh.iter_mut()).zip(self.0[..n].iter().rev()) {
            *yi = *xi - *s;
        }
        for ((xi, yi), xp) in xt.iter().zip(yt.iter_mut()).zip(x.iter()) {
            *yi = *xi - *xp;
        }

        if x.len() >= N {
            for (dst, src) in self.0.iter_mut().zip(x[x.len() - N..].iter().rev()) {
                *dst = *src;
            }
        } else {
            self.0.copy_within(..N - x.len(), x.len());
            for (dst, src) in self.0.iter_mut().zip(x.iter().rev()) {
                *dst = *src;
            }
        }
    }

    // TODO: inplace()
}
impl<X: Copy, const N: usize> Inplace<X> for Comb<[X; N]> where Self: Process<X> {}
