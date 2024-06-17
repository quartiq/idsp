use core::iter::{repeat, Repeat, Take};

use num_traits::{Num, WrappingAdd};

/// Comb stage
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub struct Comb<T>(T);

impl<T: Num + Copy> Comb<T> {
    /// Ingest a new sample into the filter and return its current output.
    pub fn update(&mut self, x: T) -> T {
        let y = x - self.0;
        self.0 = x;
        y
    }
}

/// Integrators stage
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
pub struct Integrator<T>(T);

impl<T: Num + WrappingAdd + Copy> Integrator<T> {
    /// Ingest a new sample into the filter and return its current output.
    pub fn update(&mut self, x: T) -> T {
        self.0 = self.0.wrapping_add(&x);
        self.0
    }
}

/// Cascaded integrator comb interpolatpr
///
/// Order `N` where `N = 3` is cubic.
#[derive(Clone, Debug)]
pub struct CicInterpolator<T, const N: usize> {
    rate: usize,
    combs: [Comb<T>; N],
    up: Take<Repeat<T>>,
    integrators: [Integrator<T>; N],
}

impl<T: Num + WrappingAdd + Copy + Default, const N: usize> CicInterpolator<T, N> {
    /// Return the filter gain
    pub const fn gain(&self) -> usize {
        self.rate.pow(N as _)
    }

    /// Create a new zero-initialized filter with the given rate change.
    pub fn new(rate: usize) -> Self {
        Self {
            rate,
            combs: [Comb::default(); N],
            up: repeat(T::zero()).take(rate),
            integrators: [Integrator::default(); N],
        }
    }

    /// Optionally ingest a new low-rate sample and
    /// retrieve the next output.
    pub fn update(&mut self, x: Option<T>) -> T {
        if let Some(x) = x {
            let x = self.combs.iter_mut().fold(x, |x, c| c.update(x));
            self.up = repeat(x).take(self.rate);
        }
        self.integrators
            .iter_mut()
            .fold(self.up.next().unwrap(), |x, i| i.update(x))
    }
}

/// Cascaded integrator comb decimator.
///
/// Order `N` where `N = 3` is cubic.
#[derive(Clone, Debug)]
pub struct CicDecimator<T, const N: usize> {
    rate: usize,
    integrators: [Integrator<T>; N],
    index: usize,
    combs: [Comb<T>; N],
}

impl<T: Num + WrappingAdd + Copy + Default, const N: usize> CicDecimator<T, N> {
    /// Return the filter gain
    pub fn gain(&self) -> usize {
        self.rate.pow(N as _)
    }

    /// Create a new zero-initialized filter with the given rate change.
    pub fn new(rate: usize) -> Self {
        debug_assert!(rate > 0);
        Self {
            rate,
            combs: [Comb::default(); N],
            index: 0,
            integrators: [Integrator::default(); N],
        }
    }

    /// Ingest a new high-rate sample and optionally retrieve next output.
    pub fn update(&mut self, x: T) -> Option<T> {
        let x = self.integrators.iter_mut().fold(x, |x, i| i.update(x));
        if self.index == 0 {
            self.index = self.rate - 1;
            let x = self.combs.iter_mut().fold(x, |x, c| c.update(x));
            Some(x)
        } else {
            self.index -= 1;
            None
        }
    }
}
