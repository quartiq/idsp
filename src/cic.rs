use core::iter::{repeat, Repeat, Take};

use num_traits::{Num, WrappingAdd};

/// Combs stage
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct Comb<T, const N: usize>([T; N]);

impl<T: Num + Copy, const N: usize> Default for Comb<T, N> {
    fn default() -> Self {
        Self([T::zero(); N])
    }
}

impl<T: Num + Copy, const N: usize> Comb<T, N> {
    /// Ingest a new sample into the filter and return its current output.
    pub fn update(&mut self, x: T) -> T {
        self.0.iter_mut().fold(x, |x, c| {
            let y = x - *c;
            *c = x;
            y
        })
    }
}

/// Integrators stage
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct Integrator<T, const N: usize>([T; N]);

impl<T: Num + Copy, const N: usize> Default for Integrator<T, N> {
    fn default() -> Self {
        Self([T::zero(); N])
    }
}

impl<T: Num + WrappingAdd + Copy, const N: usize> Integrator<T, N> {
    /// Ingest a new sample into the filter and return its current output.
    pub fn update(&mut self, x: T) -> T {
        self.0.iter_mut().fold(x, |x, i| {
            *i = x.wrapping_add(i);
            *i
        })
    }
}

/// Cascaded integrator comb interpolatpr
///
/// Order `N` where `N = 3` is cubic.
#[derive(Clone, Debug)]
pub struct CicInterpolator<T: Num + Copy, const N: usize> {
    rate: usize,
    combs: Comb<T, N>,
    up: Take<Repeat<T>>,
    integrators: Integrator<T, N>,
}

impl<T: Num + WrappingAdd + Copy, const N: usize> CicInterpolator<T, N> {
    /// Return the filter gain
    pub fn gain(&self) -> usize {
        self.rate.pow(N as _)
    }

    /// Create a new zero-initialized filter with the given rate change.
    pub fn new(rate: usize) -> Self {
        Self {
            rate,
            combs: Default::default(),
            up: repeat(T::zero()).take(rate),
            integrators: Default::default(),
        }
    }

    /// Optionally ingest a new low-rate sample and
    /// retrieve the next output.
    pub fn update(&mut self, x: Option<T>) -> T {
        if let Some(x) = x {
            self.up = repeat(self.combs.update(x)).take(self.rate);
        }
        self.integrators.update(self.up.next().unwrap())
    }
}

/// Cascaded integrator comb decimator.
///
/// Order `N` where `N = 3` is cubic.
#[derive(Clone, Debug)]
pub struct CicDecimator<T: Num + Copy, const N: usize> {
    rate: usize,
    integrators: Integrator<T, N>,
    index: usize,
    combs: Comb<T, N>,
}

impl<T: Num + WrappingAdd + Copy, const N: usize> CicDecimator<T, N> {
    /// Return the filter gain
    pub fn gain(&self) -> usize {
        self.rate.pow(N as _)
    }

    /// Create a new zero-initialized filter with the given rate change.
    pub fn new(rate: usize) -> Self {
        Self {
            rate,
            combs: Default::default(),
            index: 0,
            integrators: Default::default(),
        }
    }

    /// Ingest a new high-rate sample and optionally retrieve next output.
    pub fn update(&mut self, x: T) -> Option<T> {
        let x = self.integrators.update(x);
        if self.index == 0 {
            self.index = self.rate;
            Some(self.combs.update(x))
        } else {
            self.index -= 1;
            None
        }
    }
}
