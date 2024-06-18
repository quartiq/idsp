use core::ops::AddAssign;

use num_traits::{Num, WrappingAdd};

/// Cascaded integrator comb interpolatpr
///
/// Order `N` where `N = 3` is cubic.
#[derive(Clone, Debug)]
pub struct CicInterpolator<T, const N: usize> {
    /// Rate change (> 0)
    rate: usize,
    /// Comb/differentiator stages
    combs: [T; N],
    /// Zero order hold in the middle combined with the upsampler
    /// This is equivalent to a single comb + (delta-)upsampler + integrator
    zoh: T,
    /// Rate change index (count down)
    index: usize,
    /// Integrator stages
    integrators: [T; N],
}

impl<T: Num + AddAssign + Copy, const N: usize> CicInterpolator<T, N> {
    /// Create a new zero-initialized filter with the given rate change.
    pub fn new(rate: usize) -> Self {
        debug_assert!(rate > 0);
        Self {
            rate,
            combs: [T::zero(); N],
            zoh: T::zero(),
            index: 0,
            integrators: [T::zero(); N],
        }
    }

    /// Optionally ingest a new low-rate sample and
    /// retrieve the next output.
    pub fn update(&mut self, x: Option<T>) -> T {
        if let Some(x) = x {
            let x = self.combs.iter_mut().fold(x, |x, c| {
                let y = x - *c;
                *c = x;
                y
            });
            debug_assert_eq!(self.index, 0);
            self.index = self.rate - 1;
            self.zoh = x;
        } else {
            self.index -= 1;
        }
        self.integrators.iter_mut().fold(self.zoh, |x, i| {
            *i += x;
            *i
        })
    }

    /// Return the filter gain
    pub const fn gain(&self) -> usize {
        self.rate.pow(N as _)
    }
}

/// Cascaded integrator comb decimator.
///
/// Order `N` where `N = 3` is cubic.
#[derive(Clone, Debug)]
pub struct CicDecimator<T, const N: usize> {
    /// Rate change (> 0)
    rate: usize,
    /// Integration stages
    integrators: [T; N],
    /// Rate change count down
    index: usize,
    /// Comb/differentiator stages
    combs: [T; N],
}

impl<T: Num + WrappingAdd + Copy, const N: usize> CicDecimator<T, N> {
    /// Create a new zero-initialized filter with the given rate change.
    pub fn new(rate: usize) -> Self {
        debug_assert!(rate > 0);
        Self {
            rate,
            combs: [T::zero(); N],
            index: 0,
            integrators: [T::zero(); N],
        }
    }

    /// Ingest a new high-rate sample and optionally retrieve next output.
    pub fn update(&mut self, x: T) -> Option<T> {
        let x = self.integrators.iter_mut().fold(x, |x, i| {
            // Overflow is OK if bitwidth is sufficient (input * gain)
            *i = i.wrapping_add(&x);
            *i
        });
        if self.index == 0 {
            self.index = self.rate - 1;
            let x = self.combs.iter_mut().fold(x, |x, c| {
                let y = x - *c;
                *c = x;
                y
            });
            Some(x)
        } else {
            self.index -= 1;
            None
        }
    }

    /// Return the filter gain
    pub fn gain(&self) -> usize {
        self.rate.pow(N as _)
    }
}
