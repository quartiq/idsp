use core::ops::AddAssign;

use num_traits::{AsPrimitive, Num, WrappingAdd};

/// Cascaded integrator comb structure
///
/// Order `N` where `N = 3` is cubic.
#[derive(Clone, Debug)]
pub struct Cic<T, const N: usize> {
    /// Rate change (> 0)
    rate: usize,
    /// Comb/differentiator stages
    combs: [T; N],
    /// Zero order hold behind comb sections.
    /// Interpolator: In the middle combined with the upsampler
    /// Decimator: After combs to support `get()`
    /// This is equivalent to a single comb + (delta-)upsampler + integrator
    zoh: T,
    /// Rate change index (count down)
    index: usize,
    /// Integrator stages
    integrators: [T; N],
}

impl<T, const N: usize> Cic<T, N>
where
    T: Num + AddAssign + WrappingAdd + Copy + 'static,
    usize: AsPrimitive<T>,
{
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

    pub fn rate(&self) -> usize {
        self.rate
    }

    pub fn set_rate(&mut self, rate: usize) {
        debug_assert!(rate > 0);
        self.rate = rate;
    }

    pub fn clear(&mut self) {
        self.zoh = T::zero();
        self.combs = [T::zero(); N];
        self.integrators = [T::zero(); N];
    }

    pub fn settle_interpolate(&mut self, x: T) {
        self.clear();
        let g = self.gain();
        let i = self.integrators.last_mut().unwrap_or(&mut self.zoh);
        *i = x * g;
    }

    pub fn settle_decimate(&mut self, x: T) {
        self.clear();
        self.zoh = x * self.gain();
    }

    /// Optionally ingest a new low-rate sample and
    /// retrieve the next output.
    pub fn interpolate(&mut self, x: Option<T>) -> T {
        if let Some(x) = x {
            debug_assert_eq!(self.index, 0);
            self.index = self.rate - 1;
            let x = self.combs.iter_mut().fold(x, |x, c| {
                let y = x - *c;
                *c = x;
                y
            });
            self.zoh = x;
        } else {
            self.index -= 1;
        }
        self.integrators.iter_mut().fold(self.zoh, |x, i| {
            *i += x;
            *i
        })
    }

    /// Ingest a new high-rate sample and optionally retrieve next output.
    pub fn decimate(&mut self, x: T) -> Option<T> {
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
            self.zoh = x;
            Some(self.zoh)
        } else {
            self.index -= 1;
            None
        }
    }

    pub fn tick(&self) -> bool {
        self.index == 0
    }

    pub fn get_interpolate(&self) -> T {
        *self.integrators.last().unwrap_or(&self.zoh)
    }

    pub fn get_decimate(&self) -> T {
        self.zoh
    }

    /// Return the filter gain
    pub fn gain(&self) -> T {
        self.rate.pow(N as _).as_()
    }

    /// Return the right shift amount (the log2 of gain())
    ///
    /// Panics if the gain is not a power of two.
    pub fn log2_gain(&self) -> u32 {
        let s = (31 - self.rate.leading_zeros()) * N as u32;
        debug_assert_eq!(1 << s, self.rate.pow(N as _) as _);
        s
    }

    /// Return the impulse response length
    pub fn response_length(&self) -> usize {
        N * (self.rate - 1)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn new() {
        let _ = Cic::<i64, 3>::new(1);
    }

    #[test]
    fn identity() {
        let mut int = Cic::<i64, 3>::new(1);
        let mut dec = Cic::<i64, 3>::new(1);
        for x in 0..100 {
            assert_eq!(int.interpolate(Some(x)), x);
            assert_eq!(dec.decimate(x), Some(x));
            assert_eq!(x, int.get_interpolate());
            assert_eq!(x, dec.get_decimate());
        }
    }

    #[test]
    fn response_length_gain() {
        let mut int = Cic::<i64, 3>::new(33);
        let x = 99;
        for i in 0..2 * int.response_length() {
            let y = int.interpolate(if int.tick() { Some(x) } else { None });
            assert_eq!(y, int.get_interpolate());
            if i < int.response_length() {
                assert!(y < x * int.gain());
            } else {
                assert_eq!(y, x * int.gain());
            }
        }
    }

    #[test]
    fn settle() {
        let x = 99;
        let mut int = Cic::<i64, 3>::new(33);
        int.settle_interpolate(x);
        let mut dec = Cic::<i64, 3>::new(33);
        dec.settle_decimate(x);
        for _ in 0..100 {
            let y = int.interpolate(if int.tick() { Some(x) } else { None });
            assert_eq!(y, x*int.gain());
            assert_eq!(y, int.get_interpolate());
            assert_eq!(dec.get_decimate(), x*dec.gain());
            if let Some(y) = dec.decimate(x) {
                assert_eq!(y, dec.get_decimate());
            }
        }

    }
}
