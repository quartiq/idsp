use core::ops::AddAssign;

use num_traits::{AsPrimitive, Num, Pow, WrappingAdd, WrappingSub};

/// Cascaded integrator comb structure
///
/// Order `N` where `N = 3` is cubic.
#[derive(Clone, Debug)]
pub struct Cic<T, const N: usize> {
    /// Rate change (fast/slow - 1)
    /// Interpolator: output/input - 1
    /// Decimator: input/output - 1
    rate: u32,
    /// Up/downsampler state (count down)
    index: u32,
    /// Zero order hold behind comb sections.
    /// Interpolator: In the middle combined with the upsampler
    /// Decimator: After combs to support `get_decimate()`
    zoh: T,
    /// Comb/differentiator state
    combs: [T; N],
    /// Integrator state
    integrators: [T; N],
}

impl<T, const N: usize> Cic<T, N>
where
    T: Num + AddAssign + WrappingAdd + WrappingSub + Pow<usize, Output = T> + Copy + 'static,
    u32: AsPrimitive<T>,
{
    /// Create a new zero-initialized filter with the given rate change.
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            index: 0,
            zoh: T::zero(),
            combs: [T::zero(); N],
            integrators: [T::zero(); N],
        }
    }

    /// Filter order
    ///
    /// * 0: zero order hold
    /// * 1: linear
    /// * 2: quadratic
    /// * 3: cubic interpolation/decimation
    ///
    /// etc.
    pub const fn order(&self) -> usize {
        N
    }

    /// Rate change
    ///
    /// `fast/slow - 1`
    pub const fn rate(&self) -> u32 {
        self.rate
    }

    /// Set the rate change
    ///
    /// `fast/slow - 1`
    pub fn set_rate(&mut self, rate: u32) {
        self.rate = rate;
    }

    /// Zero-initialize the filter state
    pub fn clear(&mut self) {
        self.index = 0;
        self.zoh = T::zero();
        self.combs = [T::zero(); N];
        self.integrators = [T::zero(); N];
    }

    /// Accepts/provides new slow-rate sample
    ///
    /// Interpolator: accepts new input sample
    /// Decimator: returns new output sample
    pub const fn tick(&self) -> bool {
        self.index == 0
    }

    /// Current interpolator output
    pub fn get_interpolate(&self) -> T {
        *self.integrators.last().unwrap_or(&self.zoh)
    }

    /// Current decimator output
    pub fn get_decimate(&self) -> T {
        self.zoh
    }

    /// Filter gain
    pub fn gain(&self) -> T {
        (self.rate.as_() + T::one()).pow(N)
    }

    /// Right shift amount
    ///
    /// `log2(gain())` if gain is a power of two,
    /// otherwise an upper bound.
    pub const fn gain_log2(&self) -> u32 {
        (u32::BITS - self.rate.leading_zeros()) * N as u32
    }

    /// Impulse response length
    pub const fn response_length(&self) -> usize {
        self.rate as usize * N
    }

    /// Establish a settled filter state
    pub fn settle_interpolate(&mut self, x: T) {
        self.clear();
        *self.combs.first_mut().unwrap_or(&mut self.zoh) = x;
        let g = self.gain();
        if let Some(i) = self.integrators.last_mut() {
            *i = x * g;
        }
    }

    /// Establish a settled filter state
    ///
    /// Unimplemented!
    pub fn settle_decimate(&mut self, x: T) {
        self.clear();
        self.zoh = x * self.gain();
        unimplemented!();
    }

    /// Optionally ingest a new low-rate sample and
    /// retrieve the next output.
    ///
    /// A new sample must be supplied at the correct time (when [`Cic::tick()`] is true)
    pub fn interpolate(&mut self, x: Option<T>) -> T {
        if let Some(x) = x {
            debug_assert_eq!(self.index, 0);
            self.index = self.rate;
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
        if let Some(index) = self.index.checked_sub(1) {
            self.index = index;
            None
        } else {
            self.index = self.rate;
            let x = self.combs.iter_mut().fold(x, |x, c| {
                let y = x.wrapping_sub(c);
                *c = x;
                y
            });
            self.zoh = x;
            Some(self.zoh)
        }
    }
}

#[cfg(test)]
mod test {
    use core::cmp::Ordering;

    use super::*;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn new(rate: u32) {
        let _ = Cic::<i64, 3>::new(rate);
    }

    #[quickcheck]
    fn identity_dec(x: Vec<i64>) {
        let mut dec = Cic::<_, 3>::new(0);
        for x in x {
            assert_eq!(x, dec.decimate(x).unwrap());
            assert_eq!(x, dec.get_decimate());
        }
    }

    #[quickcheck]
    fn identity_int(x: Vec<i64>) {
        const N: usize = 3;
        let mut int = Cic::<_, N>::new(0);
        for x in x {
            assert_eq!(x >> N, int.interpolate(Some(x >> N)));
            assert_eq!(x >> N, int.get_interpolate());
        }
    }

    #[quickcheck]
    fn response_length_gain_settle(x: Vec<i32>, rate: u32) {
        let mut int = Cic::<_, 3>::new(rate);
        let shift = int.gain_log2();
        if shift >= 32 {
            return;
        }
        assert!(int.gain() <= 1 << shift);
        for x in x {
            while !int.tick() {
                int.interpolate(None);
            }
            let y_last = int.get_interpolate();
            let y_want = x as i64 * int.gain();
            for i in 0..2 * int.response_length() {
                let y = int.interpolate(if int.tick() { Some(x as i64) } else { None });
                assert_eq!(y, int.get_interpolate());
                if i < int.response_length() {
                    match y_want.cmp(&y_last) {
                        Ordering::Greater => assert!((y_last..y_want).contains(&y)),
                        Ordering::Less => assert!((y_want..y_last).contains(&(y - 1))),
                        Ordering::Equal => assert_eq!(y_want, y),
                    }
                } else {
                    assert_eq!(y, y_want);
                }
            }
        }
    }

    #[quickcheck]
    fn settle(rate: u32, x: i32) {
        let mut int = Cic::<i64, 3>::new(rate);
        if int.gain_log2() >= 32 {
            return;
        }
        int.settle_interpolate(x as _);
        // let mut dec = Cic::<i64, 3>::new(rate);
        // dec.settle_decimate(x as _);
        for _ in 0..100 {
            let y = int.interpolate(if int.tick() { Some(x as _) } else { None });
            assert_eq!(y, x as i64 * int.gain());
            assert_eq!(y, int.get_interpolate());
            // assert_eq!(dec.get_decimate(), x as i64 * dec.gain());
            // if let Some(y) = dec.decimate(x as _) {
            //     assert_eq!(y, x as i64 * dec.gain());
            // }
        }
    }
}
