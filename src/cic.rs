use core::ops::AddAssign;

use num_traits::{AsPrimitive, Num, Pow, WrappingAdd, WrappingSub, Zero};

use crate::iir::Process;

/// Cascaded integrator comb structure
///
/// Order `N` where `N = 3` is cubic.
/// Comb delay `M` where `M = 1` is typical.
/// Use `rate=0` and some larger `M` to implemnent a unit-rate lowpass.
#[derive(Clone, Debug)]
pub struct Cic<T, const N: usize, const M: usize = 1> {
    /// Rate change (fast/slow - 1)
    /// Interpolator: output/input - 1
    /// Decimator: input/output - 1
    rate: u32,
    /// Up/downsampler state (count down)
    index: u32,
    /// Zero order hold behind comb sections.
    /// Interpolator: Combined with the upsampler
    /// Decimator: To support `get_decimate()`
    zoh: T,
    /// Comb/differentiator state
    combs: [[T; M]; N],
    /// Integrator state
    integrators: [T; N],
}

impl<T, const N: usize, const M: usize> Cic<T, N, M>
where
    T: Num + AddAssign + WrappingAdd + WrappingSub + Pow<usize, Output = T> + Copy + 'static,
    u32: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    const _M: () = assert!(M > 0, "Comb delay must be non-zero");

    /// Create a new zero-initialized filter with the given rate change.
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            index: 0,
            zoh: T::zero(),
            combs: [[T::zero(); M]; N],
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

    /// Comb delay
    pub const fn comb_delay(&self) -> usize {
        M
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
        *self = Self::new(self.rate);
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
        self.integrators.last().copied().unwrap_or(self.zoh)
    }

    /// Current decimator output
    pub fn get_decimate(&self) -> T {
        self.zoh
    }

    /// Filter gain
    pub fn gain(&self) -> T {
        (M.as_() * (self.rate.as_() + T::one())).pow(N)
    }

    /// Right shift amount
    ///
    /// `log2(gain())` if gain is a power of two,
    /// otherwise an upper bound.
    pub const fn gain_log2(&self) -> u32 {
        (u32::BITS - (M as u32 * self.rate + (M - 1) as u32).leading_zeros()) * N as u32
    }

    /// Impulse response length
    pub const fn response_length(&self) -> usize {
        self.rate as usize * N
    }

    /// Establish a settled filter state
    pub fn settle_interpolate(&mut self, x: T) {
        self.clear();
        if let Some(c) = self.combs.first_mut() {
            *c = [x; M];
        } else {
            self.zoh = x;
        }
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
}

/// Optionally ingest a new low-rate sample and
/// retrieve the next output.
///
/// A new sample must be supplied at the correct time (when [`Cic::tick()`] is true)
impl<T, const N: usize, const M: usize> Process<Option<T>, T> for Cic<T, N, M>
where
    T: Num + AddAssign + Copy,
{
    #[inline]
    fn process(&mut self, x: Option<T>) -> T {
        if let Some(x) = x {
            debug_assert_eq!(self.index, 0);
            self.index = self.rate;
            self.zoh = self.combs.iter_mut().fold(x, |x, c| {
                let y = x - c[0];
                c.copy_within(1.., 0);
                c[M - 1] = x;
                y
            });
        } else {
            self.index -= 1;
        }
        self.integrators.iter_mut().fold(self.zoh, |x, i| {
            // Overflow is not OK
            *i += x;
            *i
        })
    }
}

impl<T, const N: usize, const M: usize, const R: usize> Process<T, [T; R]> for Cic<T, N, M>
where
    T: Num + AddAssign + Copy,
    [T; R]: Default,
{
    #[inline]
    fn process(&mut self, x: T) -> [T; R] {
        let mut y = <[T; R]>::default();
        if let Some((y0, yr)) = y.split_first_mut() {
            *y0 = self.process(Some(x));
            for y in yr.iter_mut() {
                *y = self.process(None);
            }
        }
        y
    }
}

/// Ingest a new high-rate sample and optionally retrieve next output.
impl<T, const N: usize, const M: usize> Process<T, Option<T>> for Cic<T, N, M>
where
    T: WrappingAdd + WrappingSub + Copy,
{
    #[inline]
    fn process(&mut self, x: T) -> Option<T> {
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
            self.zoh = self.combs.iter_mut().fold(x, |x, c| {
                // Overflows expected
                let y = x.wrapping_sub(&c[0]);
                c.copy_within(1.., 0);
                c[M - 1] = x;
                y
            });
            Some(self.zoh)
        }
    }
}

impl<T, const N: usize, const M: usize, const R: usize> Process<[T; R], T> for Cic<T, N, M>
where
    T: WrappingAdd + WrappingSub + Copy + Zero,
{
    #[inline]
    fn process(&mut self, x: [T; R]) -> T {
        for x in x {
            if let Some(y) = self.process(x) {
                return y;
            }
        }
        T::zero()
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
            assert_eq!(Some(x), dec.process(x));
            assert_eq!(x, dec.get_decimate());
        }
    }

    #[quickcheck]
    fn identity_int(x: Vec<i64>) {
        const N: usize = 3;
        let mut int = Cic::<_, N>::new(0);
        for x in x {
            assert_eq!(x >> N, int.process(Some(x >> N)));
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
                int.process(None);
            }
            let y_last = int.get_interpolate();
            let y_want = x as i64 * int.gain();
            for i in 0..2 * int.response_length() {
                let y = int.process(if int.tick() { Some(x as i64) } else { None });
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
            let y = int.process(if int.tick() { Some(x as _) } else { None });
            assert_eq!(y, x as i64 * int.gain());
            assert_eq!(y, int.get_interpolate());
            // assert_eq!(dec.get_decimate(), x as i64 * dec.gain());
            // if let Some(y) = dec.decimate(x as _) {
            //     assert_eq!(y, x as i64 * dec.gain());
            // }
        }
    }

    #[quickcheck]
    fn unit_rate(x: (i32, i32, i32, i32, i32)) {
        let x: [i32; 5] = x.into();
        let mut cic = Cic::<i64, 3, 3>::new(0);
        assert!(cic.gain_log2() == 6);
        assert!(cic.gain() == (cic.comb_delay() as i64).pow(cic.order() as _));
        for x in x {
            assert!(cic.tick());
            let y: Option<_> = cic.process(x as _);
            println!("{x:11} {:11}", y.unwrap());
        }
        for _ in 0..100 {
            let y: Option<_> = cic.process(0 as _);
            assert_eq!(y, Some(cic.get_decimate()));
            println!("{:11}", y.unwrap());
            println!("");
        }
    }
}
