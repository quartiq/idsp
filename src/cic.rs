use core::ops::AddAssign;

use num_traits::{AsPrimitive, Num, Pow, WrappingAdd, WrappingSub};

use dsp_process::Process;

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

/// Ingest a new high-rate sample and optionally retrieve next output.
impl<T, const N: usize, const M: usize> Process<T, Option<T>> for Cic<T, N, M>
where
    T: WrappingAdd + WrappingSub + Copy,
{
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
            println!();
        }
    }
}

#[cfg(test)]
mod modular_tests {
    use super::*;
    use dsp_process::{Comb, Downsample, Hold, Integrator, Rate, Split, Unsplit};

    fn modular_decimator<const N: usize, const R: usize, const M: usize>()
    -> impl dsp_process::Process<[i64; R], i64> {
        let ints = Split::stateful(Integrator(0)).repeat::<N>().minor();
        let down = Split::new(Downsample((R - 1) as u32), 0);
        let combs = Split::stateful(Comb([0; M])).repeat::<N>().minor().map();
        let ints_down = (ints * down).minor::<i64>();
        let inner = (ints_down * combs).minor::<Option<i64>>();
        inner.decimate()
    }

    fn modular_decimator_chunked_prefix<const N: usize, const R: usize, const M: usize>()
    -> impl dsp_process::Process<[i64; R], i64> {
        let ints = Split::stateful(Integrator(0)).repeat::<N>().minor();
        let ints = ints.chunk();
        let phase = Split::stateful(Rate::<0>);
        let combs = Split::stateful(Comb([0; M])).repeat::<N>().minor();
        let ints_phase = (ints * phase).minor::<[i64; R]>();
        (ints_phase * combs).minor::<i64>()
    }

    fn modular_interpolator<const N: usize, const R: usize, const M: usize>()
    -> impl dsp_process::Process<i64, [i64; R]> {
        let combs = Split::stateful(Comb([0; M])).repeat::<N>().minor().map();
        let hold = Split::stateful(Hold(0));
        let ints = Split::stateful(Integrator(0)).repeat::<N>().minor();
        let combs_hold = (combs * hold).minor::<Option<i64>>();
        let inner = (combs_hold * ints).minor::<i64>();
        inner.interpolate()
    }

    fn modular_interpolator_chunked_suffix<const N: usize, const R: usize, const M: usize>()
    -> impl dsp_process::Process<i64, [i64; R]> {
        let combs = Split::stateful(Comb([0; M])).repeat::<N>().minor().map();
        let hold = Split::stateful(Hold(0));
        let front = (combs * hold).minor::<Option<i64>>();
        let front = front.interpolate();
        let ints = Split::stateful(Integrator(0)).repeat::<N>().minor();
        let ints = ints.chunk();
        (front * ints).minor::<[i64; R]>()
    }

    fn reference_decimator<const N: usize, const R: usize, const M: usize>()
    -> impl dsp_process::Process<[i64; R], i64> {
        Split::new((), Unsplit(Cic::<i64, N, M>::new((R - 1) as u32))).decimate()
    }

    fn reference_interpolator<const N: usize, const R: usize, const M: usize>()
    -> impl dsp_process::Process<i64, [i64; R]> {
        Split::new((), Unsplit(Cic::<i64, N, M>::new((R - 1) as u32))).interpolate()
    }

    fn verify_decimator<const N: usize, const R: usize, const M: usize>(x: &[i64]) {
        let mut modular = modular_decimator::<N, R, M>();
        let mut reference = reference_decimator::<N, R, M>();
        for chunk in x.chunks_exact(R) {
            let chunk: [i64; R] = chunk.try_into().unwrap();
            let y_mod = dsp_process::Process::process(&mut modular, chunk);
            let y_ref = reference.process(chunk);
            assert_eq!(y_mod, y_ref);
        }
    }

    fn verify_decimator_chunked_prefix<const N: usize, const R: usize, const M: usize>(x: &[i64]) {
        let mut modular = modular_decimator_chunked_prefix::<N, R, M>();
        let mut reference = reference_decimator::<N, R, M>();
        for chunk in x.chunks_exact(R) {
            let chunk: [i64; R] = chunk.try_into().unwrap();
            let y_mod = modular.process(chunk);
            let y_ref = reference.process(chunk);
            assert_eq!(y_mod, y_ref);
        }
    }

    fn verify_interpolator<const N: usize, const R: usize, const M: usize>(x: &[i64]) {
        let mut modular = modular_interpolator::<N, R, M>();
        let mut reference = reference_interpolator::<N, R, M>();
        for &x in x {
            let y_mod = dsp_process::Process::process(&mut modular, x);
            let y_ref = dsp_process::Process::process(&mut reference, x);
            assert_eq!(y_mod, y_ref);
        }
    }

    fn verify_interpolator_chunked_suffix<const N: usize, const R: usize, const M: usize>(
        x: &[i64],
    ) {
        let mut modular = modular_interpolator_chunked_suffix::<N, R, M>();
        let mut reference = reference_interpolator::<N, R, M>();
        for &x in x {
            let y_mod = dsp_process::Process::process(&mut modular, x);
            let y_ref = dsp_process::Process::process(&mut reference, x);
            assert_eq!(y_mod, y_ref);
        }
    }

    #[test]
    fn modular_decimator_matches_reference() {
        let x: Vec<i64> = (-31..65).map(|x| x * 3 - 7).collect();
        verify_decimator::<3, 4, 1>(&x);
        verify_decimator::<2, 2, 1>(&x);
        verify_decimator::<3, 1, 3>(&x);
    }

    #[test]
    fn modular_decimator_chunked_prefix_matches_reference() {
        let x: Vec<i64> = (-31..65).map(|x| x * 3 - 7).collect();
        verify_decimator_chunked_prefix::<3, 4, 1>(&x);
        verify_decimator_chunked_prefix::<2, 2, 1>(&x);
        verify_decimator_chunked_prefix::<3, 1, 3>(&x);
    }

    #[test]
    fn modular_interpolator_matches_reference() {
        let x: Vec<i64> = (-12..20).map(|x| x * x - 3 * x + 2).collect();
        verify_interpolator::<3, 4, 1>(&x);
        verify_interpolator::<2, 2, 1>(&x);
        verify_interpolator::<3, 1, 3>(&x);
    }

    #[test]
    fn modular_interpolator_chunked_suffix_matches_reference() {
        let x: Vec<i64> = (-12..20).map(|x| x * x - 3 * x + 2).collect();
        verify_interpolator_chunked_suffix::<3, 4, 1>(&x);
        verify_interpolator_chunked_suffix::<2, 2, 1>(&x);
        verify_interpolator_chunked_suffix::<3, 1, 3>(&x);
    }

    use core::hint::black_box;

    const PERF_N: usize = 3;
    const PERF_R: usize = 8;
    const PERF_D: usize = 1;
    const PERF_ITERS: usize = 1 << 28;

    /// Spelled-out CIC decimator.
    ///
    /// About 1.28 core cycles/input sample on the pinned-core host run.
    ///
    /// Run with:
    /// `taskset -c 1 perf stat -d target/release/deps/idsp-... cic::test::perf_tests::insn_dec_ref --ignored --exact`
    #[test]
    #[ignore]
    fn insn_dec_ref() {
        let mut proc = reference_decimator::<PERF_N, PERF_R, PERF_D>();
        let x = [9; PERF_R];
        for _ in 0..PERF_ITERS {
            black_box(proc.process(black_box(x)));
        }
    }

    /// Modular `dsp-process` CIC decimator.
    ///
    /// About 1.32 core cycles/input sample on the pinned-core host run.
    ///
    /// Run with:
    /// `taskset -c 1 perf stat -d target/release/deps/idsp-... cic::test::perf_tests::insn_dec_mod --ignored --exact`
    #[test]
    #[ignore]
    fn insn_dec_mod() {
        let mut proc = modular_decimator::<PERF_N, PERF_R, PERF_D>();
        let x = [9; PERF_R];
        for _ in 0..PERF_ITERS {
            black_box(proc.process(black_box(x)));
        }
    }

    /// Modular `dsp-process` CIC decimator with a chunked integrator prefix.
    ///
    /// About 2.02 core cycles/input sample on the pinned-core host run.
    ///
    /// Run with:
    /// `taskset -c 1 perf stat -d target/release/deps/idsp-... cic::test::perf_tests::insn_dec_mod_chunked_prefix --ignored --exact`
    #[test]
    #[ignore]
    fn insn_dec_mod_chunked_prefix() {
        let mut proc = modular_decimator_chunked_prefix::<PERF_N, PERF_R, PERF_D>();
        let x = [9; PERF_R];
        for _ in 0..PERF_ITERS {
            black_box(proc.process(black_box(x)));
        }
    }

    /// Spelled-out CIC interpolator.
    ///
    /// About 1.26 core cycles/output sample on the pinned-core host run.
    ///
    /// Run with:
    /// `taskset -c 1 perf stat -d target/release/deps/idsp-... cic::test::perf_tests::insn_int_ref --ignored --exact`
    #[test]
    #[ignore]
    fn insn_int_ref() {
        let mut proc = reference_interpolator::<PERF_N, PERF_R, PERF_D>();
        let x = 9;
        for _ in 0..PERF_ITERS {
            black_box(proc.process(black_box(x)));
        }
    }

    /// Modular `dsp-process` CIC interpolator.
    ///
    /// About 1.22 core cycles/output sample on the pinned-core host run.
    ///
    /// Run with:
    /// `taskset -c 1 perf stat -d target/release/deps/idsp-... cic::test::perf_tests::insn_int_mod --ignored --exact`
    #[test]
    #[ignore]
    fn insn_int_mod() {
        let mut proc = modular_interpolator::<PERF_N, PERF_R, PERF_D>();
        let x = 9;
        for _ in 0..PERF_ITERS {
            black_box(proc.process(black_box(x)));
        }
    }

    /// Modular `dsp-process` CIC interpolator with a chunked integrator suffix.
    ///
    /// About 4.39 core cycles/output sample on the pinned-core host run.
    ///
    /// Run with:
    /// `taskset -c 1 perf stat -d target/release/deps/idsp-... cic::test::perf_tests::insn_int_mod_chunked_suffix --ignored --exact`
    #[test]
    #[ignore]
    fn insn_int_mod_chunked_suffix() {
        let mut proc = modular_interpolator_chunked_suffix::<PERF_N, PERF_R, PERF_D>();
        let x = 9;
        for _ in 0..PERF_ITERS {
            black_box(proc.process(black_box(x)));
        }
    }
}
