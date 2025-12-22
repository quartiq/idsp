//! Half-band filters and cascades
//!
//! Used to perform very efficient high-dynamic range rate changes by powers of two.

use core::{
    iter::Sum,
    ops::{Add, Mul},
    slice::{from_mut, from_ref},
};

use dsp_process::SplitProcess;

/// Symmetric FIR filter prototype.
///
/// # Generics
/// * `M`: number of taps, one-sided. The filter has effectively 2*M DSP taps
/// * `N`: state size: N = 2*M - 1 + {input/output}.len()
///
/// # Half band decimation/interpolation filters
///
/// Half-band filters (rate change of 2) and cascades of HBFs are implemented in
/// [`HbfDec`] and [`HbfInt`] etc.
/// The half-band filter has unique properties that make it preferable in many cases:
///
/// * only needs M multiplications (fused multiply accumulate) for 4*M taps
/// * HBF decimator stores less state than a generic FIR filter
/// * as a FIR filter has linear phase/flat group delay
/// * very small passband ripple and excellent stopband attenuation
/// * as a cascade of decimation/interpolation filters, the higher-rate filters
///   need successively fewer taps, allowing the filtering to be dominated by
///   only the highest rate filter with the fewest taps
/// * In a cascade of HBF the overall latency, group delay, and impulse response
///   length are dominated by the lowest-rate filter which, due to its manageable transition
///   band width (compared to single-stage filters) can be smaller, shorter, and faster.
/// * high dynamic range and inherent stability compared with an IIR filter
/// * can be combined with a CIC filter for non-power-of-two or even higher rate changes
///
/// The implementations here are all `no_std` and `no-alloc`.
/// They support (but don't require) in-place filtering to reduce memory usage.
/// They unroll and optimize extremely well targetting current architectures,
/// e.g. requiring less than 4 instructions per input item for the full `HbfDecCascade` on Skylake.
/// The filters are optimized for decent block sizes and perform best (i.e. with negligible
/// overhead) for blocks of 32 high-rate items or more, depending very much on architecture.

#[derive(Clone, Debug, Copy)]
pub struct SymFir<C>(pub C);

impl<C, const M: usize> SymFir<[C; M]> {
    /// Response length, number of taps
    pub const fn len() -> usize {
        2 * M - 1
    }
}

impl<C, const M: usize> SymFir<[C; M]> {
    /// Perform the FIR convolution and yield results iteratively.
    #[inline]
    pub fn get<T>(&self, x: &[T]) -> impl Iterator<Item = T>
    where
        C: Copy + Mul<T, Output = T>,
        T: Copy + Add<Output = T> + Sum,
    {
        x.windows(2 * M).map(|x| {
            let (old, new) = x.split_at(M);
            old.iter()
                .zip(new.iter().rev())
                .zip(self.0.iter())
                .map(|((xo, xn), tap)| *tap * (*xo + *xn))
                .sum()
        })
    }
}

// TODO: pub struct SymFirInt<R>, SymFirDec<R>

/// Half band decimator (decimate by two) state
#[derive(Clone, Debug, Copy)]
pub struct HbfDec<T> {
    even: T, // at least N - M len
    odd: T,  // N > SymFir::len()
}

impl<T: Copy + Default, const N: usize> Default for HbfDec<[T; N]> {
    fn default() -> Self {
        Self {
            even: [T::default(); _],
            odd: [T::default(); _],
        }
    }
}

impl<
    C: Copy + Mul<T, Output = T>,
    T: Copy + Default + Add<Output = T> + Sum,
    const M: usize,
    const N: usize,
> SplitProcess<[T; 2], T, HbfDec<[T; N]>> for SymFir<[C; M]>
{
    fn block(&self, state: &mut HbfDec<[T; N]>, x: &[[T; 2]], y: &mut [T]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.chunks(N - Self::len()).zip(y.chunks_mut(N - Self::len())) {
            // load input
            for (xi, (even, odd)) in x.iter().zip(
                state.even[M - 1..]
                    .iter_mut()
                    .zip(state.odd[Self::len()..].iter_mut()),
            ) {
                [*even, *odd] = *xi;
            }
            // compute output
            for (yi, (even, odd)) in y
                .iter_mut()
                .zip(state.even.iter().copied().zip(self.get(&state.odd)))
            {
                *yi = even + odd;
            }
            // keep state
            state.even.copy_within(x.len()..x.len() + M - 1, 0);
            state.odd.copy_within(x.len()..x.len() + Self::len(), 0);
        }
    }

    fn process(&self, state: &mut HbfDec<[T; N]>, x: [T; 2]) -> T {
        let mut y = Default::default();
        self.block(state, from_ref(&x), from_mut(&mut y));
        y
    }
}

/// Half band interpolator (interpolation rate 2) state
#[derive(Clone, Debug, Copy)]
pub struct HbfInt<T> {
    x: T, // len N > SymFir::len()
}

impl<T: Default + Copy, const N: usize> Default for HbfInt<[T; N]> {
    fn default() -> Self {
        Self {
            x: [T::default(); _],
        }
    }
}

impl<
    C: Copy + Mul<T, Output = T>,
    T: Copy + Default + Add<Output = T> + Sum,
    const M: usize,
    const N: usize,
> SplitProcess<T, [T; 2], HbfInt<[T; N]>> for SymFir<[C; M]>
{
    fn block(&self, state: &mut HbfInt<[T; N]>, x: &[T], y: &mut [[T; 2]]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.chunks(N - Self::len()).zip(y.chunks_mut(N - Self::len())) {
            // load input
            state.x[Self::len()..Self::len() + x.len()].copy_from_slice(x);
            // compute output
            for (yi, (even, odd)) in y
                .iter_mut()
                .zip(self.get(&state.x).zip(state.x[M..].iter().copied()))
            {
                // Choose the even item to be the interpolated one.
                // The alternative would have the same response length
                // but larger latency.
                *yi = [even, odd]; // interpolated, center tap: identity
            }
            // keep state
            state.x.copy_within(x.len()..x.len() + Self::len(), 0);
        }
    }

    fn process(&self, state: &mut HbfInt<[T; N]>, x: T) -> [T; 2] {
        let mut y = Default::default();
        self.block(state, from_ref(&x), from_mut(&mut y));
        y
    }
}

/// Standard/optimal half-band filter cascade taps
///
/// * obtained with `2*signal.remez(4*n - 1, bands=(0, .5-df/2, .5+df/2, 1), desired=(1, 0), fs=2, grid_density=512)[:2*n:2]`
/// * more than 98 dB stop band attenuation (>16 bit)
/// * 0.4 pass band (relative to lowest sample rate)
/// * less than 0.001 dB ripple
/// * linear phase/flat group delay
/// * rate change up to 2**5 = 32
/// * lowest rate filter is at 0 index
/// * use taps 0..n for 2**n interpolation/decimation
#[allow(clippy::excessive_precision, clippy::type_complexity)]
pub const HBF_TAPS_98: (
    SymFir<[f32; 15]>,
    SymFir<[f32; 6]>,
    SymFir<[f32; 3]>,
    SymFir<[f32; 3]>,
    SymFir<[f32; 2]>,
) = (
    // n=15 coefficients (effective number of DSP taps 4*15-1 = 59), transition band width df=.2 fs
    SymFir([
        7.02144012e-05,
        -2.43279582e-04,
        6.35026936e-04,
        -1.39782541e-03,
        2.74613582e-03,
        -4.96403839e-03,
        8.41806912e-03,
        -1.35827601e-02,
        2.11004053e-02,
        -3.19267647e-02,
        4.77024289e-02,
        -7.18014345e-02,
        1.12942004e-01,
        -2.03279594e-01,
        6.33592923e-01,
    ]),
    // 6, .47
    SymFir([
        -0.00086943,
        0.00577837,
        -0.02201674,
        0.06357869,
        -0.16627679,
        0.61979312,
    ]),
    // 3, .754
    SymFir([0.01414651, -0.10439639, 0.59026742]),
    // 3, .877
    SymFir([0.01227974, -0.09930782, 0.58702834]),
    // 2, .94
    SymFir([-0.06291796, 0.5629161]),
);

pub type HbfTaps = (
    SymFir<[f32; 23]>,
    SymFir<[f32; 9]>,
    SymFir<[f32; 5]>,
    SymFir<[f32; 4]>,
    SymFir<[f32; 3]>,
);

/// * 140 dB stopband, 2 µdB passband ripple, limited by f32 dynamic range
/// * otherwise like [`HBF_TAPS_98`].
#[allow(clippy::excessive_precision, clippy::type_complexity)]
pub const HBF_TAPS: HbfTaps = (
    SymFir([
        7.60376281e-07,
        -3.77494189e-06,
        1.26458572e-05,
        -3.43188258e-05,
        8.10687488e-05,
        -1.72971471e-04,
        3.40845058e-04,
        -6.29522838e-04,
        1.10128836e-03,
        -1.83933298e-03,
        2.95124925e-03,
        -4.57290979e-03,
        6.87374175e-03,
        -1.00656254e-02,
        1.44199841e-02,
        -2.03025099e-02,
        2.82462332e-02,
        -3.91128510e-02,
        5.44795655e-02,
        -7.77002648e-02,
        1.17523454e-01,
        -2.06185386e-01,
        6.34588718e-01,
    ]),
    SymFir([
        3.13788260e-05,
        -2.90598691e-04,
        1.46009063e-03,
        -5.22455620e-03,
        1.48913004e-02,
        -3.62276956e-02,
        8.02305192e-02,
        -1.80019379e-01,
        6.25149012e-01,
    ]),
    SymFir([
        7.62032287e-04,
        -7.64759816e-03,
        3.85545008e-02,
        -1.39896080e-01,
        6.08227193e-01,
    ]),
    SymFir([
        -2.65761488e-03,
        2.49805823e-02,
        -1.21497065e-01,
        5.99174082e-01,
    ]),
    SymFir([1.18773514e-02, -9.81294960e-02, 5.86252153e-01]),
);

/// Passband width in units of lowest sample rate
pub const HBF_PASSBAND: f32 = 0.4;

/// Max low-rate block size (HbfIntCascade input, HbfDecCascade output)
pub const HBF_CASCADE_BLOCK: usize = 1 << 6;

/// Half-band decimation filter cascade with optimal taps
///
/// See [HBF_TAPS].
/// Only in-place processing is implemented.
/// Supports rate changes of 1, 2, 4, 8, and 16.
#[derive(Copy, Clone, Debug)]
pub struct HbfDecCascade {
    buf: (
        [f32; HBF_CASCADE_BLOCK * 2],
        [f32; HBF_CASCADE_BLOCK * 4],
        [f32; HBF_CASCADE_BLOCK * 8],
    ),
    stages: (
        HbfDec<[f32; 2 * HBF_TAPS.0.0.len() - 1 + HBF_CASCADE_BLOCK]>,
        HbfDec<[f32; 2 * HBF_TAPS.1.0.len() - 1 + HBF_CASCADE_BLOCK * 2]>,
        HbfDec<[f32; 2 * HBF_TAPS.2.0.len() - 1 + HBF_CASCADE_BLOCK * 4]>,
        HbfDec<[f32; 2 * HBF_TAPS.3.0.len() - 1 + HBF_CASCADE_BLOCK * 8]>,
    ),
}

impl Default for HbfDecCascade {
    fn default() -> Self {
        Self {
            buf: (
                [Default::default(); _],
                [Default::default(); _],
                [Default::default(); _],
            ),
            stages: Default::default(),
        }
    }
}

impl<const R: usize> SplitProcess<[f32; R], f32, HbfDecCascade> for HbfTaps {
    fn block(&self, state: &mut HbfDecCascade, x: &[[f32; R]], y: &mut [f32]) {
        debug_assert_eq!(x.len(), y.len());
        debug_assert!([0, 1, 2, 3, 4].map(|i| 1 << i).contains(&R));
        for (x, y) in x
            .chunks(HBF_CASCADE_BLOCK)
            .zip(y.chunks_mut(HBF_CASCADE_BLOCK))
        {
            let mut x = x.as_flattened();
            if R > 1 << 3 {
                let u = &mut state.buf.2[..x.len() / 2];
                self.3.block(&mut state.stages.3, x.as_chunks().0, u);
                x = u;
            }
            if R > 1 << 2 {
                let u = &mut state.buf.1[..x.len() / 2];
                self.2.block(&mut state.stages.2, x.as_chunks().0, u);
                x = u;
            }
            if R > 1 << 1 {
                let u = &mut state.buf.0[..x.len() / 2];
                self.1.block(&mut state.stages.1, x.as_chunks().0, u);
                x = u;
            }
            if R > 1 << 0 {
                self.0.block(&mut state.stages.0, x.as_chunks().0, y);
            } else {
                y.copy_from_slice(x);
            }
        }
    }

    fn process(&self, state: &mut HbfDecCascade, x: [f32; R]) -> f32 {
        let mut y = 0.0;
        self.block(state, from_ref(&x), from_mut(&mut y));
        y
    }
}

impl HbfDecCascade {
    /// Response length, effective number of taps
    pub const fn len(rate: usize) -> usize {
        let mut n = 0;
        if rate > 8 {
            n = n / 2 + 2 * HBF_TAPS.3.0.len() - 1
        }
        if rate > 4 {
            n = n / 2 + 2 * HBF_TAPS.2.0.len() - 1;
        }
        if rate > 2 {
            n = n / 2 + 2 * HBF_TAPS.1.0.len() - 1;
        }
        if rate > 1 {
            n = n / 2 + 2 * HBF_TAPS.0.0.len() - 1;
        }
        n
    }
}

/// Half-band interpolation filter cascade with optimal taps.
///
/// This is a no_alloc version without trait objects.
/// The price to pay is fixed and flat memory usage independent
/// of block size and cascade length.
///
/// See [HBF_TAPS].
/// Only in-place processing is implemented.
/// Supports rate changes of 1, 2, 4, 8, and 16.
#[derive(Copy, Clone, Debug)]
pub struct HbfIntCascade {
    buf: (
        [[f32; 2]; HBF_CASCADE_BLOCK],
        [[f32; 2]; HBF_CASCADE_BLOCK * 2],
        [[f32; 2]; HBF_CASCADE_BLOCK * 4],
    ),
    stages: (
        HbfInt<[f32; 2 * HBF_TAPS.0.0.len() - 1 + HBF_CASCADE_BLOCK]>,
        HbfInt<[f32; 2 * HBF_TAPS.1.0.len() - 1 + HBF_CASCADE_BLOCK * 2]>,
        HbfInt<[f32; 2 * HBF_TAPS.2.0.len() - 1 + HBF_CASCADE_BLOCK * 4]>,
        HbfInt<[f32; 2 * HBF_TAPS.3.0.len() - 1 + HBF_CASCADE_BLOCK * 8]>,
    ),
}

impl Default for HbfIntCascade {
    fn default() -> Self {
        Self {
            buf: (
                [Default::default(); _],
                [Default::default(); _],
                [Default::default(); _],
            ),
            stages: Default::default(),
        }
    }
}

impl<const R: usize> SplitProcess<f32, [f32; R], HbfIntCascade> for HbfTaps {
    fn block(&self, state: &mut HbfIntCascade, x: &[f32], y: &mut [[f32; R]]) {
        debug_assert_eq!(x.len(), y.len());
        debug_assert!([0, 1, 2, 3, 4].map(|i| 1 << i).contains(&R));
        for (mut x, y) in x
            .chunks(HBF_CASCADE_BLOCK)
            .zip(y.chunks_mut(HBF_CASCADE_BLOCK))
        {
            let y = y.as_flattened_mut().as_chunks_mut().0;
            let mut u;
            if R == 1 << 0 {
                y.as_flattened_mut().copy_from_slice(x);
            }
            if R == 1 << 1 {
                self.0.block(&mut state.stages.0, x, y);
            } else if R > 1 << 1 {
                u = &mut state.buf.0[..x.len()];
                self.0.block(&mut state.stages.0, x, u);
                x = u.as_flattened();
            }
            if R == 1 << 2 {
                self.1.block(&mut state.stages.1, x, y);
            } else if R > 1 << 2 {
                u = &mut state.buf.1[..x.len()];
                self.1.block(&mut state.stages.1, x, u);
                x = u.as_flattened();
            }
            if R == 1 << 3 {
                self.2.block(&mut state.stages.2, x, y);
            } else if R > 1 << 3 {
                u = &mut state.buf.2[..x.len()];
                self.2.block(&mut state.stages.2, x, u);
                x = u.as_flattened();
            }
            if R == 1 << 4 {
                self.3.block(&mut state.stages.3, x, y);
            }
        }
    }

    fn process(&self, state: &mut HbfIntCascade, x: f32) -> [f32; R] {
        let mut y = [0.0; _];
        self.block(state, from_ref(&x), from_mut(&mut y));
        y
    }
}

impl HbfIntCascade {
    /// Response length, effective number of taps
    pub const fn len(rate: usize) -> usize {
        let mut n = 0;
        if rate > 1 {
            n = 2 * (n + 2 * HBF_TAPS.0.0.len() - 1);
        }
        if rate > 2 {
            n = 2 * (n + 2 * HBF_TAPS.1.0.len() - 1);
        }
        if rate > 4 {
            n = 2 * (n + 2 * HBF_TAPS.2.0.len() - 1);
        }
        if rate > 8 {
            n = 2 * (n + 2 * HBF_TAPS.3.0.len() - 1);
        }
        n
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use dsp_process::{Process, Split};
    use rustfft::{FftPlanner, num_complex::Complex};

    #[test]
    fn test() {
        let mut h = Split::new(SymFir([0.5]), HbfDec::<[_; 5]>::default());
        h.as_mut().block(&[], &mut []);

        let x = [1.0; 8];
        let mut y = [0.0; 4];
        h.as_mut().block(x.as_chunks().0, &mut y);
        assert_eq!(y, [1.5, 2.0, 2.0, 2.0]);

        let mut h = Split::new(&HBF_TAPS.3, HbfDec::<[_; 11]>::default());
        let x: Vec<_> = (0..8).map(|i| i as f32).collect();
        h.as_mut().block(x.as_chunks().0, &mut y);
        println!("{:?}", y);
    }

    #[test]
    fn decim() {
        let mut h = HbfDecCascade::default();
        const R: usize = 1 << 4;
        let mut y = vec![0.0; 2];
        let x: Vec<_> = (0..y.len() * R).map(|i| i as f32).collect();
        HBF_TAPS.block(&mut h, x.as_chunks::<R>().0, &mut y);
        println!("{:?}", y);
    }

    #[test]
    fn response_length_dec() {
        let mut h = HbfDecCascade::default();
        const R: usize = 1 << 4;
        let mut y = [0.0; 100];
        let x: Vec<f32> = (0..R * y.len()).map(|_| rand::random()).collect();
        HBF_TAPS.block(&mut h, x.as_chunks::<R>().0, &mut y);
        let x = vec![0.0; 1 << 10];
        HBF_TAPS.block(&mut h, x.as_chunks::<R>().0, &mut y[..x.len() / R]);
        let n = HbfDecCascade::len(R);
        assert!(y[n - 1] != 0.0);
        assert_eq!(y[n], 0.0);
    }

    #[test]
    fn interp() {
        let mut h = HbfIntCascade::default();
        const R: usize = 1 << 4;
        let r = HbfIntCascade::len(R);
        let mut x = vec![0.0; r / R + 1];
        x[0] = 1.0;
        let mut y = vec![0.0; x.len() * R];
        HBF_TAPS.block(&mut h, &x, y.as_chunks_mut::<R>().0);
        println!("{:?}", y); // interpolator impulse response
        assert!(y[r] != 0.0);
        assert_eq!(y[r + 1..], vec![0.0; y.len() - r - 1]);

        let mut y: Vec<_> = y
            .iter()
            .map(|&x| Complex {
                re: x as f64 / R as f64,
                im: 0.0,
            })
            .collect();
        // pad
        y.resize(5 << 10, Default::default());
        FftPlanner::new().plan_fft_forward(y.len()).process(&mut y);
        // transfer function
        let p: Vec<_> = y.iter().map(|y| 10.0 * y.norm_sqr().log10()).collect();
        let f = p.len() as f32 / R as f32;
        // pass band ripple
        let p_pass = p[..(f * HBF_PASSBAND).floor() as _]
            .iter()
            .fold(0.0, |m, p| p.abs().max(m));
        assert!(p_pass < 2e-6, "{p_pass}");
        // stop band attenuation
        let p_stop = p[(f * (1.0 - HBF_PASSBAND)).ceil() as _..p.len() / 2]
            .iter()
            .fold(-200.0, |m, p| p.max(m));
        assert!(p_stop < -140.0, "{p_stop}");
    }

    /// small 32 block size, single stage, 3 mul (11 tap) decimator
    /// 3.5 insn per input item, > 1 GS/s per core on Skylake
    #[test]
    #[ignore]
    fn insn_dec() {
        const N: usize = HBF_TAPS.4.0.len();
        assert_eq!(N, 3);
        let mut h = HbfDec::<[_; 2 * N - 1 + (1 << 4)]>::default();
        let x = [9.0; 1 << 5];
        let mut y = [0.0; 1 << 4];
        for _ in 0..1 << 25 {
            HBF_TAPS.4.block(&mut h, x.as_chunks().0, &mut y);
        }
    }

    /// 1k block size, single stage, 23 mul (91 tap) decimator
    /// 4.9 insn: > 1 GS/s
    #[test]
    #[ignore]
    fn insn_dec2() {
        const N: usize = HBF_TAPS.0.0.len();
        assert_eq!(N, 23);
        const M: usize = 1 << 10;
        let mut h = HbfDec::<[_; 2 * N - 1 + M]>::default();
        let x = [9.0; M];
        let mut y = [0.0; M / 2];
        for _ in 0..1 << 20 {
            HBF_TAPS.0.block(&mut h, x.as_chunks().0, &mut y);
        }
    }

    /// full block size full decimator cascade (depth 4, 1024 items per input block)
    /// 4.1 insn: > 1 GS/s
    #[test]
    #[ignore]
    fn insn_casc() {
        let mut h = HbfDecCascade::default();
        const R: usize = 1 << 4;
        let x = [9.0; R << 6];
        let mut y = [0.0; 1 << 6];
        for _ in 0..1 << 20 {
            HBF_TAPS.block(&mut h, x.as_chunks::<R>().0, &mut y);
        }
    }

    // // sdr crate, setup like insn_dec2()
    // // 187 insn
    // #[test]
    // #[ignore]
    // fn insn_sdr() {
    //     use sdr::fir;
    //     const N: usize = HBF_TAPS.0.len();
    //     const M: usize = 1 << 10;
    //     let mut taps = [0.0f64; { 4 * N - 1 }];
    //     let (old, new) = taps.split_at_mut(2 * N - 1);
    //     for (tap, (old, new)) in HBF_TAPS.0.iter().zip(
    //         old.iter_mut()
    //             .step_by(2)
    //             .zip(new.iter_mut().rev().step_by(2)),
    //     ) {
    //         *old = (*tap * 0.5).into();
    //         *new = *old;
    //     }
    //     taps[2 * N - 1] = 0.5;
    //     let mut h = fir::FIR::new(&taps, 2, 1);
    //     let x = [9.0; M];
    //     // let mut h1 = HbfDec::<N, { 2 * N - 1 + M }>::new(&HBF_TAPS.0);
    //     // let mut y1 = [0.0; M / 2];
    //     for _ in 0..1 << 16 {
    //         let _y = h.process(&x);
    //         // h1.process_block(Some(&x), &mut y1);
    //         // assert_eq!(y1.len(), y.len());
    //         // assert!(y1.iter().zip(y.iter()).all(|(y1, y)| (y1 - y).abs() < 1e-6));
    //     }
    // }

    // // // futuredsp crate, setup like insn_dec2()
    // // // 315 insn
    // #[test]
    // #[ignore]
    // fn insn_futuredsp() {
    //     use futuredsp::{fir::PolyphaseResamplingFirKernel, UnaryKernel};
    //     const N: usize = HBF_TAPS.0.len();
    //     const M: usize = 1 << 10;
    //     let mut taps = [0.0f32; { 4 * N - 1 }];
    //     let (old, new) = taps.split_at_mut(2 * N - 1);
    //     for (tap, (old, new)) in HBF_TAPS.0.iter().zip(
    //         old.iter_mut()
    //             .step_by(2)
    //             .zip(new.iter_mut().rev().step_by(2)),
    //     ) {
    //         *old = *tap * 0.5;
    //         *new = *old;
    //     }
    //     taps[2 * N - 1] = 0.5;
    //     let x = [9.0f32; M];
    //     let mut y = [0.0f32; M];
    //     let fir = PolyphaseResamplingFirKernel::<_, _, _, _>::new(1, 2, taps);
    //     for _ in 0..1 << 14 {
    //         fir.work(&x, &mut y);
    //     }
    // }
}
