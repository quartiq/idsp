//! Half-band filters and cascades
//!
//! Used to perform very efficient high-dynamic range rate changes by powers of two.
//!
//! Symmetric and anti-symmetric FIR filter prototype.
//!
//! # Generics
//! * `M`: number of taps, one-sided. The filter has effectively 2*M DSP taps
//!
//! # Half band decimation/interpolation filters
//!
//! Half-band filters (rate change of 2) and cascades of HBFs are implemented in
//! [`HbfDec`] and [`HbfInt`] etc.
//! The half-band filter has unique properties that make it preferable in many cases:
//!
//! * only needs M multiplications (fused multiply accumulate) for 4*M taps
//! * HBF decimator stores less state than a generic FIR filter
//! * as a FIR filter has linear phase/flat group delay
//! * very small passband ripple and excellent stopband attenuation
//! * as a cascade of decimation/interpolation filters, the higher-rate filters
//!   need successively fewer taps, allowing the filtering to be dominated by
//!   only the highest rate filter with the fewest taps
//! * In a cascade of HBF the overall latency, group delay, and impulse response
//!   length are dominated by the lowest-rate filter which, due to its manageable transition
//!   band width (compared to single-stage filters) can be smaller, shorter, and faster.
//! * high dynamic range and inherent stability compared with an IIR filter
//! * can be combined with a CIC filter for non-power-of-two or even higher rate changes
//!
//! The implementations here are all `no_std` and `no-alloc`.
//! They support (but don't require) in-place filtering to reduce memory usage.
//! They unroll and optimize extremely well targetting current architectures,
//! e.g. requiring less than 4 instructions per input item for the full `HbfDecCascade` on Skylake.
//! The filters are optimized for decent block sizes and perform best (i.e. with negligible
//! overhead) for blocks of 32 high-rate items or more, depending very much on architecture.

use core::{
    iter::Sum,
    ops::{Add, Mul, Sub},
    slice::{from_mut, from_ref},
};

use dsp_process::{ChunkIn, ChunkOut, Major, SplitProcess};

/// Perform the FIR convolution and yield results iteratively.
#[inline]
fn get<C: Copy, T, const M: usize, const ODD: bool, const SYM: bool>(
    c: &[C; M],
    x: &[T],
) -> impl Iterator<Item = T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<C, Output = T> + Sum,
{
    // https://doc.rust-lang.org/std/primitive.slice.html#method.array_windows
    x.windows(2 * M + ODD as usize).map(|x| {
        let Some((old, new)) = x.first_chunk::<M>().zip(x.last_chunk::<M>()) else {
            unreachable!()
        };
        // Taps from small (large distance from center) to large (center taps)
        // to reduce FP cancellation a bit
        let xc = old
            .iter()
            .zip(new.iter().rev())
            .zip(c.iter())
            .map(|((old, new), tap)| (if SYM { *new + *old } else { *new - *old }) * *tap)
            .sum();
        if ODD { xc + x[M] } else { xc }
    })
}

macro_rules! type_fir {
    ($name:ident, $odd:literal, $sym:literal) => {
        #[doc = concat!("Linear phase FIR taps for odd = ", stringify!($odd), " and symmetric = ", stringify!($sym))]
        #[derive(Clone, Copy, Debug, Default)]
        #[repr(transparent)]
        pub struct $name<C>(pub C);
        impl<T, const M: usize> $name<[T; M]> {
            /// Response length/number of taps minus one
            pub const LEN: usize = 2 * M - 1 + $odd as usize;
        }

        impl<
            C: Copy,
            T: Copy + Default + Sub<T, Output = T> + Add<Output = T> + Mul<C, Output = T> + Sum,
            const M: usize,
            const N: usize,
        > SplitProcess<T, T, [T; N]> for $name<[C; M]>
        {
            fn block(&self, state: &mut [T; N], x: &[T], y: &mut [T]) {
                for (x, y) in x.chunks(N - Self::LEN).zip(y.chunks_mut(N - Self::LEN)) {
                    state[Self::LEN..][..x.len()].copy_from_slice(x);
                    for (y, x) in y.iter_mut().zip(get::<_, _, _, $odd, $sym>(&self.0, state)) {
                        *y = x;
                    }
                    state.copy_within(x.len()..x.len() + Self::LEN, 0);
                }
            }

            fn process(&self, state: &mut [T; N], x: T) -> T {
                let mut y = T::default();
                self.block(state, from_ref(&x), from_mut(&mut y));
                y
            }
        }
    };
}
// Type 1 taps
// Center tap is unity
type_fir!(OddSymmetric, true, true);
// Type 2 taps
type_fir!(EvenSymmetric, false, true);
// Type 3 taps
// Center tap is zero
type_fir!(OddAntiSymmetric, true, false);
// Type 4 taps
type_fir!(EvenAntiSymmetric, false, false);

/// Half band decimator (decimate by two) state
#[derive(Clone, Debug)]
pub struct HbfDec<T> {
    even: T, // at least N - M len
    odd: T,  // N > 2*M - 1 (=EvenSymmetric::LEN)
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
    C: Copy,
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<C, Output = T> + Sum,
    const M: usize,
    const N: usize,
> SplitProcess<[T; 2], T, HbfDec<[T; N]>> for EvenSymmetric<[C; M]>
{
    fn block(&self, state: &mut HbfDec<[T; N]>, x: &[[T; 2]], y: &mut [T]) {
        debug_assert_eq!(x.len(), y.len());
        const { assert!(N > Self::LEN) }
        for (x, y) in x.chunks(N - Self::LEN).zip(y.chunks_mut(N - Self::LEN)) {
            // load input
            for (x, (even, odd)) in x.iter().zip(
                state.even[M - 1..]
                    .iter_mut()
                    .zip(state.odd[Self::LEN..].iter_mut()),
            ) {
                *even = x[0];
                *odd = x[1];
            }
            // compute output
            let odd = get::<_, _, _, false, true>(&self.0, &state.odd);
            for (y, (odd, even)) in y.iter_mut().zip(odd.zip(state.even.iter().copied())) {
                *y = odd + even;
            }
            // keep state
            state.even.copy_within(x.len()..x.len() + M - 1, 0);
            state.odd.copy_within(x.len()..x.len() + Self::LEN, 0);
        }
    }

    fn process(&self, state: &mut HbfDec<[T; N]>, x: [T; 2]) -> T {
        let mut y = Default::default();
        self.block(state, from_ref(&x), from_mut(&mut y));
        y
    }
}

/// Half band interpolator (interpolation rate 2) state
#[derive(Clone, Debug)]
pub struct HbfInt<T> {
    x: T, // len N > EvenSymmetric::LEN
}

impl<T: Default + Copy, const N: usize> Default for HbfInt<[T; N]> {
    fn default() -> Self {
        Self {
            x: [T::default(); _],
        }
    }
}

impl<
    C: Copy,
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<C, Output = T> + Sum,
    const M: usize,
    const N: usize,
> SplitProcess<T, [T; 2], HbfInt<[T; N]>> for EvenSymmetric<[C; M]>
{
    fn block(&self, state: &mut HbfInt<[T; N]>, x: &[T], y: &mut [[T; 2]]) {
        debug_assert_eq!(x.len(), y.len());
        const { assert!(N > Self::LEN) }
        for (x, y) in x.chunks(N - Self::LEN).zip(y.chunks_mut(N - Self::LEN)) {
            // load input
            state.x[Self::LEN..][..x.len()].copy_from_slice(x);
            // compute output
            let odd = get::<_, _, _, false, true>(&self.0, &state.x);
            for (y, (even, odd)) in y.iter_mut().zip(odd.zip(state.x[M..].iter().copied())) {
                *y = [even, odd]; // interpolated, center tap: identity
            }
            // keep state
            state.x.copy_within(x.len()..x.len() + Self::LEN, 0);
        }
    }

    fn process(&self, state: &mut HbfInt<[T; N]>, x: T) -> [T; 2] {
        let mut y = Default::default();
        self.block(state, from_ref(&x), from_mut(&mut y));
        y
    }
}

/// Half band filter cascade taps for a 98 dB filter
type HbfTaps98 = (
    EvenSymmetric<[f32; 15]>,
    EvenSymmetric<[f32; 6]>,
    EvenSymmetric<[f32; 3]>,
    EvenSymmetric<[f32; 3]>,
    EvenSymmetric<[f32; 2]>,
);

/// Half band filter cascade taps
///
/// * obtained with `signal.remez(2*n, bands=(0, .4, .5, .5), desired=(1, 0), fs=1, grid_density=1<<10)[:n]`
/// * more than 98 dB stop band attenuation (>16 bit)
/// * 0.4 pass band (relative to lowest sample rate)
/// * less than 0.001 dB ripple
/// * linear phase/flat group delay
/// * rate change up to 2**5 = 32
/// * lowest rate filter is at 0 index
/// * use taps 0..n for 2**n interpolation/decimation
#[allow(clippy::excessive_precision)]
pub const HBF_TAPS_98: HbfTaps98 = (
    // n=15 coefficients (effective number of DSP taps 4*15-1 = 59), transition band width df=.2 fs
    EvenSymmetric([
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
    EvenSymmetric([
        -0.00086943,
        0.00577837,
        -0.02201674,
        0.06357869,
        -0.16627679,
        0.61979312,
    ]),
    // 3, .754
    EvenSymmetric([0.01414651, -0.10439639, 0.59026742]),
    // 3, .877
    EvenSymmetric([0.01227974, -0.09930782, 0.58702834]),
    // 2, .94
    EvenSymmetric([-0.06291796, 0.5629161]),
);

/// Half band filter cascade taps
type HbfTaps = (
    EvenSymmetric<[f32; 23]>,
    EvenSymmetric<[f32; 10]>,
    EvenSymmetric<[f32; 5]>,
    EvenSymmetric<[f32; 4]>,
    EvenSymmetric<[f32; 3]>,
);

/// Half band filters taps
///
/// * 140 dB stopband, 0.2 µB passband ripple, limited by f32 dynamic range
/// * otherwise like [`HBF_TAPS_98`].
#[allow(clippy::excessive_precision)]
pub const HBF_TAPS: HbfTaps = (
    EvenSymmetric([
        7.60375795e-07,
        -3.77494111e-06,
        1.26458559e-05,
        -3.43188253e-05,
        8.10687478e-05,
        -1.72971467e-04,
        3.40845059e-04,
        -6.29522864e-04,
        1.10128831e-03,
        -1.83933299e-03,
        2.95124926e-03,
        -4.57290964e-03,
        6.87374176e-03,
        -1.00656257e-02,
        1.44199840e-02,
        -2.03025100e-02,
        2.82462332e-02,
        -3.91128509e-02,
        5.44795658e-02,
        -7.77002672e-02,
        1.17523452e-01,
        -2.06185388e-01,
        6.34588695e-01,
    ]),
    EvenSymmetric([
        -1.12811343e-05,
        1.12724671e-04,
        -6.07439343e-04,
        2.31904511e-03,
        -7.00322950e-03,
        1.78225473e-02,
        -4.01209836e-02,
        8.43315989e-02,
        -1.83189521e-01,
        6.26346521e-01,
    ]),
    EvenSymmetric([0.0007686, -0.00768669, 0.0386536, -0.14002434, 0.60828885]),
    EvenSymmetric([-0.00261331, 0.02476858, -0.12112638, 0.59897111]),
    EvenSymmetric([0.01186105, -0.09808109, 0.58622005]),
);

/// Passband width in units of lowest sample rate
pub const HBF_PASSBAND: f32 = 0.4;

/// Cascade block size
///
/// Heuristically performs well.
pub const HBF_CASCADE_BLOCK: usize = 1 << 5;

/// Half-band decimation filter state
///
/// See [HBF_TAPS] and [HBF_DEC_CASCADE].
/// Supports rate changes are power of two up to 32.
pub type HbfDec2 =
    HbfDec<[f32; EvenSymmetric::<[f32; HBF_TAPS.0.0.len()]>::LEN + HBF_CASCADE_BLOCK]>;
/// HBF Decimate-by-4 cascade state
pub type HbfDec4 = (
    HbfDec<[f32; EvenSymmetric::<[f32; HBF_TAPS.1.0.len()]>::LEN + (HBF_CASCADE_BLOCK << 1)]>,
    HbfDec2,
);
/// HBF Decimate-by-8 cascade state
pub type HbfDec8 = (
    HbfDec<[f32; EvenSymmetric::<[f32; HBF_TAPS.2.0.len()]>::LEN + (HBF_CASCADE_BLOCK << 2)]>,
    HbfDec4,
);
/// HBF Decimate-by-16 cascade state
pub type HbfDec16 = (
    HbfDec<[f32; EvenSymmetric::<[f32; HBF_TAPS.3.0.len()]>::LEN + (HBF_CASCADE_BLOCK << 3)]>,
    HbfDec8,
);
/// HBF Decimate-by-32 cascade state
pub type HbfDec32 = (
    HbfDec<[f32; EvenSymmetric::<[f32; HBF_TAPS.4.0.len()]>::LEN + (HBF_CASCADE_BLOCK << 4)]>,
    HbfDec16,
);

type HbfDecCascade<const B: usize = HBF_CASCADE_BLOCK> = Major<
    (
        ChunkIn<&'static EvenSymmetric<[f32; HBF_TAPS.4.0.len()]>, 2>,
        Major<
            (
                ChunkIn<&'static EvenSymmetric<[f32; HBF_TAPS.3.0.len()]>, 2>,
                Major<
                    (
                        ChunkIn<&'static EvenSymmetric<[f32; HBF_TAPS.2.0.len()]>, 2>,
                        Major<
                            (
                                ChunkIn<&'static EvenSymmetric<[f32; HBF_TAPS.1.0.len()]>, 2>,
                                &'static EvenSymmetric<[f32; HBF_TAPS.0.0.len()]>,
                            ),
                            [[f32; 2]; B],
                        >,
                    ),
                    [[f32; 4]; B],
                >,
            ),
            [[f32; 8]; B],
        >,
    ),
    [[f32; 16]; B],
>;

/// HBF decimation cascade
pub const HBF_DEC_CASCADE: HbfDecCascade = Major::new((
    ChunkIn(&HBF_TAPS.4),
    Major::new((
        ChunkIn(&HBF_TAPS.3),
        Major::new((
            ChunkIn(&HBF_TAPS.2),
            Major::new((ChunkIn(&HBF_TAPS.1), &HBF_TAPS.0)),
        )),
    )),
));

/// Response length, effective number of taps
pub const fn hbf_dec_response_length(depth: usize) -> usize {
    assert!(depth < 5);
    let mut n = 0;
    if depth > 0 {
        n /= 2;
        n += EvenSymmetric::<[f32; HBF_TAPS.3.0.len()]>::LEN;
    }
    if depth > 1 {
        n /= 2;
        n += EvenSymmetric::<[f32; HBF_TAPS.2.0.len()]>::LEN;
    }
    if depth > 2 {
        n /= 2;
        n += EvenSymmetric::<[f32; HBF_TAPS.1.0.len()]>::LEN;
    }
    if depth > 3 {
        n /= 2;
        n += EvenSymmetric::<[f32; HBF_TAPS.0.0.len()]>::LEN;
    }
    n
}

/// Half-band interpolation filter state
///
/// See [HBF_TAPS] and [HBF_INT_CASCADE].
/// Supports rate changes are power of two up to 32.
pub type HbfInt2 =
    HbfInt<[f32; EvenSymmetric::<[f32; HBF_TAPS.0.0.len()]>::LEN + HBF_CASCADE_BLOCK]>;
/// HBF interpolate-by-4 cascade state
pub type HbfInt4 = (
    HbfInt2,
    HbfInt<[f32; EvenSymmetric::<[f32; HBF_TAPS.1.0.len()]>::LEN + (HBF_CASCADE_BLOCK << 1)]>,
);
/// HBF interpolate-by-8 cascade state
pub type HbfInt8 = (
    HbfInt4,
    HbfInt<[f32; EvenSymmetric::<[f32; HBF_TAPS.2.0.len()]>::LEN + (HBF_CASCADE_BLOCK << 2)]>,
);
/// HBF interpolate-by-16 cascade state
pub type HbfInt16 = (
    HbfInt8,
    HbfInt<[f32; EvenSymmetric::<[f32; HBF_TAPS.3.0.len()]>::LEN + (HBF_CASCADE_BLOCK << 3)]>,
);
/// HBF interpolate-by-32 cascade state
pub type HbfInt32 = (
    HbfInt16,
    HbfInt<[f32; EvenSymmetric::<[f32; HBF_TAPS.4.0.len()]>::LEN + (HBF_CASCADE_BLOCK << 4)]>,
);

type HbfIntCascade<const B: usize = HBF_CASCADE_BLOCK> = Major<
    (
        Major<
            (
                Major<
                    (
                        Major<
                            (
                                &'static EvenSymmetric<[f32; HBF_TAPS.0.0.len()]>,
                                ChunkOut<&'static EvenSymmetric<[f32; HBF_TAPS.1.0.len()]>, 2>,
                            ),
                            [[f32; 2]; B],
                        >,
                        ChunkOut<&'static EvenSymmetric<[f32; HBF_TAPS.2.0.len()]>, 2>,
                    ),
                    [[f32; 4]; B],
                >,
                ChunkOut<&'static EvenSymmetric<[f32; HBF_TAPS.3.0.len()]>, 2>,
            ),
            [[f32; 8]; B],
        >,
        ChunkOut<&'static EvenSymmetric<[f32; HBF_TAPS.4.0.len()]>, 2>,
    ),
    [[f32; 16]; B],
>;

/// HBF interpolation cascade
pub const HBF_INT_CASCADE: HbfIntCascade = Major::new((
    Major::new((
        Major::new((
            Major::new((&HBF_TAPS.0, ChunkOut(&HBF_TAPS.1))),
            ChunkOut(&HBF_TAPS.2),
        )),
        ChunkOut(&HBF_TAPS.3),
    )),
    ChunkOut(&HBF_TAPS.4),
));

/// Response length, effective number of taps
pub const fn hbf_int_response_length(depth: usize) -> usize {
    assert!(depth < 5);
    let mut n = 0;
    if depth > 0 {
        n += EvenSymmetric::<[f32; HBF_TAPS.0.0.len()]>::LEN;
        n *= 2;
    }
    if depth > 1 {
        n += EvenSymmetric::<[f32; HBF_TAPS.1.0.len()]>::LEN;
        n *= 2;
    }
    if depth > 2 {
        n += EvenSymmetric::<[f32; HBF_TAPS.2.0.len()]>::LEN;
        n *= 2;
    }
    if depth > 3 {
        n += EvenSymmetric::<[f32; HBF_TAPS.3.0.len()]>::LEN;
        n *= 2;
    }
    n
}

#[cfg(test)]
mod test {
    use super::*;
    use dsp_process::{Process, Split};
    use rustfft::{FftPlanner, num_complex::Complex};

    #[test]
    fn test() {
        let mut h = Split::new(EvenSymmetric([0.5]), HbfDec::<[_; 5]>::default());
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
        let mut h = HbfDec16::default();
        const R: usize = 1 << 4;
        let mut y = vec![0.0; 2];
        let x: Vec<_> = (0..y.len() * R).map(|i| i as f32).collect();
        HBF_DEC_CASCADE
            .inner
            .1
            .block(&mut h, x.as_chunks::<R>().0, &mut y);
        println!("{:?}", y);
    }

    #[test]
    fn response_length_dec() {
        let mut h = HbfDec16::default();
        const R: usize = 4;
        let mut y = [0.0; 100];
        let x: Vec<f32> = (0..y.len() << R).map(|_| rand::random()).collect();
        HBF_DEC_CASCADE
            .inner
            .1
            .block(&mut h, x.as_chunks::<{ 1 << R }>().0, &mut y);
        let x = vec![0.0; 1 << 10];
        HBF_DEC_CASCADE.inner.1.block(
            &mut h,
            x.as_chunks::<{ 1 << R }>().0,
            &mut y[..x.len() >> R],
        );
        let n = hbf_dec_response_length(R);
        assert!(y[n - 1] != 0.0);
        assert_eq!(y[n], 0.0);
    }

    #[test]
    fn interp() {
        let mut h = HbfInt16::default();
        const R: usize = 4;
        let r = hbf_int_response_length(R);
        let mut x = vec![0.0; (r >> R) + 1];
        x[0] = 1.0;
        let mut y = vec![[0.0; 1 << R]; x.len()];
        HBF_INT_CASCADE.inner.0.block(&mut h, &x, &mut y);
        println!("{:?}", y); // interpolator impulse response
        let y = y.as_flattened();
        assert_ne!(y[r], 0.0);
        assert_eq!(y[r + 1..], vec![0.0; y.len() - r - 1]);

        let mut y: Vec<_> = y
            .iter()
            .map(|&x| Complex {
                re: x as f64 / (1 << R) as f64,
                im: 0.0,
            })
            .collect();
        // pad
        y.resize(5 << 10, Default::default());
        FftPlanner::new().plan_fft_forward(y.len()).process(&mut y);
        // transfer function
        let p: Vec<_> = y.iter().map(|y| 10.0 * y.norm_sqr().log10()).collect();
        let f = p.len() as f32 / (1 << R) as f32;
        // pass band ripple
        let p_pass = p[..(f * HBF_PASSBAND).floor() as _]
            .iter()
            .fold(0.0, |m, p| p.abs().max(m));
        assert!(p_pass < 1e-6, "{p_pass}");
        // stop band attenuation
        let p_stop = p[(f * (1.0 - HBF_PASSBAND)).ceil() as _..p.len() / 2]
            .iter()
            .fold(-200.0, |m, p| p.max(m));
        assert!(p_stop < -141.5, "{p_stop}");
    }

    /// small 32 block size, single stage, 3 mul (11 tap) decimator
    /// 2.5 cycles per input item, > 2 GS/s per core on Skylake
    #[test]
    #[ignore]
    fn insn_dec() {
        const N: usize = HBF_TAPS.4.0.len();
        assert_eq!(N, 3);
        const M: usize = 1 << 4;
        let mut h = HbfDec::<[_; EvenSymmetric::<[f32; N]>::LEN + M]>::default();
        let mut x = [[9.0; 2]; M];
        let mut y = [0.0; M];
        for _ in 0..1 << 25 {
            HBF_TAPS.4.block(&mut h, &x, &mut y);
            x[13][1] = y[11]; // prevent the entire loop from being optimized away
        }
    }

    /// 1k block size, single stage, 23 mul (91 tap) decimator
    /// 2.6 cycles: > 1 GS/s
    #[test]
    #[ignore]
    fn insn_dec2() {
        const N: usize = HBF_TAPS.0.0.len();
        assert_eq!(N, 23);
        const M: usize = 1 << 9;
        let mut h = HbfDec::<[_; EvenSymmetric::<[f32; N]>::LEN + M]>::default();
        let mut x = [[9.0; 2]; M];
        let mut y = [0.0; M];
        for _ in 0..1 << 20 {
            HBF_TAPS.0.block(&mut h, &x, &mut y);
            x[33][1] = y[11]; // prevent the entire loop from being optimized away
        }
    }

    /// full block size full decimator cascade (depth 4, 1024 items per input block)
    /// 1.8 cycles: > 2 GS/s
    #[test]
    #[ignore]
    fn insn_casc() {
        let mut h = HbfDec16::default();
        const R: usize = 4;
        let mut x = [[9.0; 1 << R]; 1 << 6];
        let mut y = [0.0; 1 << 6];
        for _ in 0..1 << 20 {
            HBF_DEC_CASCADE.inner.1.block(&mut h, &x, &mut y);
            x[33][1] = y[11]; // prevent the entire loop from being optimized away
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
