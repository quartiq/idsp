/// Filter input items into output items.
pub trait Filter {
    /// Input/output item type.
    // TODO: impl with generic item type
    type Item;

    /// Process a block of items.
    ///
    /// Input items can be either in `x` or in `y`.
    /// In the latter case the filtering operation is done in-place.
    /// Output is always written into `y`.
    /// The slice of items written into `y` is returned.
    /// Input and output size relations must match the filter requirements
    /// (decimation/interpolation and maximum block size).
    /// When using in-place operation, `y` needs to contain the input items
    /// (fewer than `y.len()` in the case of interpolation) and must be able to
    /// contain the output items.
    fn process_block<'a>(
        &mut self,
        x: Option<&[Self::Item]>,
        y: &'a mut [Self::Item],
    ) -> &'a mut [Self::Item];

    /// Return the block size granularity and the maximum block size.
    ///
    /// For in-place processing, this refers to constraints on `y`.
    /// Otherwise this refers to the larger of `x` and `y` (`x` for decimation and `y` for interpolation).
    /// The granularity is also the rate change in the case of interpolation/decimation filters.
    fn block_size(&self) -> (usize, usize);

    /// Finite impulse response length in numer of output items minus one
    /// Get this many to drain all previous memory
    fn response_length(&self) -> usize;

    // TODO: process items with automatic blocks
    // fn process(&mut self, x: Option<&[Self::Item]>, y: &mut [Self::Item]) -> usize {}
}

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
/// The half-band filter has unique properties that make it preferrable in many cases:
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
/// They unroll and optimmize extremely well targetting current architectures,
/// e.g. requiring less than 4 instructions per input item for the full `HbfDecCascade` on Skylake.
/// The filters are optimized for decent block sizes and perform best (i.e. with negligible
/// overhead) for blocks of 32 high-rate items or more, depending very much on architecture.

#[derive(Clone, Debug, Copy)]
pub struct SymFir<'a, const M: usize, const N: usize> {
    x: [f32; N],
    taps: &'a [f32; M],
}

impl<'a, const M: usize, const N: usize> SymFir<'a, M, N> {
    /// Create a new `SymFir`.
    ///
    /// # Args
    /// * `taps`: one-sided FIR coefficients, expluding center tap, oldest to one-before-center
    pub fn new(taps: &'a [f32; M]) -> Self {
        debug_assert!(N >= M * 2);
        Self { x: [0.0; N], taps }
    }

    /// Obtain a mutable reference to the input items buffer space.
    #[inline]
    pub fn buf_mut(&mut self) -> &mut [f32] {
        &mut self.x[2 * M - 1..]
    }

    /// Perform the FIR convolution and yield results iteratively.
    #[inline]
    pub fn get(&self) -> impl Iterator<Item = f32> + '_ {
        self.x.windows(2 * M).map(|x| {
            let (old, new) = x.split_at(M);
            old.iter()
                .zip(new.iter().rev())
                .zip(self.taps.iter())
                .map(|((xo, xn), tap)| (xo + xn) * tap)
                .sum()
        })
    }

    /// Move items as new filter state.
    ///
    /// # Args
    /// * `offset`: Keep the `2*M-1` items at `offset` as the new filter state.
    #[inline]
    pub fn keep_state(&mut self, offset: usize) {
        self.x.copy_within(offset..offset + 2 * M - 1, 0);
    }
}

// TODO: pub struct SymFirInt<R>, SymFirDec<R>

/// Half band decimator (decimate by two)
///
/// The effective number of DSP taps is 4*M - 1.
///
/// M: number of taps
/// N: state size: N = 2*M - 1 + output.len()
#[derive(Clone, Debug, Copy)]
pub struct HbfDec<'a, const M: usize, const N: usize> {
    even: [f32; N], // This is an upper bound to N - M (unstable const expr)
    odd: SymFir<'a, M, N>,
}

impl<'a, const M: usize, const N: usize> HbfDec<'a, M, N> {
    /// Create a new `HbfDec`.
    ///
    /// # Args
    /// * `taps`: The FIR filter coefficients. Only the non-zero (odd) taps
    ///   from oldest to one-before-center. Normalized such that center tap is 1.
    pub fn new(taps: &'a [f32; M]) -> Self {
        Self {
            even: [0.0; N],
            odd: SymFir::new(taps),
        }
    }
}

impl<'a, const M: usize, const N: usize> Filter for HbfDec<'a, M, N> {
    type Item = f32;

    #[inline]
    fn block_size(&self) -> (usize, usize) {
        (2, 2 * (N - (2 * M - 1)))
    }

    #[inline]
    fn response_length(&self) -> usize {
        2 * M - 1
    }

    fn process_block<'b>(
        &mut self,
        x: Option<&[Self::Item]>,
        y: &'b mut [Self::Item],
    ) -> &'b mut [Self::Item] {
        let x = x.unwrap_or(y);
        debug_assert_eq!(x.len() & 1, 0);
        let k = x.len() / 2;
        // load input
        for (xi, (even, odd)) in x.chunks_exact(2).zip(
            self.even[M - 1..][..k]
                .iter_mut()
                .zip(self.odd.buf_mut()[..k].iter_mut()),
        ) {
            *even = xi[0];
            *odd = xi[1];
        }
        // compute output
        for (yi, (even, odd)) in y[..k]
            .iter_mut()
            .zip(self.even[..k].iter().zip(self.odd.get()))
        {
            *yi = 0.5 * (even + odd);
        }
        // keep state
        self.even.copy_within(k..k + M - 1, 0);
        self.odd.keep_state(k);
        &mut y[..k]
    }
}

/// Half band interpolator (interpolation rate 2)
///
/// The effective number of DSP taps is 4*M - 1.
///
/// M: number of taps
/// N: state size: N = 2*M - 1 + input.len()
#[derive(Clone, Debug, Copy)]
pub struct HbfInt<'a, const M: usize, const N: usize> {
    fir: SymFir<'a, M, N>,
}

impl<'a, const M: usize, const N: usize> HbfInt<'a, M, N> {
    /// Non-zero (odd) taps from oldest to one-before-center.
    /// Normalized such that center tap is 1.
    pub fn new(taps: &'a [f32; M]) -> Self {
        Self {
            fir: SymFir::new(taps),
        }
    }

    /// Obtain a mutable reference to the input items buffer space
    pub fn buf_mut(&mut self) -> &mut [f32] {
        self.fir.buf_mut()
    }
}

impl<'a, const M: usize, const N: usize> Filter for HbfInt<'a, M, N> {
    type Item = f32;

    #[inline]
    fn block_size(&self) -> (usize, usize) {
        (2, 2 * (N - (2 * M - 1)))
    }

    #[inline]
    fn response_length(&self) -> usize {
        4 * M - 2
    }

    fn process_block<'b>(
        &mut self,
        x: Option<&[Self::Item]>,
        y: &'b mut [Self::Item],
    ) -> &'b mut [Self::Item] {
        debug_assert_eq!(y.len() & 1, 0);
        let k = y.len() / 2;
        let x = x.unwrap_or(&y[..k]);
        // load input
        self.fir.buf_mut()[..k].copy_from_slice(x);
        // compute output
        for (yi, (even, &odd)) in y
            .chunks_exact_mut(2)
            .zip(self.fir.get().zip(self.fir.x[M..][..k].iter()))
        {
            // Choose the even item to be the interpolated one.
            // The alternative would have the same response length
            // but larger latency.
            yi[0] = even; // interpolated
            yi[1] = odd; // center tap: identity
        }
        // keep state
        self.fir.keep_state(k);
        y
    }
}

/// Standard/optimal half-band filter cascade taps
///
/// * more than 98 dB stop band attenuation
/// * 0.4 pass band (relative to lowest sample rate)
/// * less than 0.001 dB ripple
/// * linear phase/flat group delay
/// * rate change up to 2**5 = 32
/// * lowest rate filter is at 0 index
/// * use taps 0..n for 2**n interpolation/decimation
#[allow(clippy::excessive_precision, clippy::type_complexity)]
pub const HBF_TAPS: ([f32; 15], [f32; 6], [f32; 3], [f32; 3], [f32; 2]) = (
    // 15 coefficients (effective number of DSP taps 4*15-1 = 59), transition band width .2 fs
    [
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
    ],
    // 6, .47
    [
        -0.00086943,
        0.00577837,
        -0.02201674,
        0.06357869,
        -0.16627679,
        0.61979312,
    ],
    // 3, .754
    [0.01414651, -0.10439639, 0.59026742],
    // 3, .877
    [0.01227974, -0.09930782, 0.58702834],
    // 2, .94
    [-0.06291796, 0.5629161],
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
    depth: usize,
    stages: (
        HbfDec<'static, { HBF_TAPS.0.len() }, { 2 * HBF_TAPS.0.len() - 1 + HBF_CASCADE_BLOCK }>,
        HbfDec<'static, { HBF_TAPS.1.len() }, { 2 * HBF_TAPS.1.len() - 1 + HBF_CASCADE_BLOCK * 2 }>,
        HbfDec<'static, { HBF_TAPS.2.len() }, { 2 * HBF_TAPS.2.len() - 1 + HBF_CASCADE_BLOCK * 4 }>,
        HbfDec<'static, { HBF_TAPS.3.len() }, { 2 * HBF_TAPS.3.len() - 1 + HBF_CASCADE_BLOCK * 8 }>,
    ),
}

impl Default for HbfDecCascade {
    fn default() -> Self {
        Self {
            depth: 0,
            stages: (
                HbfDec::new(&HBF_TAPS.0),
                HbfDec::new(&HBF_TAPS.1),
                HbfDec::new(&HBF_TAPS.2),
                HbfDec::new(&HBF_TAPS.3),
            ),
        }
    }
}

impl HbfDecCascade {
    #[inline]
    pub fn set_depth(&mut self, n: usize) {
        assert!(n <= 4);
        self.depth = n;
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl Filter for HbfDecCascade {
    type Item = f32;

    #[inline]
    fn block_size(&self) -> (usize, usize) {
        (
            1 << self.depth,
            match self.depth {
                0 => usize::MAX,
                1 => self.stages.0.block_size().1,
                2 => self.stages.1.block_size().1,
                3 => self.stages.2.block_size().1,
                _ => self.stages.3.block_size().1,
            },
        )
    }

    #[inline]
    fn response_length(&self) -> usize {
        let mut n = 0;
        if self.depth > 3 {
            n = n / 2 + self.stages.3.response_length();
        }
        if self.depth > 2 {
            n = n / 2 + self.stages.2.response_length();
        }
        if self.depth > 1 {
            n = n / 2 + self.stages.1.response_length();
        }
        if self.depth > 0 {
            n = n / 2 + self.stages.0.response_length();
        }
        n
    }

    fn process_block<'a>(
        &mut self,
        x: Option<&[Self::Item]>,
        mut y: &'a mut [Self::Item],
    ) -> &'a mut [Self::Item] {
        if x.is_some() {
            unimplemented!(); // TODO: pair of intermediate buffers
        }
        let n = y.len();

        if self.depth > 3 {
            y = self.stages.3.process_block(None, y);
        }
        if self.depth > 2 {
            y = self.stages.2.process_block(None, y);
        }
        if self.depth > 1 {
            y = self.stages.1.process_block(None, y);
        }
        if self.depth > 0 {
            y = self.stages.0.process_block(None, y);
        }
        debug_assert_eq!(y.len(), n >> self.depth);
        y
    }
}

/// Half-band interpolation filter cascade with optimal taps.
///
/// This is a no_alloc version without trait objects.
/// The price to pay is fixed and maximal memory usage independent
/// of block size and cascade length.
///
/// See [HBF_TAPS].
/// Only in-place processing is implemented.
/// Supports rate changes of 1, 2, 4, 8, and 16.
#[derive(Copy, Clone, Debug)]
pub struct HbfIntCascade {
    depth: usize,
    pub stages: (
        HbfInt<'static, { HBF_TAPS.0.len() }, { 2 * HBF_TAPS.0.len() - 1 + HBF_CASCADE_BLOCK }>,
        HbfInt<'static, { HBF_TAPS.1.len() }, { 2 * HBF_TAPS.1.len() - 1 + HBF_CASCADE_BLOCK * 2 }>,
        HbfInt<'static, { HBF_TAPS.2.len() }, { 2 * HBF_TAPS.2.len() - 1 + HBF_CASCADE_BLOCK * 4 }>,
        HbfInt<'static, { HBF_TAPS.3.len() }, { 2 * HBF_TAPS.3.len() - 1 + HBF_CASCADE_BLOCK * 8 }>,
    ),
}

impl Default for HbfIntCascade {
    fn default() -> Self {
        Self {
            depth: 4,
            stages: (
                HbfInt::new(&HBF_TAPS.0),
                HbfInt::new(&HBF_TAPS.1),
                HbfInt::new(&HBF_TAPS.2),
                HbfInt::new(&HBF_TAPS.3),
            ),
        }
    }
}

impl HbfIntCascade {
    pub fn set_depth(&mut self, n: usize) {
        assert!(n <= 4);
        self.depth = n;
    }

    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl Filter for HbfIntCascade {
    type Item = f32;

    #[inline]
    fn block_size(&self) -> (usize, usize) {
        (
            1 << self.depth,
            match self.depth {
                0 => usize::MAX,
                1 => self.stages.0.block_size().1,
                2 => self.stages.1.block_size().1,
                3 => self.stages.2.block_size().1,
                _ => self.stages.3.block_size().1,
            },
        )
    }

    #[inline]
    fn response_length(&self) -> usize {
        let mut n = 0;
        if self.depth > 0 {
            n = 2 * n + self.stages.0.response_length();
        }
        if self.depth > 1 {
            n = 2 * n + self.stages.1.response_length();
        }
        if self.depth > 2 {
            n = 2 * n + self.stages.2.response_length();
        }
        if self.depth > 3 {
            n = 2 * n + self.stages.3.response_length();
        }
        n
    }

    fn process_block<'a>(
        &mut self,
        x: Option<&[Self::Item]>,
        y: &'a mut [Self::Item],
    ) -> &'a mut [Self::Item] {
        if x.is_some() {
            unimplemented!(); // TODO: one intermediate buffer and `y`
        }
        // TODO: use buf_mut() and write directly into next filters' input buffer

        let mut n = y.len() >> self.depth;
        if self.depth > 0 {
            n = self.stages.0.process_block(None, &mut y[..2 * n]).len();
        }
        if self.depth > 1 {
            n = self.stages.1.process_block(None, &mut y[..2 * n]).len();
        }
        if self.depth > 2 {
            n = self.stages.2.process_block(None, &mut y[..2 * n]).len();
        }
        if self.depth > 3 {
            n = self.stages.3.process_block(None, &mut y[..2 * n]).len();
        }
        debug_assert_eq!(n, y.len());
        &mut y[..n]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rustfft::{num_complex::Complex, FftPlanner};

    #[test]
    fn test() {
        let mut h = HbfDec::<1, 5>::new(&[0.5]);
        assert_eq!(h.process_block(None, &mut []), &[]);

        let mut x = [1.0; 8];
        assert_eq!((2, x.len()), h.block_size());
        let x = h.process_block(None, &mut x);
        assert_eq!(x, [0.75, 1.0, 1.0, 1.0]);

        let mut h = HbfDec::<3, 9>::new(&HBF_TAPS.3);
        let mut x: Vec<_> = (0..8).map(|i| i as f32).collect();
        assert_eq!((2, x.len()), h.block_size());
        let x = h.process_block(None, &mut x);
        println!("{:?}", x);
    }

    #[test]
    fn decim() {
        let mut h = HbfDecCascade::default();
        h.set_depth(4);
        assert_eq!(
            h.block_size(),
            (1 << h.depth(), HBF_CASCADE_BLOCK << h.depth())
        );
        let mut x: Vec<_> = (0..2 << h.depth()).map(|i| i as f32).collect();
        let x = h.process_block(None, &mut x);
        println!("{:?}", x);
    }

    #[test]
    fn interp() {
        let mut h = HbfIntCascade::default();
        h.set_depth(4);
        assert_eq!(
            h.block_size(),
            (1 << h.depth(), HBF_CASCADE_BLOCK << h.depth())
        );
        let k = h.block_size().0;
        let r = h.response_length();
        let mut x = vec![0.0; (r + 1 + k - 1) / k * k];
        x[0] = 1.0;
        let x = h.process_block(None, &mut x);
        println!("{:?}", x); // interpolator impulse response
        assert!(x[r] != 0.0);
        assert_eq!(x[r + 1..], vec![0.0; x.len() - r - 1]);

        let g = (1 << h.depth()) as f32;
        let mut y = Vec::from_iter(x.iter().map(|&x| Complex { re: x / g, im: 0.0 }));
        // pad
        y.resize(5 << 10, Complex::default());
        FftPlanner::new().plan_fft_forward(y.len()).process(&mut y);
        // transfer function
        let p = Vec::from_iter(y.iter().map(|y| 10.0 * y.norm_sqr().log10()));
        let f = p.len() as f32 / g;
        // pass band ripple
        let p_pass = p[..(f * HBF_PASSBAND).floor() as _]
            .iter()
            .fold(0.0, |m, p| p.abs().max(m));
        assert!(p_pass < 0.00035);
        // stop band attenuation
        let p_stop = p[(f * (1.0 - HBF_PASSBAND)).ceil() as _..p.len() / 2]
            .iter()
            .fold(-200.0, |m, p| p.max(m));
        assert!(p_stop < -98.4);
    }

    /// small 32 batch size, single stage, 3 mul (11 tap) decimator
    /// 3.5 insn per input sample, > 1 GS/s on Skylake
    #[test]
    #[ignore]
    fn insn_dec() {
        const N: usize = HBF_TAPS.3.len();
        let mut h = HbfDec::<N, { 2 * N - 1 + (1 << 4) }>::new(&HBF_TAPS.3);
        let mut x = [9.0; 1 << 5];
        for _ in 0..1 << 26 {
            h.process_block(None, &mut x);
        }
    }

    /// 1k batch size, single stage, 15 mul (59 tap) decimator
    /// 5 insn per input sample, > 1 GS/s on Skylake
    #[test]
    #[ignore]
    fn insn_dec2() {
        const N: usize = HBF_TAPS.0.len();
        assert_eq!(N, 15);
        const M: usize = 1 << 10;
        let mut h = HbfDec::<N, { 2 * N - 1 + M }>::new(&HBF_TAPS.0);
        let mut x = [9.0; M];
        for _ in 0..1 << 20 {
            h.process_block(None, &mut x);
        }
    }

    /// large batch size full decimator cascade (depth 4, 1024 sampled per batch)
    /// 4.1 insns per input sample, > 1 GS/s per core
    #[test]
    #[ignore]
    fn insn_casc() {
        let mut x = [9.0; 1 << 10];
        let mut h = HbfDecCascade::default();
        h.set_depth(4);
        for _ in 0..1 << 20 {
            h.process_block(None, &mut x);
        }
    }
}
