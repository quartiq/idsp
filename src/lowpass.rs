use dsp_process::{SplitInplace, SplitProcess};

/// Arbitrary order, high dynamic range, wide coefficient range,
/// lowpass filter implementation. DC gain is 1.
///
/// Type argument N is the filter order. N must be `1` or `2`.
///
/// The filter will cleanly saturate towards the `i32` range.
///
/// Both filters have been optimized for accuracy, dynamic range, and
/// speed on Cortex-M7.
#[derive(Clone, Debug)]
pub struct Lowpass<const N: usize>(pub [i32; N]);

/// Lowpass filter state
#[derive(Clone, Debug)]
pub struct LowpassState<const N: usize>(pub [i64; N]);

impl<const N: usize> Default for LowpassState<N>
where
    [i64; N]: Default,
{
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<const N: usize> SplitProcess<i32, i32, LowpassState<N>> for Lowpass<N> {
    /// The filter configuration `Config` contains the filter gains.
    ///
    /// For the first-order lowpass this is a single element array `[k]` with
    /// the corner frequency in scaled Q31:
    /// `k = pi*(1 << 31)*f0/fn` where
    /// `f0` is the 3dB corner frequency and
    /// `fn` is the Nyquist frequency.
    /// The corner frequency is warped in the usual way.
    ///
    /// For the second-order lowpass this is `[k**2/(1 << 32), -k/q]` with `q = 1/sqrt(2)`
    /// for a Butterworth response.
    ///
    /// In addition to the poles at the corner frequency the filters have zeros at Nyquist.
    ///
    /// The first-order lowpass works fine and accurate for any positive gain
    /// `1 <= k <= (1 << 31) - 1`.
    /// The second-order lowpass works and is accurate for
    /// `1 << 16 <= k <= q*(1 << 31)`.
    fn process(&self, state: &mut LowpassState<N>, x: i32) -> i32 {
        // d = (x0 - p1)*k0
        // p0 = p1 + 2d
        // y0 = p1 + d
        //
        // d = (x0 - p1)*k0 + q1*k1
        // q0 = q1 + 2d
        // p0 = p1 + 2q1 + 2d
        // y0 = p1 + q1 + d
        let mut d = x.saturating_sub((state.0[0] >> 32) as i32) as i64 * self.0[0] as i64;
        let y;
        if N == 1 {
            state.0[0] += d;
            y = (state.0[0] >> 32) as i32;
            state.0[0] += d;
        } else if N == 2 {
            d += (state.0[1] >> 32) * self.0[1] as i64;
            state.0[1] += d;
            state.0[0] += state.0[1];
            y = (state.0[0] >> 32) as i32;
            // This creates the double Nyquist zero,
            // compensates the gain lost in the signed i32 as (i32 as i64)*(i64 >> 32)
            // multiplication while keeping the lowest bit significant, and
            // copes better with wrap-around than Nyquist averaging.
            state.0[0] += state.0[1];
            state.0[1] += d;
        } else {
            unimplemented!()
        }
        y
    }
}

impl<const N: usize> SplitInplace<i32, LowpassState<N>> for Lowpass<N> {}

impl<const N: usize> Default for Lowpass<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

/// First order lowpass
pub type Lowpass1 = Lowpass<1>;
/// Second order lowpass
pub type Lowpass2 = Lowpass<2>;
