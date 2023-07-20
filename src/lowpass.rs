use crate::Filter;

/// Arbitrary order, high dynamic range, wide coefficient range,
/// lowpass filter implementation. DC gain is 1.
///
/// Type argument N is the filter order. N must be `1` or `2`.
///
/// The filter will cleanly saturate towards the `i32` range.
///
/// The filter configuration `Config` contains its gains.
///
/// For the first-order lowpass this the corner frequency in scaled Q31:
/// `k = pi*(1 << 31)*f0/fn` where
/// `f0` is the 3dB corner frequency and
/// `fn` is the Nyquist frequency.
/// The corner frequency is warped in the usual way.
///
/// For the second-order lowpass this is `[k**2/(1 << 32), -k/q]` with `q = 1/sqrt(2)`
/// for a Butterworth response.
/// In addition to the poles at the corner frequency The filters have zeros at Nyquist.
///
/// The first-order lowpass works fine and accurate for any positive gain
/// `1 <= k <= (1 << 31) - 1`.
/// The second-order lowpass works and is accurate for
/// `1 << 16 <= k <= q*(1 << 31)`.
///
/// Both filters have been optimized for accuracy, dynamic range, and
/// speed on Cortex-M7.
#[derive(Copy, Clone)]
pub struct Lowpass<const N: usize>(pub(crate) [i64; N]);
impl<const N: usize> Filter for Lowpass<N> {
    type Config = [i32; N];
    fn update(&mut self, x: i32, k: &Self::Config) -> i32 {
        let mut d = x.saturating_sub((self.0[0] >> 32) as i32) as i64 * k[0] as i64;
        let y;
        if N >= 2 {
            d += (self.0[1] >> 32) * k[1] as i64;
            self.0[1] += d;
            self.0[0] += self.0[1];
            y = self.get();
            // This creates the double Nyquist zero,
            // compensates the gain lost in the signed i32 as (i32 as i64)*(i64 >> 32)
            // multiplication while keeping the lowest bit significant, and
            // copes better with wrap-around than Nyquist averaging.
            self.0[0] += self.0[1];
            self.0[1] += d;
        } else {
            self.0[0] += d;
            y = self.get();
            self.0[0] += d;
        }
        y
    }
    fn get(&self) -> i32 {
        (self.0[0] >> 32) as i32
    }
    fn set(&mut self, x: i32) {
        self.0[0] = (x as i64) << 32;
    }
}

impl<const N: usize> Default for Lowpass<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

/// First order lowpass
pub type Lowpass1 = Lowpass<1>;
/// Second order lowpass
pub type Lowpass2 = Lowpass<2>;
