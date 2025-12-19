use serde::{Deserialize, Serialize};

use dsp_process::SplitProcess;

/// Type-II, sampled phase, discrete time PLL
///
/// This PLL tracks the frequency and phase of an input signal with respect to the sampling clock.
/// The open loop transfer function is I^2,I from input phase to output phase and P,I from input
/// phase to output frequency.
///
/// The transfer functions (for phase and frequency) contain an additional zero at Nyquist.
///
/// The PLL locks to any frequency (i.e. it locks to the alias in the first Nyquist zone) and is
/// stable for any gain (1 <= shift <= 30). It has a single parameter that determines the loop
/// bandwidth in octave steps. The gain can be changed freely between updates.
///
/// The frequency and phase settling time constants for a frequency/phase jump are `1 << shift`
/// update cycles. The loop bandwidth is `1/(2*pi*(1 << shift))` in units of the sample rate.
/// While the phase is being settled after settling the frequency, there is a typically very
/// small frequency overshoot.
///
/// All math is naturally wrapping 32 bit integer. Phase and frequency are understood modulo that
/// overflow in the first Nyquist zone. Expressing the IIR equations in other ways (e.g. single
/// (T)-DF-{I,II} biquad/IIR) would break on overflow (i.e. every cycle).
///
/// There are no floating point rounding errors here. But there is integer quantization/truncation
/// error of the `shift` lowest bits leading to a phase offset for very low gains. Truncation
/// bias is applied. Rounding is "half up". The phase truncation error can be removed very
/// efficiently by dithering.
///
/// This PLL does not unwrap phase slips accumulated during (frequency) lock acquisition.
/// This can and should be implemented elsewhere by unwrapping and scaling the input phase
/// and un-scaling and wrapping output phase and frequency. This then affects dynamic range,
/// gain, and noise accordingly.
///
/// The extension to I^3,I^2,I behavior to track chirps phase-accurately or to i64 data to
/// increase resolution for extremely narrowband applications is obvious.
///
/// This PLL implements first order noise shaping to reduce quantization errors.
#[derive(Copy, Debug, Clone, Default, Deserialize, Serialize)]
pub struct PLL {
    // last input phase
    x: i32,
    // last output phase
    y0: i32,
    // last output frequency
    f0: i32,
    // filtered frequency
    f: i64,
    // filtered output phase
    y: i64,
}

impl SplitProcess<Option<i32>, (), PLL> for i32 {
    /// Update the PLL with a new phase sample. This needs to be called (sampled) periodically.
    /// The signal's phase/frequency is reconstructed relative to the sampling period.
    ///
    /// Args:
    /// * `x`: New input phase sample or None if a sample has been missed.
    /// * `k`: Feedback gain.
    ///
    /// Returns:
    /// A tuple of instantaneous phase and frequency estimates.
    fn process(&self, state: &mut PLL, x: Option<i32>) {
        if let Some(x) = x {
            let dx = x.wrapping_sub(state.x);
            state.x = x;
            let df = dx.wrapping_sub((state.f >> 32) as i32) as i64 * *self as i64;
            state.f = state.f.wrapping_add(df);
            state.y = state.y.wrapping_add(state.f);
            state.f = state.f.wrapping_add(df);
            let dy = x.wrapping_sub((state.y >> 32) as i32) as i64 * *self as i64;
            state.y = state.y.wrapping_add(dy);
            let y = (state.y >> 32) as i32;
            state.y = state.y.wrapping_add(dy);
            state.f0 = y.wrapping_sub(state.y0);
            state.y0 = y;
        } else {
            state.y = state.y.wrapping_add(state.f);
            state.x = state.x.wrapping_add(state.f0);
            state.y0 = state.y0.wrapping_add(state.f0);
        }
    }
}

impl PLL {
    /// Return the current phase estimate
    pub fn phase(&self) -> i32 {
        self.y0
    }

    /// Return the current frequency estimate
    pub fn frequency(&self) -> i32 {
        self.f0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_process::{Process, Split};
    #[test]
    fn mini() {
        let mut p = Split::new(1 << 24, PLL::default());
        p.as_mut().process(Some(0x10000));
        assert_eq!(p.state.phase(), 0x1ff);
        assert_eq!(p.state.frequency(), 0x1ff);
    }

    #[test]
    fn converge() {
        let mut p = PLL::default();
        let k = 1 << 24;
        let f0 = 0x71f63049_i32;
        let n = 1 << 14;
        let mut x = 0i32;
        for i in 0..n {
            x = x.wrapping_add(f0);
            Split::new(&k, &mut p).process(Some(x));
            if i > n / 4 {
                assert_eq!(p.frequency().wrapping_sub(f0).abs() <= 1, true);
            }
            if i > n / 2 {
                assert_eq!(p.phase().wrapping_sub(x).abs() <= 1, true);
            }
        }
    }
}
