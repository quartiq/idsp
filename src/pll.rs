use core::num::Wrapping as W;
use dsp_fixedpoint::{Q, Q32, W32};
use dsp_process::{Process, SplitProcess};
use num_traits::{AsPrimitive, Float};

use crate::{Accu, ClampWrap, iir::DirectForm};

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
#[derive(Copy, Debug, Clone, Default)]
pub struct PLL {
    // last input phase
    x: W<i32>,
    // last output phase
    y0: W<i32>,
    // last output frequency
    f0: W<i32>,
    // filtered frequency
    f: Q<W<i64>, W<i32>, 32>,
    // filtered output phase
    y: Q<W<i64>, W<i32>, 32>,
}

impl SplitProcess<Option<W<i32>>, Accu<W<i32>>, PLL> for W32<32> {
    /// Update the PLL with a new phase sample. This needs to be called (sampled) periodically.
    /// The signal's phase/frequency is reconstructed relative to the sampling period.
    ///
    /// Args:
    /// * `x`: New input phase sample or None if a sample has been missed.
    ///
    /// Returns:
    /// A tuple of instantaneous phase and frequency estimates.
    fn process(&self, state: &mut PLL, x: Option<W<i32>>) -> Accu<W<i32>> {
        if let Some(x) = x {
            let dx = x - state.x;
            state.x = x;
            let df = *self * (dx - state.f.quantize());
            state.f += df;
            state.y += state.f;
            state.f += df;
            let dy = *self * (x - state.y.quantize());
            state.y += dy;
            let y = state.y.quantize();
            state.y += dy;
            state.f0 = y - state.y0;
            state.y0 = y;
        } else {
            state.y += state.f;
            state.x += state.f0;
            state.y0 += state.f0;
        }
        Accu::new(state.y0, state.f0)
    }
}

impl PLL {
    /// Return the current phase estimate
    pub fn phase(&self) -> W<i32> {
        self.y0
    }

    /// Return the current frequency estimate
    pub fn frequency(&self) -> W<i32> {
        self.f0
    }
}

/// A PLL
///
/// Type 2, order 3
///
/// The phase detector is symmetric (additive): the loop filter should have negative gain.
/// The output will compensate the input phase: it will settle to the complement.
/// The output phase increment (the loop filter output) is the negative of the input increment.
#[derive(Debug, Clone, Default)]
pub struct Pll {
    /// Lead lag coefficients
    ///
    /// `f0 += b0*y0 + b1*y1 + a1*f1`
    pub ba: [Q32<32>; 3],
}

impl Pll {
    /// Return Pll from pole offsets
    pub fn from_zpk(zero: f32, pole: f32, gain: f32) -> Self {
        Self {
            ba: [gain, -gain * zero, -(1.0 - pole)].map(Q32::from_f32),
        }
    }
}

/// PLL state
#[derive(Debug, Clone, Default)]
pub struct PllState {
    /// Input phase difference clamp
    pub clamp: ClampWrap<W<i32>>,
    /// Loop filter state: after clamp
    pub z0: i32,
    /// After nyquist zero
    pub y0: i32,
    /// After lead-lag
    pub f0: i64,
    /// After DC pole
    pub f1: W<i64>,
    /// Current output phase
    pub y: W<i32>,
}

impl PllState {
    /// Return the current phase estimate
    pub fn phase(&self) -> W<i32> {
        self.y
    }

    /// Return the current frequency estimate
    pub fn frequency(&self) -> W<i32> {
        W((self.f1.0 >> 32) as _)
    }
}

impl SplitProcess<W<i32>, W<i32>, PllState> for Pll {
    fn process(&self, state: &mut PllState, x: W<i32>) -> W<i32> {
        // phase error
        let z0 = state.clamp.process(x + state.y).0 >> 1;
        // nyquist zero
        let y0 = z0 + state.z0;
        state.z0 = z0;
        // lead lag
        state.f0 +=
            (self.ba[0] * y0 + self.ba[1] * state.y0 + self.ba[2] * (state.f0 >> 32) as i32).inner
                + ((self.ba[2].inner as i64 * state.f0 as u32 as i64) >> 32);
        state.y0 = y0;
        // DC pole
        state.f1 += W(state.f0);
        // advance output phase, oscillator DC pole
        let y = state.y;
        state.y += W((state.f1.0 >> 32) as i32);
        y
    }
}

#[cfg(test)]
mod tests {
    use crate::iir::{Biquad, DirectForm1Dither, Pair};
    use dsp_fixedpoint::Q32;

    use super::*;
    use core::num::Wrapping as W;
    use dsp_process::{Process, Split};

    #[test]
    fn mini() {
        let mut p = Split::new(W32::new(W(1 << 24)), PLL::default());
        let a = p.process(Some(W(0x10000)));
        assert_eq!(a.state.0, 0x1ff);
        assert_eq!(a.step.0, 0x1ff);
    }

    #[test]
    fn converge() {
        let mut p = PLL::default();
        let k = W32::new(W(1 << 24));
        let f0 = W(0x71f63049);
        let n = 1 << 14;
        let mut x = W(0i32);
        for i in 0..n {
            x += f0;
            let a = k.process(&mut p, Some(x));
            if i > n / 4 {
                assert_eq!((a.step - f0).0.abs() <= 1, true);
            }
            if i > n / 2 {
                assert_eq!((a.state - x).0.abs() <= 1, true);
            }
        }
    }

    #[test]
    fn converge_pll() {
        let p = Pll::from_zpk(1.0 - 4e-2, 1.0 - 4e-1, -8e-2);
        println!("{p:?}");
        let mut s = PllState::default();
        let a = Accu::<W<i32>>::new(W(0x0), W(0x71f63049));
        let n = 1 << 10;
        for (i, x) in a.take(n).enumerate() {
            let y = p.process(&mut s, x);
            println!("x: {x:#010x} y+x: {:#010x}", y + x);
            if i > n / 2 {
                assert!((a.step + s.frequency()).0.abs() <= 1);
                assert!((x + y).0.abs() <= 4);
            }
        }
    }

    #[test]
    fn converge_narrow() {
        let p = Pll::from_zpk(1.0 - 1e-4, 1.0 - 1.5e-3, -2e-6);
        println!("{p:?}");
        let mut s = PllState::default();
        let a = Accu::<W<i32>>::new(W(0x0), W(0x140_1235));
        let n = 1 << 19;
        for (i, x) in a.take(n).enumerate() {
            let y = p.process(&mut s, x);
            println!("x: {x:#010x} y+x: {:#010x}", y + x);
            if i > n / 2 {
                assert!((a.step + s.frequency()).0.abs() <= 1 << 16);
                assert!((x + y).0.abs() <= 1 << 16);
            }
        }
    }
}
