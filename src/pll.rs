use core::num::Wrapping as W;
use dsp_fixedpoint::Q32;
use dsp_process::{Process, SplitProcess};

use crate::ClampWrap;

/// Type-2, order-3 sampled phase, discrete time PLL
///
/// This PLL tracks the frequency and phase of an input signal with respect to the sampling clock.
/// The open loop transfer function is type 2 (DC double integrator) from input phase to output phase.
///
/// The transfer functions (for phase and frequency) contain an additional zero at Nyquist.
///
/// The PLL locks to any frequency (i.e. it locks to the alias in the first Nyquist zone) and is
/// stable for any numerically valid gain (in units of the sample rate: 7e-5 to 5e-2).
/// It has a single parameter that determines the loop bandwidth.
/// The gains can be changed freely between updates.
///
/// All math is naturally wrapping 32 bit integer. Phase and frequency are understood modulo that
/// overflow in the first Nyquist zone. Expressing the IIR equations in other ways (e.g. single
/// (T)-DF-{I,II} biquad/IIR) would break on overflow (i.e. every cycle).
///
/// There are no floating point rounding errors. The integer quantization/truncation
/// error is fed back (first order noise shaping).
///
/// This PLL clamps phase wraps accumulated during (frequency) lock acquisition.
///
/// The phase detector is symmetric (additive): the loop filter should have negative gain.
/// The output will compensate the input phase: it will settle to the complement.
/// The output phase increment (the loop filter output) is the negative of the input increment.
#[derive(Debug, Clone, Default)]
pub struct PLL {
    /// Lead lag coefficients
    ///
    /// `f0 += b0*y0 + b1*y1 + a1*f1`
    pub ba: [Q32<32>; 3],
}

impl PLL {
    /// Return Pll from zeros/pole/gain
    pub fn from_zpk(zero: f32, pole: f32, gain: f32) -> Self {
        Self {
            ba: [gain, -gain * zero, -(1.0 - pole)].map(Q32::from_f32),
        }
    }

    /// Given a crossover create a PLL
    ///
    /// About 2 dB peaking, 55 deg phase margin
    pub fn from_bandwidth(mut bw: f32) -> Self {
        bw *= 2.0;
        const PZ: f32 = 10.0;
        Self::from_zpk(
            1.0 - bw,
            1.0 - PZ * bw,
            (-PZ * core::f32::consts::PI) * (bw * bw),
        )
    }
}

/// PLL state
#[derive(Debug, Clone, Default)]
pub struct PLLState {
    /// Input phase difference clamp
    pub clamp: ClampWrap<W<i32>>,
    /// Loop filter state: after clamp
    pub z0: i32,
    /// After nyquist zero
    pub y0: i32,
    /// After lead-lag
    pub f0: i64,
    /// After DC pole
    pub f: W<i64>,
    /// Current output phase
    pub y: W<i32>,
}

impl PLLState {
    /// Return the current phase estimate
    pub fn phase(&self) -> W<i32> {
        self.y
    }

    /// Return the current frequency estimate
    pub fn frequency(&self) -> W<i32> {
        W((self.f.0 >> 32) as _)
    }
}

impl SplitProcess<W<i32>, W<i32>, PLLState> for PLL {
    fn process(&self, state: &mut PLLState, x: W<i32>) -> W<i32> {
        // phase error
        let z0 = state.clamp.process(x + state.y).0 >> 1;
        // nyquist zero
        let y0 = z0 + state.z0;
        state.z0 = z0;
        // lead lag, wide state
        state.f0 +=
            (self.ba[0] * y0 + self.ba[1] * state.y0 + self.ba[2] * (state.f0 >> 32) as i32).inner
                + ((self.ba[2].inner as i64 * state.f0 as u32 as i64) >> 32);
        state.y0 = y0;
        // DC pole, frequency, wide state
        state.f += W(state.f0);
        // advance output phase, oscillator DC pole
        let y = state.y;
        state.y += W((state.f.0 >> 32) as i32);
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
    fn converge_pll() {
        let p = PLL::from_bandwidth(5e-2);
        println!("{p:?}");
        let mut s = PLLState::default();
        let a = Accu::<W<i32>>::new(W(0x0), W(0x71f63049));
        let n = 1 << 9;
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
        let p = PLL::from_bandwidth(7e-5);
        println!("{p:?}");
        let mut s = PLLState::default();
        let a = Accu::<W<i32>>::new(W(0x0), W(0x140_1235));
        let n = 1 << 18;
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
