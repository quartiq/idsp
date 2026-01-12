use super::Complex;
use core::num::Wrapping;
use dsp_fixedpoint::Q32;
use dsp_process::SplitProcess;

/// Lockin filter
///
/// Combines two [`SplitProcess`] filters and an NCO to perform demodulation
#[derive(Copy, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct Lockin<C>(
    // Lowpass configuration
    pub C,
);

/// Input and LO
impl<C: SplitProcess<i32, i32, S>, S> SplitProcess<(i32, Complex<Q32<31>>), Complex<i32>, [S; 2]>
    for Lockin<C>
{
    /// Update the lockin with a sample taken at a local oscillator IQ value.
    fn process(&self, state: &mut [S; 2], x: (i32, Complex<Q32<31>>)) -> Complex<i32> {
        Complex::new(
            self.0.process(&mut state[0], x.0 * x.1.re()),
            self.0.process(&mut state[1], x.0 * x.1.im()),
        )
    }
}

/// Sample and phase
impl<C: SplitProcess<i32, i32, S>, S> SplitProcess<(i32, Wrapping<i32>), Complex<i32>, [S; 2]>
    for Lockin<C>
{
    /// Update the lockin with a sample taken at a given phase.
    fn process(&self, state: &mut [S; 2], x: (i32, Wrapping<i32>)) -> Complex<i32> {
        // Get the LO signal for demodulation and mix the sample;
        self.process(state, (x.0, Complex::from_angle(x.1)))
    }
}
