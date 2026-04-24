use super::Complex;
use core::{num::Wrapping, ops::Mul};
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
impl<X: Copy + Mul<U, Output = X>, U: Copy, C: SplitProcess<X, X, S>, S>
    SplitProcess<(X, Complex<U>), Complex<X>, [S; 2]> for Lockin<C>
{
    /// Update the lockin with a sample taken at a local oscillator IQ value.
    fn process(&self, state: &mut [S; 2], x: (X, Complex<U>)) -> Complex<X> {
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
        let lo = Complex::<Q32<32>>::from_bits(Complex::from_angle(x.1));
        self.process(state, (x.0, lo))
    }
}
