use super::{Complex, ComplexExt, MulScaled};
use dsp_process::SplitProcess;

/// Lockin filter
///
/// Combines two [`SplitProcess`] filters and an NCO to perform demodulation
#[derive(Copy, Clone, Default)]
pub struct Lockin<C> {
    lp: C,
}

impl<C: SplitProcess<i32, i32, S>, S> SplitProcess<(i32, Complex<i32>), Complex<i32>, [S; 2]>
    for Lockin<C>
{
    /// Update the lockin with a sample taken at a local oscillator IQ value.
    fn process(&self, state: &mut [S; 2], x: (i32, Complex<i32>)) -> Complex<i32> {
        let mix = x.1.mul_scaled(x.0);
        Complex::new(
            self.lp.process(&mut state[0], mix.re()),
            self.lp.process(&mut state[1], mix.im()),
        )
    }
}

impl<C: SplitProcess<i32, i32, S>, S> SplitProcess<(i32, i32), Complex<i32>, [S; 2]> for Lockin<C> {
    /// Update the lockin with a sample taken at a given phase.
    fn process(&self, state: &mut [S; 2], x: (i32, i32)) -> Complex<i32> {
        // Get the LO signal for demodulation and mix the sample;
        self.process(state, (x.0, Complex::from_angle(x.1)))
    }
}
