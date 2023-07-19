use super::{Chain, Complex, ComplexExt, Filter, Lowpass, MulScaled};

#[derive(Copy, Clone, Default)]
pub struct Lockin<const N: usize, const O: usize> {
    state: [Chain<N, Lowpass<O>>; 2],
}

impl<const N: usize, const O: usize> Lockin<N, O> {
    /// Update the lockin with a sample taken at a local oscillator IQ value.
    pub fn update_iq(
        &mut self,
        sample: i32,
        lo: Complex<i32>,
        k: &<Lowpass<O> as Filter>::Config,
    ) -> Complex<i32> {
        let mix = lo.mul_scaled(sample);

        // Filter with the IIR lowpass,
        // return IQ (in-phase and quadrature) data.
        Complex {
            re: self.state[0].update(mix.re, k),
            im: self.state[1].update(mix.im, k),
        }
    }

    /// Update the lockin with a sample taken at a given phase.
    pub fn update(
        &mut self,
        sample: i32,
        phase: i32,
        k: &<Lowpass<O> as Filter>::Config,
    ) -> Complex<i32> {
        // Get the LO signal for demodulation and mix the sample;
        self.update_iq(sample, Complex::from_angle(phase), k)
    }
}
