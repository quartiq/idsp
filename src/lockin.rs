use super::{Complex, ComplexExt, Lowpass, Lowpass1, MulScaled};

#[derive(Copy, Clone, Default)]
pub struct Lockin<const N: usize> {
    state: [Lowpass<N>; 2],
}

impl<const N: usize> Lockin<N> {
    /// Update the lockin with a sample taken at a local oscillator IQ value.
    pub fn update_iq(&mut self, sample: i32, lo: Complex<i32>, k: u32) -> Complex<i32> {
        let mix = lo.mul_scaled(sample);

        // Filter with the IIR lowpass,
        // return IQ (in-phase and quadrature) data.
        Complex {
            re: self.state[0].update(mix.re, k),
            im: self.state[1].update(mix.im, k),
        }
    }

    /// Update the lockin with a sample taken at a given phase.
    pub fn update(&mut self, sample: i32, phase: i32, k: u32) -> Complex<i32> {
        // Get the LO signal for demodulation and mix the sample;
        self.update_iq(sample, Complex::from_angle(phase), k)
    }
}

#[derive(Copy, Clone)]
pub struct Lockin1<const N: usize> {
    state: [[Lowpass1; N]; 2],
}

impl<const N: usize> Default for Lockin1<N> {
    fn default() -> Self {
        Self {
            state: [[Default::default(); N]; 2],
        }
    }
}

impl<const N: usize> Lockin1<N> {
    /// Update the lockin with a sample taken at a local oscillator IQ value.
    pub fn update_iq(&mut self, sample: i32, lo: Complex<i32>, k: u32) -> Complex<i32> {
        let mix = lo.mul_scaled(sample);

        // Filter with the IIR lowpass,
        // return IQ (in-phase and quadrature) data.
        let mut x = mix.re;
        for l in self.state[0].iter_mut() {
            x = (l.update(x, k) >> 32) as _;
        }
        let re = x;
        let mut x = mix.im;
        for l in self.state[1].iter_mut() {
            x = (l.update(x, k) >> 32) as _;
        }
        let im = x;
        Complex { re, im }
    }

    /// Update the lockin with a sample taken at a given phase.
    pub fn update(&mut self, sample: i32, phase: i32, k: u32) -> Complex<i32> {
        // Get the LO signal for demodulation and mix the sample;
        self.update_iq(sample, Complex::from_angle(phase), k)
    }
}
