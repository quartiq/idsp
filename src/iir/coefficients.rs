use num_traits::{AsPrimitive, Float, FloatConst};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
enum Shape<T> {
    InverseQ(T),
    Bandwidth(T),
    Slope(T),
}

impl<T: Float + FloatConst> Default for Shape<T> {
    fn default() -> Self {
        Self::InverseQ(T::SQRT_2())
    }
}

/// Standard audio biquad filter builder
///
/// <https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html>
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Filter<T> {
    /// Angular critical frequency (in units of sampling frequency)
    /// Corner frequency, or 3dB cutoff frequency,
    w0: T,
    /// Passband gain
    gain: T,
    /// Shelf gain (only for peaking, lowshelf, highshelf)
    /// Relative to passband gain
    shelf: T,
    /// Inverse Q
    shape: Shape<T>,
}

impl<T: Float + FloatConst> Default for Filter<T> {
    fn default() -> Self {
        Self {
            w0: T::zero(),
            gain: T::one(),
            shape: Shape::default(),
            shelf: T::one(),
        }
    }
}

impl<T> Filter<T>
where
    T: 'static + Float + FloatConst,
    f32: AsPrimitive<T>,
{
    pub fn frequency(self, critical_frequency: T, sample_frequency: T) -> Self {
        self.critical_frequency(critical_frequency / sample_frequency)
    }

    pub fn critical_frequency(self, critical_frequency: T) -> Self {
        self.angular_critical_frequency(T::TAU() * critical_frequency)
    }

    pub fn angular_critical_frequency(mut self, w0: T) -> Self {
        self.w0 = w0;
        self
    }

    pub fn gain(mut self, k: T) -> Self {
        self.gain = k;
        self
    }

    pub fn gain_db(self, k_db: T) -> Self {
        self.gain(10.0.as_().powf(k_db / 20.0.as_()))
    }

    pub fn shelf(mut self, a: T) -> Self {
        self.shelf = a;
        self
    }

    pub fn shelf_db(self, k_db: T) -> Self {
        self.gain(10.0.as_().powf(k_db / 20.0.as_()))
    }

    pub fn inverse_q(mut self, qi: T) -> Self {
        self.shape = Shape::InverseQ(qi);
        self
    }

    pub fn q(self, q: T) -> Self {
        self.inverse_q(T::one() / q)
    }

    /// Set [`FilterBuilder::frequency()`] first.
    /// In octaves.
    pub fn bandwidth(mut self, bw: T) -> Self {
        self.shape = Shape::Bandwidth(bw);
        self
    }

    /// Set [`FilterBuilder::gain()`] first.
    pub fn shelf_slope(mut self, s: T) -> Self {
        self.shape = Shape::Slope(s);
        self
    }

    fn qi(&self) -> T {
        match self.shape {
            Shape::InverseQ(qi) => qi,
            Shape::Bandwidth(bw) => {
                2.0.as_() * (T::LN_2() / 2.0.as_() * bw * self.w0 / self.w0.sin()).sinh()
            }
            Shape::Slope(s) => {
                ((self.gain + T::one() / self.gain) * (T::one() / s - T::one()) + 2.0.as_()).sqrt()
            }
        }
    }

    fn alpha(&self) -> T {
        0.5.as_() * self.w0.sin() * self.qi()
    }

    /// Lowpass biquad filter.
    ///
    /// ```
    /// use idsp::iir::*;
    /// let ba = Filter::default().critical_frequency(0.1).lowpass();
    /// println!("{ba:?}");
    /// ```
    pub fn lowpass(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        let b = self.gain * 0.5.as_() * (T::one() - fcos);
        [
            b,
            b + b,
            b,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn highpass(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        let b = self.gain * 0.5.as_() * (T::one() + fcos);
        [
            b,
            -(b + b),
            b,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn bandpass(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        let b = self.gain * alpha;
        [
            b,
            T::zero(),
            b,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn notch(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        [
            self.gain,
            -(fcos + fcos) * self.gain,
            self.gain,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn allpass(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        [
            (T::one() - alpha) * self.gain,
            -(fcos + fcos) * self.gain,
            (T::one() + alpha) * self.gain,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn peaking(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        [
            (T::one() + alpha * self.shelf) * self.gain,
            -(fcos + fcos) * self.gain,
            (T::one() - alpha * self.shelf) * self.gain,
            T::one() + alpha / self.shelf,
            -(fcos + fcos),
            T::one() - alpha / self.shelf,
        ]
    }

    pub fn lowshelf(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let sp1 = self.shelf + T::one();
        let sm1 = self.shelf - T::one();
        let tsa = 2.0.as_() * self.shelf.sqrt() * self.alpha();
        [
            self.shelf * self.gain * (sp1 - sm1 * fcos + tsa),
            2.0.as_() * self.shelf * self.gain * (sm1 - sp1 * fcos),
            self.shelf * self.gain * (sp1 - sm1 * fcos - tsa),
            sp1 + sm1 * fcos + tsa,
            (-2.0).as_() * (sm1 + sp1 * fcos),
            sp1 + sm1 * fcos - tsa,
        ]
    }

    pub fn highshelf(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let sp1 = self.shelf + T::one();
        let sm1 = self.shelf - T::one();
        let tsa = 2.0.as_() * self.shelf.sqrt() * self.alpha();
        [
            self.shelf * self.gain * (sp1 + sm1 * fcos + tsa),
            (-2.0).as_() * self.shelf * self.gain * (sm1 + sp1 * fcos),
            self.shelf * self.gain * (sp1 + sm1 * fcos - tsa),
            sp1 - sm1 * fcos + tsa,
            2.0.as_() * (sm1 - sp1 * fcos),
            sp1 - sm1 * fcos - tsa,
        ]
    }

    // TODO
    // PI-notch
    //
    // SOS cascades:
    // butterworth
    // elliptic
    // chebychev1/2
    // bessel
}

#[cfg(test)]
mod test {
    use super::*;

    use core::f64;
    use num_complex::Complex64;

    use crate::iir::*;

    #[test]
    fn lowpass_gen() {
        let ba = Biquad::<i32>::from(
            Filter::default()
                .critical_frequency(2e-9f64)
                .gain(2e7)
                .lowpass(),
        );
        println!("{:?}", ba);
    }

    fn polyval(p: &[f64], x: Complex64) -> Complex64 {
        p.iter()
            .fold(
                (Complex64::default(), Complex64::new(1.0, 0.0)),
                |(a, xi), pi| (a + xi * *pi, xi * x),
            )
            .0
    }

    fn freqz(b: &[f64], a: &[f64], f: f64) -> Complex64 {
        let z = Complex64::new(0.0, -f64::consts::TAU * f).exp();
        polyval(b, z) / polyval(a, z)
    }

    #[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
    enum Tol {
        GainDb(f64, f64),
        GainBelowDb(f64),
        GainAboveDb(f64),
    }
    impl Tol {
        fn check(&self, h: Complex64) -> bool {
            let g = 10.0 * h.norm_sqr().log10();
            match self {
                Self::GainDb(want, tol) => (g - want).abs() <= *tol,
                Self::GainAboveDb(want) => g >= *want,
                Self::GainBelowDb(want) => g <= *want,
            }
        }
    }

    fn check(f: f64, g: Tol, ba: &[f64; 6]) {
        let h = freqz(&ba[..3], &ba[3..], f);
        let hp = h.to_polar();
        assert!(
            g.check(h),
            "freq {f}: response {h}={hp:?} does not meet {g:?}"
        );
    }

    #[test]
    fn lowpass() {
        let ba = Filter::default()
            .critical_frequency(0.01)
            .gain_db(20.0)
            .lowpass();
        println!("{ba:?}");

        let bai = Biquad::<i32>::from(ba).into();
        println!("{bai:?}");

        for (f, g) in [
            (1e-3, Tol::GainDb(20.0, 0.01)),
            (0.01, Tol::GainDb(17.0, 0.1)),
            (4e-1, Tol::GainBelowDb(-40.0)),
        ] {
            check(f, g, &ba);
            check(f, g, &bai);
        }
    }

    #[test]
    fn highpass() {
        let ba = Filter::default()
            .critical_frequency(0.1)
            .gain_db(-2.0)
            .highpass();
        println!("{ba:?}");

        let bai = Biquad::<i32>::from(ba).into();
        println!("{bai:?}");

        for (f, g) in [
            (1e-3, Tol::GainBelowDb(-40.0)),
            (0.1, Tol::GainDb(-5.0, 0.1)),
            (4e-1, Tol::GainDb(-2.0, 0.01)),
        ] {
            check(f, g, &ba);
            check(f, g, &bai);
        }
    }
}
