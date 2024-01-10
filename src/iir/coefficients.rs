use num_traits::{AsPrimitive, Float, FloatConst};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
enum Shape<T> {
    /// Inverse W
    InverseQ(T),
    /// Relative bandwidth in octaves
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
        self.shelf(10.0.as_().powf(k_db / 20.0.as_()))
    }

    pub fn inverse_q(mut self, qi: T) -> Self {
        self.shape = Shape::InverseQ(qi);
        self
    }

    pub fn q(self, q: T) -> Self {
        self.inverse_q(T::one() / q)
    }

    /// Relative bandwidth in octaves.
    pub fn bandwidth(mut self, bw: T) -> Self {
        self.shape = Shape::Bandwidth(bw);
        self
    }

    /// Shelf slope.
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

    fn fcos_alpha(&self) -> (T, T) {
        let (fsin, fcos) = self.w0.sin_cos();
        (fcos, 0.5.as_() * fsin * self.qi())
    }

    /// Low pass filter
    ///
    /// Builds second order biquad low pass filter coefficients.
    ///
    /// ```
    /// use idsp::iir::*;
    /// let ba = Filter::default()
    ///     .critical_frequency(0.1)
    ///     .gain(1000.0)
    ///     .lowpass();
    /// let iir = Biquad::<i32>::from(&ba);
    /// let mut xy = [0; 4];
    /// let x = vec![3, -4, 5, 7, -3, 2];
    /// let y: Vec<_> = x.iter().map(|x0| iir.update(&mut xy, *x0)).collect();
    /// assert_eq!(y, [5, 3, 9, 25, 42, 49]);
    /// ```
    pub fn lowpass(self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
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

    /// High pass filter
    ///
    /// Builds second order biquad high pass filter coefficients.
    ///
    /// ```
    /// use idsp::iir::*;
    /// let ba = Filter::default()
    ///     .critical_frequency(0.1)
    ///     .gain(1000.0)
    ///     .highpass();
    /// let iir = Biquad::<i32>::from(&ba);
    /// let mut xy = [0; 4];
    /// let x = vec![3, -4, 5, 7, -3, 2];
    /// let y: Vec<_> = x.iter().map(|x0| iir.update(&mut xy, *x0)).collect();
    /// assert_eq!(y, [5, -9, 11, 12, -1, 17]);
    /// ```
    pub fn highpass(self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
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

    /// Band pass
    ///
    /// ```
    /// use idsp::iir::*;
    /// let ba = Filter::default()
    ///     .frequency(1000.0, 48e3)
    ///     .q(5.0)
    ///     .gain_db(3.0)
    ///     .bandpass();
    /// println!("{ba:?}");
    /// ```
    pub fn bandpass(self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let b = self.gain * alpha;
        [
            b,
            T::zero(),
            -b,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn notch(self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
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
        let (fcos, alpha) = self.fcos_alpha();
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
        let (fcos, alpha) = self.fcos_alpha();
        [
            (T::one() + alpha * self.shelf) * self.gain,
            -(fcos + fcos) * self.gain,
            (T::one() - alpha * self.shelf) * self.gain,
            T::one() + alpha / self.shelf,
            -(fcos + fcos),
            T::one() - alpha / self.shelf,
        ]
    }

    /// Low shelf
    ///
    /// ```
    /// use idsp::iir::*;
    /// let ba = Filter::default()
    ///     .frequency(1000.0, 48e3)
    ///     .shelf_slope(2.0)
    ///     .shelf_db(20.0)
    ///     .lowshelf();
    /// println!("{ba:?}");
    /// ```
    pub fn lowshelf(self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let tsa = 2.0.as_() * self.shelf.sqrt() * alpha;
        let sp1 = self.shelf + T::one();
        let sm1 = self.shelf - T::one();
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
        let (fcos, alpha) = self.fcos_alpha();
        let tsa = 2.0.as_() * self.shelf.sqrt() * alpha;
        let sp1 = self.shelf + T::one();
        let sm1 = self.shelf - T::one();
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
}

// TODO
// SOS cascades:
// butterworth
// elliptic
// chebychev1/2
// bessel

#[cfg(test)]
mod test {
    use super::*;

    use core::f64;
    use num_complex::Complex64;

    use crate::iir::*;

    #[test]
    fn lowpass_gen() {
        let ba = Biquad::<i32>::from(
            &Filter::default()
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
    }
    impl Tol {
        fn check(&self, h: Complex64) -> bool {
            let g = 10.0 * h.norm_sqr().log10();
            match self {
                Self::GainDb(want, tol) => (g - want).abs() <= *tol,
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

    fn check_coeffs(ba: &[f64; 6], fg: &[(f64, Tol)]) {
        println!("{ba:?}");

        for (f, g) in fg {
            check(*f, *g, ba);
        }

        // Quantize
        let bai = (&Biquad::<i32>::from(ba)).into();
        println!("{bai:?}");

        for (f, g) in fg {
            check(*f, *g, &bai);
        }
    }

    #[test]
    fn lowpass() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.01)
                .gain_db(20.0)
                .lowpass(),
            &[
                (1e-3, Tol::GainDb(20.0, 0.01)),
                (0.01, Tol::GainDb(17.0, 0.1)),
                (4e-1, Tol::GainBelowDb(-40.0)),
            ],
        );
    }

    #[test]
    fn highpass() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.1)
                .gain_db(-2.0)
                .highpass(),
            &[
                (1e-3, Tol::GainBelowDb(-40.0)),
                (0.1, Tol::GainDb(-5.0, 0.1)),
                (4e-1, Tol::GainDb(-2.0, 0.01)),
            ],
        );
    }

    #[test]
    fn notch() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.02)
                .bandwidth(2.0)
                .notch(),
            &[
                (1e-4, Tol::GainDb(0.0, 0.01)),
                (0.01, Tol::GainDb(-3.0, 0.1)),
                (0.02, Tol::GainBelowDb(-80.0)),
                (0.04, Tol::GainDb(-3.0, 0.1)),
                (4e-1, Tol::GainDb(0.0, 0.01)),
            ],
        );
    }
}
