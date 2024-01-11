use num_traits::{AsPrimitive, Float, FloatConst};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
enum Shape<T> {
    /// Inverse Q, sqrt(2) for critical
    InverseQ(T),
    /// Relative bandwidth in octaves
    Bandwidth(T),
    /// Slope steepnes, 1 for critical
    Slope(T),
}

impl<T: Float + FloatConst> Default for Shape<T> {
    fn default() -> Self {
        Self::InverseQ(T::SQRT_2())
    }
}

/// Standard audio biquad filter builder
///
/// <https://www.w3.org/TR/audio-eq-cookbook/>
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
    /// Set crititcal frequency from absolute units.
    ///
    /// # Arguments
    /// * `critical_frequency`: "Relevant" or "corner" or "center" frequency
    ///   in the same units as `sample_frequency`
    /// * `sample_frequency`: The sample frequency in the same units as `critical_frequency`.
    ///   E.g. both in SI Hertz or `rad/s`.
    pub fn frequency(&mut self, critical_frequency: T, sample_frequency: T) -> &mut Self {
        self.critical_frequency(critical_frequency / sample_frequency)
    }

    /// Set relative critical frequency
    ///
    /// # Arguments
    /// * `f0`: Relative critical frequency in units of the sample frequency.
    ///   Must be `0 <= f0 <= 0.5`.
    pub fn critical_frequency(&mut self, f0: T) -> &mut Self {
        self.angular_critical_frequency(T::TAU() * f0)
    }

    /// Set relative critical angular frequency
    ///
    /// # Arguments
    /// * `w0`: Relative critical angular frequency.
    ///   Must be `0 <= w0 <= π`. Defaults to `0.0`.
    pub fn angular_critical_frequency(&mut self, w0: T) -> &mut Self {
        self.w0 = w0;
        self
    }

    /// Set reference gain
    ///
    /// # Arguments
    /// * `k`: Linear reference gain. Defaults to `1.0`.
    pub fn gain(&mut self, k: T) -> &mut Self {
        self.gain = k;
        self
    }

    /// Set reference gain in dB
    ///
    /// # Arguments
    /// * `k_db`: Reference gain in dB. Defaults to `0.0`.
    pub fn gain_db(&mut self, k_db: T) -> &mut Self {
        self.gain(10.0.as_().powf(k_db / 20.0.as_()))
    }

    /// Set linear shelf gain
    ///
    /// Used only for `peaking`, `highshelf`, `lowshelf` filters.
    ///
    /// # Arguments
    /// * `a`: Linear shelf gain. Defaults to `0.0`.
    pub fn shelf(&mut self, a: T) -> &mut Self {
        self.shelf = a;
        self
    }

    /// Set shelf gain in dB
    ///
    /// Used only for `peaking`, `highshelf`, `lowshelf` filters.
    ///
    /// # Arguments
    /// * `a_db`: Linear shelf gain. Defaults to `0.0`.
    pub fn shelf_db(&mut self, a_db: T) -> &mut Self {
        self.shelf(10.0.as_().powf(a_db / 40.0.as_()))
    }

    /// Set inverse Q parameter of the filter
    ///
    /// The inverse "steepness"/"narrowness" of the filter transition.
    /// Defaults `sqrt(2)` which is as steep as possible without overshoot.
    ///
    /// # Arguments
    /// * `qi`: Inverse Q parameter.
    pub fn inverse_q(&mut self, qi: T) -> &mut Self {
        self.shape = Shape::InverseQ(qi);
        self
    }

    /// Set Q parameter of the filter
    ///
    /// The "steepness"/"narrowness" of the filter transition.
    /// Defaults `1/sqrt(2)` which is as steep as possible without overshoot.
    ///
    /// This affects the same parameter as `bandwidth()` and `shelf_slope()`.
    /// Use only one of them.
    ///
    /// # Arguments
    /// * `q`: Q parameter.
    pub fn q(&mut self, q: T) -> &mut Self {
        self.inverse_q(T::one() / q)
    }

    /// Set the relative bandwidth
    ///
    /// This affects the same parameter as `inverse_q()` and `shelf_slope()`.
    /// Use only one of them.
    ///
    /// # Arguments
    /// * `bw`: Bandwidth in octaves
    pub fn bandwidth(&mut self, bw: T) -> &mut Self {
        self.shape = Shape::Bandwidth(bw);
        self
    }

    /// Set the shelf slope.
    ///
    /// This affects the same parameter as `inverse_q()` and `bandwidth()`.
    /// Use only one of them.
    ///
    /// # Arguments
    /// * `s`: Shelf slope. A slope of `1.0` is maximally steep without overshoot.
    pub fn shelf_slope(&mut self, s: T) -> &mut Self {
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
    pub fn lowpass(&self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let b = self.gain * 0.5.as_() * (T::one() - fcos);
        [
            b,
            (2.0).as_() * b,
            b,
            T::one() + alpha,
            (-2.0).as_() * fcos,
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
    pub fn highpass(&self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let b = self.gain * 0.5.as_() * (T::one() + fcos);
        [
            b,
            (-2.0).as_() * b,
            b,
            T::one() + alpha,
            (-2.0).as_() * fcos,
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
    pub fn bandpass(&self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let b = self.gain * alpha;
        [
            b,
            T::zero(),
            -b,
            T::one() + alpha,
            (-2.0).as_() * fcos,
            T::one() - alpha,
        ]
    }

    /// A notch filter
    ///
    /// Has zero gain at the critical frequency.
    pub fn notch(&self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let f2 = (-2.0).as_() * fcos;
        [
            self.gain,
            f2 * self.gain,
            self.gain,
            T::one() + alpha,
            f2,
            T::one() - alpha,
        ]
    }

    /// An allpass filter
    ///
    /// Has constant `gain` at all frequency but a variable phase shift.
    pub fn allpass(&self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let f2 = (-2.0).as_() * fcos;
        [
            (T::one() - alpha) * self.gain,
            f2 * self.gain,
            (T::one() + alpha) * self.gain,
            T::one() + alpha,
            f2,
            T::one() - alpha,
        ]
    }

    /// A peaking/dip filter
    ///
    /// Has `gain*shelf_gain` at critical frequency and `gain` elsewhere.
    pub fn peaking(&self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let f2 = (-2.0).as_() * fcos;
        [
            (T::one() + alpha * self.shelf) * self.gain,
            f2 * self.gain,
            (T::one() - alpha * self.shelf) * self.gain,
            T::one() + alpha / self.shelf,
            f2,
            T::one() - alpha / self.shelf,
        ]
    }

    /// Low shelf
    ///
    /// Approaches `gain*shelf_gain` below critical frequency and `gain` above.
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
    pub fn lowshelf(&self) -> [T; 6] {
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

    /// Low shelf
    ///
    /// Approaches `gain*shelf_gain` above critical frequency and `gain` below.
    pub fn highshelf(&self) -> [T; 6] {
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

    /// I/HO
    ///
    /// Notch, integrating below, flat `shelf_gain` above
    pub fn iho(&self) -> [T; 6] {
        let (fcos, alpha) = self.fcos_alpha();
        let a = (T::one() + fcos) / (10.0.sqrt().as_() * self.shelf);
        [
            2.0.as_() * self.gain * (T::one() + alpha),
            (-4.0).as_() * self.gain * fcos,
            2.0.as_() * self.gain * (T::one() - alpha),
            a + self.w0.sin(),
            (-2.0).as_() * a,
            a - self.w0.sin(),
        ]
    }
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
                (0.01, Tol::GainDb(17.0, 0.02)),
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
                (0.1, Tol::GainDb(-5.0, 0.02)),
                (4e-1, Tol::GainDb(-2.0, 0.01)),
            ],
        );
    }

    #[test]
    fn bandpass() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.02)
                .bandwidth(2.0)
                .gain_db(3.0)
                .bandpass(),
            &[
                (1e-4, Tol::GainBelowDb(-35.0)),
                (0.01, Tol::GainDb(0.0, 0.02)),
                (0.02, Tol::GainDb(3.0, 0.01)),
                (0.04, Tol::GainDb(0.0, 0.04)),
                (4e-1, Tol::GainBelowDb(-25.0)),
            ],
        );
    }

    #[test]
    fn allpass() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.02)
                .gain_db(-10.0)
                .allpass(),
            &[
                (1e-4, Tol::GainDb(-10.0, 0.01)),
                (0.01, Tol::GainDb(-10.0, 0.01)),
                (0.02, Tol::GainDb(-10.0, 0.01)),
                (0.04, Tol::GainDb(-10.0, 0.01)),
                (4e-1, Tol::GainDb(-10.0, 0.01)),
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
                (0.01, Tol::GainDb(-3.0, 0.02)),
                (0.02, Tol::GainBelowDb(-140.0)),
                (0.04, Tol::GainDb(-3.0, 0.02)),
                (4e-1, Tol::GainDb(0.0, 0.01)),
            ],
        );
    }

    #[test]
    fn peaking() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.02)
                .bandwidth(2.0)
                .gain_db(-10.0)
                .shelf_db(20.0)
                .peaking(),
            &[
                (1e-4, Tol::GainDb(-10.0, 0.01)),
                (0.01, Tol::GainDb(0.0, 0.04)),
                (0.02, Tol::GainDb(10.0, 0.01)),
                (0.04, Tol::GainDb(0.0, 0.04)),
                (4e-1, Tol::GainDb(-10.0, 0.05)),
            ],
        );
    }

    #[test]
    fn highshelf() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.02)
                .gain_db(-20.0)
                .shelf_db(10.0)
                .highshelf(),
            &[
                (1e-6, Tol::GainDb(-20.0, 0.01)),
                (1e-4, Tol::GainDb(-20.0, 0.01)),
                (0.02, Tol::GainDb(-15.0, 0.01)),
                (4e-1, Tol::GainDb(-10.0, 0.01)),
            ],
        );
    }

    #[test]
    fn lowshelf() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.02)
                .gain_db(-10.0)
                .shelf_db(30.0)
                .lowshelf(),
            &[
                (1e-6, Tol::GainDb(20.0, 0.01)),
                (1e-4, Tol::GainDb(20.0, 0.01)),
                (0.02, Tol::GainDb(5.0, 0.01)),
                (4e-1, Tol::GainDb(-10.0, 0.01)),
            ],
        );
    }

    #[test]
    fn iho() {
        check_coeffs(
            &Filter::default()
                .critical_frequency(0.01)
                .gain_db(-20.0)
                .shelf_db(20.0)
                .q(10.)
                .iho(),
            &[
                (1e-5, Tol::GainDb(40.0, 0.01)),
                (0.01, Tol::GainDb(-40.0, 0.05)),
                (4.99e-1, Tol::GainDb(0.0, 0.01)),
            ],
        );
    }
}
