//! Normal form second order section

use core::ops::{Add, Mul, Neg};
use dsp_process::{SplitInplace, SplitProcess};

use num_traits::AsPrimitive;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::Complex;

use super::DirectForm1;

/// Normal form second order section
///
/// Also known as Rader Gold oscillator, or Chamberlain form IIR.
/// A direct form implementation has bad pole resolution near the real axis.
/// The normal form has constant pole resolution in the plane.
///
/// [`DirectForm1::y`] holds the two rotation components in its first row.
///
/// Size the state for the input amplitude and pole radius, independently of
/// output gain. With zero initial state, `|input| <= X`, and pole radius
/// `r < 1`, its norm is bounded by `X / (1 - r)` before quantization.
/// Allow for quantization and coefficient growth near real poles when choosing
/// fixed-point formats.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Normal<C> {
    /// Output weights for the input and the two previous rotation components.
    pub b: [C; 3],
    /// Pole
    ///
    /// Conjugate pole pair at: `p.re() +- 1j*p.im()`
    pub p: Complex<C>,
}

impl<
    C: Copy + Mul<T, Output = A> + Neg<Output = C>,
    A: Add<Output = A> + AsPrimitive<T>,
    T: 'static + Copy + Add<Output = T>,
> SplitProcess<T, T, DirectForm1<T>> for Normal<C>
{
    fn process(&self, state: &mut DirectForm1<T>, x0: T) -> T {
        let [u, v] = state.y[0];
        let y = (self.b[0] * x0 + self.b[1] * u + self.b[2] * v).as_();
        let rotated: T = (self.p.re() * u + self.p.im() * v).as_();
        state.y[0] = [rotated + x0, (self.p.re() * v + -self.p.im() * u).as_()];
        y
    }
}

impl<C, T: Copy> SplitInplace<T, DirectForm1<T>> for Normal<C> where
    Self: SplitProcess<T, T, DirectForm1<T>>
{
}

/// Convert a section with a non-real conjugate pole pair.
///
/// Panics for real poles. Output weights can grow large near the real axis.
impl<C: 'static + Copy> From<&[[f64; 3]; 2]> for Normal<C>
where
    f64: AsPrimitive<C>,
{
    fn from(ba: &[[f64; 3]; 2]) -> Self {
        let a0 = ba[1][0].recip();
        let [b0, b1, b2] = ba[0].map(|b| b * a0);
        let re = -0.5 * ba[1][1] * a0;
        let norm = ba[1][2] * a0;
        let im2 = norm - re * re;
        assert!(im2 > 0.0, "normal form requires non-real poles");
        let im = im2.sqrt();
        // Rotation driven along u gives U/X = z^-1(1 - re*z^-1)/D
        // and V/X = -im*z^-2/D, where D = 1 - 2*re*z^-1 + norm*z^-2.
        let c1 = b1 + 2.0 * re * b0;
        let b = [b0, c1, (norm * b0 - re * c1 - b2) / im];
        Self {
            b: b.map(AsPrimitive::as_),
            p: Complex([re.as_(), im.as_()]),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::iir::Biquad;

    #[test]
    #[should_panic(expected = "normal form requires non-real poles")]
    fn rejects_real_poles() {
        let _ = Normal::<f64>::from(&[[1.0, 0.0, 0.0], [1.0, -1.0, 0.25]]);
    }

    #[test]
    fn fixed_point_transfer() {
        use dsp_fixedpoint::Q32;
        let ba = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.25]];
        let normal = Normal::<Q32<30>>::from(&ba);
        let direct = Biquad::<Q32<30>>::from(ba);
        let mut ns = DirectForm1::default();
        let mut ds = DirectForm1::default();
        for i in 0..32 {
            let x: i32 = if i == 0 { 1 << 20 } else { 0 };
            let actual: i32 = normal.process(&mut ns, x);
            let expected: i32 = direct.process(&mut ds, x);
            assert!((actual - expected).abs() <= 1);
        }
    }

    #[test]
    fn fixed_point_step_with_headroom() {
        use dsp_fixedpoint::Q32;
        // Unity DC gain, poles 0.99 +/- 0.01j. The rotation, not output gain,
        // determines state headroom: 101 * amplitude fits i32.
        const AMPLITUDE: i32 = 1 << 24;
        let ba = [[0.0002, 0.0, 0.0], [1.0, -1.98, 0.9802]];
        let normal = Normal::<Q32<30>>::from(&ba);
        let reference = Biquad::<f64>::from(ba);
        let mut ns = DirectForm1::default();
        let mut rs = DirectForm1::default();
        for x in [AMPLITUDE, -AMPLITUDE, 0] {
            for _ in 0..1024 {
                let actual: i32 = normal.process(&mut ns, x);
                let expected: f64 = reference.process(&mut rs, x as f64);
                assert!((actual as f64 - expected).abs() < AMPLITUDE as f64 * 1e-5);
                assert!(
                    ns.y[0]
                        .iter()
                        .all(|s| s.unsigned_abs() < 101 * AMPLITUDE as u32)
                );
            }
        }
    }

    #[test]
    fn transfer_matches_biquad() {
        for ba in [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.25]],
            [[0.2, -0.3, 0.4], [2.0, -1.2, 0.5]],
            [[0.5, 0.2, -0.1], [1.0, 1.6, 0.81]],
        ] {
            let normal = Normal::<f64>::from(&ba);
            let direct = Biquad::<f64>::from(ba);
            let mut ns = DirectForm1::default();
            let mut ds = DirectForm1::default();
            for i in 0..256 {
                let x = if i == 0 { 1.0 } else { 0.0 };
                let expected = direct.process(&mut ds, x);
                let actual = normal.process(&mut ns, x);
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }
}
