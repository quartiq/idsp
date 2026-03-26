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
/// This implementation includes a second order all-zeros before the all-poles section.
///
/// The `y0`/`y1` fields of [`DirectForm1`] hold the in-phase and quadrature
/// components of the current output.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Normal<C> {
    /// Feed forward coefficients
    pub b: [C; 3],
    /// Pole
    ///
    /// Conjugate pole pair at: `p.re() +- 1j*p.im()`
    pub p: Complex<C>,
}

/// The y1, y2 aren't DF1 but y1.re() and y1.im()
impl<
    C: Copy + Mul<T, Output = A> + Neg<Output = C>,
    A: Add<Output = A> + AsPrimitive<T>,
    T: 'static + Copy,
> SplitProcess<T, T, DirectForm1<T>> for Normal<C>
{
    fn process(&self, state: &mut DirectForm1<T>, x0: T) -> T {
        let y1 = (self.b[0] * x0
            + self.b[1] * state.x[0]
            + self.b[2] * state.x[1]
            + self.p.re() * state.y[0][1]
            + -self.p.im() * state.y[0][0])
            .as_();
        let y0: T = (self.p.im() * state.y[0][1] + self.p.re() * state.y[0][0]).as_();
        state.x = [x0, state.x[0]];
        state.y[0] = [y0, y1];
        y0
    }
}

impl<C, T: Copy> SplitInplace<T, DirectForm1<T>> for Normal<C> where
    Self: SplitProcess<T, T, DirectForm1<T>>
{
}

impl<C: Neg<Output = C> + From<f64>> From<&[[f64; 3]; 2]> for Normal<C> {
    fn from(ba: &[[f64; 3]; 2]) -> Self {
        let a0 = 1.0 / ba[1][0];
        let b = [ba[0][0] * a0, ba[0][1] * a0, ba[0][2] * a0];
        // Roots of a0 * z * z + a1 * z + a2
        let p2 = -0.5 * ba[1][1];
        let pq = ba[1][0] * ba[1][2] - p2.powi(2);
        assert!(pq >= 0.0);
        let p = [p2 * a0, pq.sqrt() * a0];
        Self {
            b: b.map(Into::into),
            p: Complex(p.map(Into::into)),
        }
    }
}
