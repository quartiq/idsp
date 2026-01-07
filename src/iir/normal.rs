use core::ops::{Add, Mul, Neg};
use dsp_process::{SplitInplace, SplitProcess};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::Complex;

use super::SosState;

/// Normal form second order section
///
/// Also known as Rader Gold oscillator, or Chamberlain form IIR.
/// A direct form implementation has bad pole resolution near the real axis.
/// The normal form has constant pole resolution in the plane.
///
/// This implementation includes a second order all-zeros before the all-poles section.
///
/// The `y0`/`y1` fields of [`SosState`] hold the in-phase and quadrature
/// components of the current output.
#[derive(Debug, Clone, Default)]
pub struct Normal<C> {
    /// Feed forward coefficients
    pub b: [C; 3],
    /// Pole
    ///
    /// Conjugate pole pair at: `p[0] +- 1j*p[1]`
    pub p: Complex<C>,
}

impl<C: Copy + Mul<T, Output = A> + Neg<Output = C>, A: Add<Output = A> + Into<T>, T: Copy>
    SplitProcess<T, T, SosState<T>> for Normal<C>
{
    fn process(&self, state: &mut SosState<T>, x0: T) -> T {
        let b = &self.b;
        let p = &self.p;
        let xy = &mut state.xy;
        let y1: T =
            (b[0] * x0 + b[1] * xy[0] + b[2] * xy[1] + p.re() * xy[3] + (-p.im() * xy[2])).into();
        let y0: T = (p.im() * xy[3] + p.re() * xy[2]).into();
        // let mut acc = 0;
        // acc += x0 as i64 * b[0].inner as i64;
        // acc += xy[0] as i64 * b[1].inner as i64;
        // acc += xy[1] as i64 * b[2].inner as i64;
        // acc += xy[3] as i64 * p[0].inner as i64;
        // acc += xy[2] as i64 * -p[1].inner as i64;
        // let y1 = (acc >> F) as i32;
        // let mut acc = 0;
        // acc += xy[3] as i64 * p[1].inner as i64;
        // acc += xy[2] as i64 * p[0].inner as i64;
        // let y0 = (acc >> F) as i32;
        *xy = [x0, xy[0], y0, y1];
        y0
    }
}

impl<C, T: Copy> SplitInplace<T, SosState<T>> for Normal<C> where
    Self: SplitProcess<T, T, SosState<T>>
{
}

impl<C: Neg<Output = C> + From<f64>> From<&[[f64; 3]; 2]> for Normal<C> {
    fn from(ba: &[[f64; 3]; 2]) -> Self {
        let a0 = 1.0 / ba[1][0];
        let b = [ba[0][0] * a0, ba[0][1] * a0, ba[0][2] * a0];
        // Roots of a0 * z * z + a1 * z + a2
        let p2 = -0.5 * ba[1][1];
        let p = [p2 * a0, ((ba[1][0] * ba[1][2] - p2.powi(2)).sqrt() * a0)];
        Self {
            b: b.map(Into::into),
            p: Complex(p.map(Into::into)),
        }
    }
}
