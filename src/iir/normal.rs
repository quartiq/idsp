#[cfg(not(feature = "std"))]
use num_traits::float::Float as _;

use super::SosState;
use dsp_fixedpoint::Q32;
use dsp_process::{SplitInplace, SplitProcess};

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
pub struct Normal<const F: i8> {
    /// Feed forward coefficients
    pub b: [Q32<F>; 3],
    /// Pole
    ///
    /// Conjugate pole pair at: `p[0] +- 1j*p[1]`
    pub p: [Q32<F>; 2],
}

impl<const F: i8> SplitProcess<i32, i32, SosState> for Normal<F> {
    fn process(&self, state: &mut SosState, x0: i32) -> i32 {
        let b = &self.b;
        let p = &self.p;
        let xy = &mut state.xy;
        let mut acc = 0;
        acc += x0 as i64 * b[0].inner as i64;
        acc += xy[0] as i64 * b[1].inner as i64;
        acc += xy[1] as i64 * b[2].inner as i64;
        acc += xy[3] as i64 * p[0].inner as i64;
        acc += xy[2] as i64 * -p[1].inner as i64;
        let y1 = (acc >> F) as i32;
        let mut acc = 0;
        acc += xy[3] as i64 * p[1].inner as i64;
        acc += xy[2] as i64 * p[0].inner as i64;
        let y0 = (acc >> F) as i32;
        *xy = [x0, xy[0], y0, y1];
        y0
    }
}

impl<const F: i8> SplitInplace<i32, SosState> for Normal<F> {}

impl<const F: i8> From<&[[f64; 3]; 2]> for Normal<F> {
    fn from(ba: &[[f64; 3]; 2]) -> Self {
        let a0 = (1u64 << F) as f64 / ba[1][0];
        let b = [
            (ba[0][0] * a0).round() as i32,
            (ba[0][1] * a0).round() as i32,
            (ba[0][2] * a0).round() as i32,
        ]
        .map(Q32::new);
        // Roots of a0 * z * z + a1 * z + a2
        let p2 = -0.5 * ba[1][1];
        let p = [
            (p2 * a0).round() as i32,
            ((ba[1][0] * ba[1][2] - p2.powi(2)).sqrt() * a0).round() as i32,
        ]
        .map(Q32::new);
        Self { b, p }
    }
}
