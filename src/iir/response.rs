//! Frequency-response utilities for second-order sections.

use crate::Complex;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::Float as _;
use num_traits::{AsPrimitive, Float, FloatConst};

use crate::iir::Biquad;

fn polyval<T: Float>(p: &[T], z: Complex<T>) -> Complex<T> {
    p.iter()
        .rev()
        .fold(Complex::new(T::zero(), T::zero()), |a, &p| {
            a * z + Complex::new(p, T::zero())
        })
}

/// Evaluate a normalized `[b, a]` transfer function on the unit circle.
///
/// `frequency` is relative to the sample rate.
pub fn ba_frequency_response<T: Float + FloatConst>(b: &[T], a: &[T], frequency: T) -> Complex<T> {
    let (i, r) = (-T::TAU() * frequency).sin_cos();
    let z = Complex::new(r, i);
    let q = polyval(a, z);
    polyval(b, z) * q.conj() / (q.re() * q.re() + q.im() * q.im())
}

impl<C> Biquad<C> {
    /// Evaluate the transfer function on the unit circle.
    ///
    /// `frequency` is relative to the sample rate.
    pub fn frequency_response<T: 'static + Float + FloatConst>(&self, frequency: T) -> Complex<T>
    where
        C: Copy + AsPrimitive<T>,
    {
        let [b0, b1, b2, a1, a2] = self.ba.map(AsPrimitive::as_);
        ba_frequency_response(&[b0, b1, b2], &[T::one(), -a1, -a2], frequency)
    }
}
