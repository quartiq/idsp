//! Frequency-response utilities for second-order sections.

use core::iter;

use num_complex::Complex;
use num_traits::{AsPrimitive, Float, FloatConst};

use crate::iir::Biquad;

fn polyval<T: Float>(p: &[T], x: Complex<T>) -> Complex<T> {
    p.iter()
        .copied()
        .chain(iter::once(T::zero()))
        .fold(Complex::new(T::zero(), T::zero()), |a, pi| {
            a * x + Complex::new(pi, T::zero())
        })
}

/// Evaluate a normalized `[b, a]` transfer function on the unit circle.
///
/// `frequency` is relative to the sample rate.
pub fn ba_frequency_response<T: Float + FloatConst>(b: &[T], a: &[T], frequency: T) -> Complex<T> {
    let z = Complex::new(T::zero(), -T::TAU() * frequency).exp();
    polyval(b, z) / polyval(a, z)
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
