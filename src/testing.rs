//! Tools to test and benchmark algorithms
#![allow(dead_code)]
use super::Complex;
use num_traits::Float;

/// Maximum acceptable error between a computed and actual value given fixed and relative
/// tolerances.
///
/// # Args
/// * `a` - First input.
/// * `b` - Second input. The relative tolerance is computed with respect to the maximum of the
/// absolute values of the first and second inputs.
/// * `rtol` - Relative tolerance.
/// * `atol` - Fixed tolerance.
///
/// # Returns
/// Maximum acceptable error.
pub fn max_error<T: Float>(a: T, b: T, rtol: T, atol: T) -> T {
    rtol * a.abs().max(b.abs()) + atol
}

/// Return whether two numbers are within absolute plus relative tolerance
pub fn isclose<T: Float>(a: T, b: T, rtol: T, atol: T) -> bool {
    (a - b).abs() <= max_error(a, b, rtol, atol)
}

/// Return whether all values are close
pub fn allclose<T: Float>(a: &[T], b: &[T], rtol: T, atol: T) -> bool {
    a.iter().zip(b).all(|(a, b)| isclose(*a, *b, rtol, atol))
}

/// Return whether both real and imaginary component are close
pub fn complex_isclose<T: Float>(a: Complex<T>, b: Complex<T>, rtol: T, atol: T) -> bool {
    isclose(a.re, b.re, rtol, atol) && isclose(a.im, b.im, rtol, atol)
}

/// Return whether all complex values are close
pub fn complex_allclose<T: Float>(a: &[Complex<T>], b: &[Complex<T>], rtol: T, atol: T) -> bool {
    a.iter()
        .zip(b)
        .all(|(a, b)| complex_isclose(*a, *b, rtol, atol))
}
