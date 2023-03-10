use core::ops::{Add, Mul, Neg};

pub fn abs<T>(x: T) -> T
where
    T: PartialOrd + Default + Neg<Output = T>,
{
    if x >= T::default() {
        x
    } else {
        -x
    }
}

// These are implemented here because core::f32 doesn't have them (yet).
// They are naive and don't handle inf/nan.
// `compiler-intrinsics`/llvm should have better (robust, universal, and
// faster) implementations.

pub fn copysign<T>(x: T, y: T) -> T
where
    T: PartialOrd + Default + Neg<Output = T>,
{
    if (x >= T::default() && y >= T::default()) || (x <= T::default() && y <= T::default()) {
        x
    } else {
        -x
    }
}

// Multiply-accumulate vectors `x` and `a`.
//
// A.k.a. dot product.
// Rust/LLVM optimize this nicely.
pub fn macc<T>(y0: T, x: &[T], a: &[T]) -> T
where
    T: Add<Output = T> + Mul<Output = T> + Copy,
{
    x.iter()
        .zip(a)
        .map(|(x, a)| *x * *a)
        .fold(y0, |y, xa| y + xa)
}

pub fn macc_i32(y0: i32, x: &[i32], a: &[i32], shift: u32) -> i32 {
    // Rounding bias, half up
    let y0 = ((y0 as i64) << shift) + (1 << (shift - 1));
    let y = x
        .iter()
        .zip(a)
        .map(|(x, a)| *x as i64 * *a as i64)
        .fold(y0, |y, xa| y + xa);
    (y >> shift) as i32
}

#[cfg(feature = "nightly")]
pub fn unchecked_shr(x: i32, k: u32) -> i32 {
    unsafe { x.unchecked_shr(k as _) }
}

#[cfg(not(feature = "nightly"))]
pub fn unchecked_shr(x: i32, k: u32) -> i32 {
    x >> k
}

#[cfg(feature = "nightly")]
pub fn unchecked_shl(x: i32, k: u32) -> i32 {
    unsafe { x.unchecked_shl(k as _) }
}

#[cfg(not(feature = "nightly"))]
pub fn unchecked_shl(x: i32, k: u32) -> i32 {
    x << k
}
