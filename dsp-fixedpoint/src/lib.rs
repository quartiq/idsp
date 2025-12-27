#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::FloatCore;

use core::{
    fmt, iter,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Rem, Shl, Shr, Sub},
};

/// Fixed point with F fractional bits.
///
/// * Q<i32, 32> is [-0.5, 0.5[
/// * Q<i16, 15> is [-1, 1[
/// * Q<u8, 4> is [0, 16-1/16]
///
/// F negative is supported analogously.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Q<T, A, const F: i8> {
    /// The inner value representing the fixed point number
    pub inner: T,
    /// The accumulator type
    _accu: PhantomData<A>,
}

impl<T, A, const F: i8> Q<T, A, F> {
    /// Create a new fixed point number
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            _accu: PhantomData,
        }
    }
}

/// Signed shift (positive: left)
///
/// `x*2**f`
#[inline(always)]
fn ssh<T: Shl<i8, Output = T> + Shr<i8, Output = T>>(x: T, f: i8) -> T {
    if f >= 0 { x << f } else { x >> -f }
}

impl<T: Shl<i8, Output = T> + Shr<i8, Output = T>, A, const F: i8> Q<T, A, F> {
    /// Convert to a different number of fractional bits (truncating)
    pub fn scale<const F1: i8>(self) -> Q<T, A, F1> {
        Q::new(ssh(self.inner, F1 - F))
    }

    /// Return the integer part
    pub fn trunc(self) -> Self {
        ssh(self.inner, -F).into()
    }
}

impl<T: Shl<i8, Output = T> + Shr<i8, Output = T>, A, const F: i8> From<T> for Q<T, A, F> {
    fn from(value: T) -> Self {
        Self::new(ssh(value, F))
    }
}

impl<T: Shl<i8, Output = T> + Shr<i8, Output = T>, A, const F: i8> From<(T, i8)> for Q<T, A, F> {
    fn from(value: (T, i8)) -> Self {
        Self::new(ssh(value.0, F - value.1))
    }
}

impl<T, A, const F: i8> From<Q<T, A, F>> for (T, i8) {
    fn from(value: Q<T, A, F>) -> Self {
        (value.inner, F)
    }
}

impl<T: Neg<Output = T>, A, const F: i8> Neg for Q<T, A, F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.inner)
    }
}

impl<U, T: Shr<U, Output = T>, A, const F: i8> Shr<U> for Q<T, A, F> {
    type Output = Self;

    fn shr(self, rhs: U) -> Self::Output {
        Self::new(self.inner >> rhs)
    }
}

impl<U, T: Shl<U, Output = T>, A, const F: i8> Shl<U> for Q<T, A, F> {
    type Output = Self;

    fn shl(self, rhs: U) -> Self::Output {
        Self::new(self.inner << rhs)
    }
}

impl<T: Rem<T, Output = T>, A, const F: i8> Rem for Q<T, A, F> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::new(self.inner % rhs.inner)
    }
}

impl<T: Add<T, Output = T>, A, const F: i8> Add for Q<T, A, F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.inner + rhs.inner)
    }
}

impl<T: Sub<T, Output = T>, A, const F: i8> Sub for Q<T, A, F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.inner - rhs.inner)
    }
}

/// Q*Q -> Q
impl<T, A, const F: i8> Mul for Q<T, A, F>
where
    Self: Mul<T, Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self * rhs.inner)
    }
}

/// Q/I -> Q
impl<T: Div<Output = T>, A, const F: i8> Div<T> for Q<T, A, F> {
    type Output = Q<T, A, F>;

    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.inner / rhs)
    }
}

impl<T: iter::Sum, A, const F: i8> iter::Sum for Q<T, A, F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new(iter.map(|i| i.inner).sum())
    }
}

impl<T: iter::Product, A, const F: i8> iter::Product for Q<T, A, F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new(iter.map(|i| i.inner).product())
    }
}

impl<T, A, const F: i8> fmt::Display for Q<T, A, F>
where
    f64: From<Self>,
    Self: Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&f64::from(*self), f)
    }
}

impl<T, A, const F: i8> fmt::UpperExp for Q<T, A, F>
where
    f64: From<Self>,
    Self: Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&f64::from(*self), f)
    }
}

impl<T, A, const F: i8> fmt::LowerExp for Q<T, A, F>
where
    f64: From<Self>,
    Self: Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&f64::from(*self), f)
    }
}

// TODO: Binary, Octal, UpperHex, LowerHex

macro_rules! impl_mul_q {
    ($alias:ident, $q:ty, $a:ty) => {
        /// Alias for fixed point integer T with accumulator A
        pub type $alias<const F: i8> = Q<$q, $a, F>;

        /// Integer truncation
        impl<A, const F: i8> From<Q<$q, A, F>> for $q {
            fn from(value: Q<$q, A, F>) -> $q {
                ssh(value.inner, -F)
            }
        }

        /// I*Q -> Q
        impl<A, const F: i8> Mul<Q<$q, A, F>> for $q {
            type Output = Q<$q, A, F>;

            fn mul(self, rhs: Q<$q, A, F>) -> Self::Output {
                Q::new(self * rhs.inner)
            }
        }

        /// Q/Q -> Q
        impl<const F: i8> Div for Q<$q, $a, F> {
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                Self::new(if F >= 0 {
                    ssh(self.inner as $a, F) / rhs.inner as $a
                } else {
                    self.inner as $a / ssh(rhs.inner as $a, -F)
                } as _)
            }
        }

        /// Q*I -> I
        impl<const F: i8> Mul<$q> for Q<$q, $a, F> {
            type Output = $q;

            fn mul(self, rhs: $q) -> Self::Output {
                ssh(self.inner as $a * rhs as $a, -F) as _
            }
        }

        /// I/Q -> I
        impl<const F: i8> Div<Q<$q, $a, F>> for $q {
            type Output = $q;

            fn div(self, rhs: Q<$q, $a, F>) -> Self::Output {
                (if F >= 0 {
                    ssh(self as $a, F) / rhs.inner as $a
                } else {
                    self as $a / ssh(rhs.inner as $a, -F) as $a
                }) as _
            }
        }

        impl<const F: i8> From<f32> for Q<$q, $a, F> {
            fn from(value: f32) -> Self {
                Self::new((value * (1 << F) as f32).round() as _)
            }
        }

        impl<const F: i8> From<f64> for Q<$q, $a, F> {
            fn from(value: f64) -> Self {
                Self::new((value * (1 << F) as f64).round() as _)
            }
        }

        impl<const F: i8> From<Q<$q, $a, F>> for f32 {
            fn from(value: Q<$q, $a, F>) -> Self {
                value.inner as Self * (1.0 / (1 << F) as Self)
            }
        }

        impl<const F: i8> From<Q<$q, $a, F>> for f64 {
            fn from(value: Q<$q, $a, F>) -> Self {
                value.inner as Self * (1.0 / (1 << F) as Self)
            }
        }
    };
}
impl_mul_q!(Q8, i8, i16);
impl_mul_q!(Q16, i16, i32);
impl_mul_q!(Q32, i32, i64);
impl_mul_q!(Q64, i64, i128);
impl_mul_q!(P8, u8, u16);
impl_mul_q!(P16, u16, u32);
impl_mul_q!(P32, u32, u64);
impl_mul_q!(P64, u64, u128);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple() {
        assert_eq!(Q32::<5>::from(4) * Q::from(3), (3 * 4).into());
        assert_eq!(Q32::<5>::from(12) / Q::from(6), 2.into());
        assert_eq!(Q32::<4>::new(0x33) * 7, 7 * 3 + ((3 * 7) >> 4));
    }
}
