#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

use num_traits::AsPrimitive;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::FloatCore;

use core::{
    fmt, iter,
    marker::PhantomData,
    num::Wrapping,
    ops::{Add, Div, Mul, Neg, Rem, Shl, Shr, Sub},
};

// TODO: Assign, Bit, BitAssign traits

/// Fixed point with F fractional bits.
///
/// F negative is supported analogously
///
/// * Q<i32, 31> is (-1..1).step_by(2^-31)
/// * Q<i16, 20> is (-1/32..1/32).step_by(2^-20)
/// * Q<u8, 4> is (0..16).step_by(1/16)
/// * Q<u8, -2> is (0..1024).step_by(4)
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
    /// The accumulator type
    _accu: PhantomData<A>,
    /// The inner value representation
    pub inner: T,
}

impl<T, A, const F: i8> Q<T, A, F> {
    /// Create a new fixed point number
    pub const fn new(inner: T) -> Self {
        Self {
            _accu: PhantomData,
            inner,
        }
    }
}

/// Signed shift (positive: left)
///
/// `x*2**f`
#[inline(always)]
fn shs<T: Shl<usize, Output = T> + Shr<usize, Output = T>>(x: T, f: i8) -> T {
    if f >= 0 {
        x << (f as _)
    } else {
        x >> (-f as _)
    }
}

/// Numerical summary trait
pub trait Num<Other: 'static + Copy>:
    'static
    + Sized
    + Copy
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
{
    /// Convert to/from larger
    fn as_(self) -> Other;
}

impl<T: Num<A>, A: Num<T>, const F: i8> Q<T, A, F> {
    /// Convert to a different number of fractional bits (truncating)
    ///
    /// Use this liberally for Add/Sub/Rem with Q's of different F.
    pub fn scale<const F1: i8>(self) -> Q<T, A, F1> {
        Q::new(shs(self.inner, F1 - F))
    }

    /// Return the integer part
    pub fn trunc(self) -> Self {
        Self::new(shs(self.inner, -F))
    }
}

impl<T: Num<A>, A: Num<T>, const F: i8> From<(T, i8)> for Q<T, A, F> {
    fn from(value: (T, i8)) -> Self {
        Self::new(shs(value.0, F - value.1))
    }
}

impl<T, A, const F: i8> From<Q<T, A, F>> for (T, i8) {
    fn from(value: Q<T, A, F>) -> Self {
        (value.inner, F)
    }
}

impl<T: 'static + Copy, A, const F: i8> From<f32> for Q<T, A, F>
where
    f32: AsPrimitive<T>,
{
    fn from(value: f32) -> Self {
        Self::new((value * (1u128 << F) as f32).round().as_())
    }
}

impl<T: 'static + Copy, A, const F: i8> From<f64> for Q<T, A, F>
where
    f64: AsPrimitive<T>,
{
    fn from(value: f64) -> Self {
        Self::new((value * (1u128 << F) as f64).round().as_())
    }
}

impl<T: 'static + Copy + AsPrimitive<Self>, A, const F: i8> From<Q<T, A, F>> for f32 {
    fn from(value: Q<T, A, F>) -> Self {
        value.inner.as_() * (1.0 / (1u128 << F) as Self)
    }
}

impl<T: 'static + Copy + AsPrimitive<Self>, A, const F: i8> From<Q<T, A, F>> for f64 {
    fn from(value: Q<T, A, F>) -> Self {
        AsPrimitive::<f64>::as_(value.inner) * (1.0 / (1u128 << F) as Self)
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

/// Q*Q' -> Q
impl<T: Num<A>, A: Num<T>, const F: i8, const F1: i8> Mul<Q<T, A, F1>> for Q<T, A, F> {
    type Output = Self;

    fn mul(self, rhs: Q<T, A, F1>) -> Self::Output {
        Self::new(shs(self.inner.as_() * rhs.inner.as_(), -F1).as_())
    }
}

/// Q*I -> I
impl<T: Num<A>, A: Num<T>, const F: i8> Mul<T> for Q<T, A, F> {
    type Output = T;

    fn mul(self, rhs: T) -> Self::Output {
        shs(self.inner.as_() * rhs.as_(), -F).as_()
    }
}

/// Q/Q' -> Q
impl<T: Num<A>, A: Num<T>, const F: i8, const F1: i8> Div<Q<T, A, F1>> for Q<T, A, F> {
    type Output = Self;

    fn div(self, rhs: Q<T, A, F1>) -> Self::Output {
        Self::new(if F1 > 0 {
            (shs(self.inner.as_(), F1) / rhs.inner.as_()).as_()
        } else {
            shs(self.inner, F1) / rhs.inner
        })
    }
}

/// Q/I -> Q
impl<T: Div<T, Output = T>, A, const F: i8> Div<T> for Q<T, A, F> {
    type Output = Self;

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
    Self: Copy + Into<f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&(*self).into(), f)
    }
}

impl<T, A, const F: i8> fmt::UpperExp for Q<T, A, F>
where
    Self: Copy + Into<f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&(*self).into(), f)
    }
}

impl<T, A, const F: i8> fmt::LowerExp for Q<T, A, F>
where
    Self: Copy + Into<f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&(*self).into(), f)
    }
}

// TODO: Binary, Octal, UpperHex, LowerHex

macro_rules! impl_mul_q {
    ($alias:ident, $q:ty, $a:ty, $as:expr) => {
        impl Num<$a> for $q {
            #[inline(always)]
            fn as_(self) -> $a {
                $as(self)
            }
        }
        impl Num<$q> for $a {
            #[inline(always)]
            fn as_(self) -> $q {
                $as(self)
            }
        }

        /// Fixed point with F fractional bits
        pub type $alias<const F: i8> = Q<$q, $a, F>;

        /// Scale from integer base type
        impl<const F: i8> From<$q> for Q<$q, $a, F> {
            fn from(value: $q) -> Self {
                Self::new(shs(value, F))
            }
        }

        /// Integer truncation into base type
        impl<const F: i8> From<Q<$q, $a, F>> for $q {
            fn from(value: Q<$q, $a, F>) -> $q {
                shs(value.inner, -F)
            }
        }

        /// I*Q -> Q
        impl<const F: i8> Mul<Q<$q, $a, F>> for $q {
            type Output = Q<$q, $a, F>;

            fn mul(self, rhs: Q<$q, $a, F>) -> Self::Output {
                Q::new(self * rhs.inner)
            }
        }

        /// I/Q -> I
        impl<const F: i8> Div<Q<$q, $a, F>> for $q {
            type Output = $q;

            fn div(self, rhs: Q<$q, $a, F>) -> Self::Output {
                if F > 0 {
                    Num::<$q>::as_(shs(Num::<$a>::as_(self), F) / Num::<$a>::as_(rhs.inner))
                } else {
                    shs(self, F) / rhs.inner
                }
            }
        }
    };
}
// Signed
impl_mul_q!(Q8, i8, i16, |x| x as _);
impl_mul_q!(Q16, i16, i32, |x| x as _);
impl_mul_q!(Q32, i32, i64, |x| x as _);
impl_mul_q!(Q64, i64, i128, |x| x as _);
// Unsigned (_P_ositive)
impl_mul_q!(P8, u8, u16, |x| x as _);
impl_mul_q!(P16, u16, u32, |x| x as _);
impl_mul_q!(P32, u32, u64, |x| x as _);
impl_mul_q!(P64, u64, u128, |x| x as _);
// Wrapping signed
impl_mul_q!(W8, Wrapping<i8>, Wrapping<i16>, |x: Wrapping<_>| Wrapping(
    x.0 as _
));
impl_mul_q!(
    W16,
    Wrapping<i16>,
    Wrapping<i32>,
    |x: Wrapping<_>| Wrapping(x.0 as _)
);
impl_mul_q!(
    W32,
    Wrapping<i32>,
    Wrapping<i64>,
    |x: Wrapping<_>| Wrapping(x.0 as _)
);
impl_mul_q!(W64, Wrapping<i64>, Wrapping<i128>, |x: Wrapping<_>| {
    Wrapping(x.0 as _)
});
// Wrapping unsigned
impl_mul_q!(V8, Wrapping<u8>, Wrapping<u16>, |x: Wrapping<_>| Wrapping(
    x.0 as _
));
impl_mul_q!(
    V16,
    Wrapping<u16>,
    Wrapping<u32>,
    |x: Wrapping<_>| Wrapping(x.0 as _)
);
impl_mul_q!(
    V32,
    Wrapping<u32>,
    Wrapping<u64>,
    |x: Wrapping<_>| Wrapping(x.0 as _)
);
impl_mul_q!(V64, Wrapping<u64>, Wrapping<u128>, |x: Wrapping<_>| {
    Wrapping(x.0 as _)
});

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple() {
        assert_eq!(Q32::<5>::from(4) * Q32::<5>::from(3), (3 * 4).into());
        assert_eq!(Q32::<5>::from(12) / Q32::<5>::from(6), 2.into());
        assert_eq!(Q32::<4>::new(0x33) * 7, 7 * 3 + ((3 * 7) >> 4));
    }
}
