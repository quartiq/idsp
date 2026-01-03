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
    ops::{AddAssign, DivAssign, MulAssign, RemAssign, ShlAssign, ShrAssign, SubAssign},
    ops::{BitAnd, BitOr, BitXor, Not},
    ops::{BitAndAssign, BitOrAssign, BitXorAssign},
};

/// Shift summary trait
///
/// Wrapping supports `Sh{lr}<usize>`
pub trait Base: Copy + Shl<usize, Output = Self> + Shr<usize, Output = Self> {
    /// Signed shift (positive: left)
    ///
    /// `x*2**f`
    #[inline(always)]
    fn shs(self, f: i8) -> Self {
        if f >= 0 {
            self << (f as _)
        } else {
            self >> (-f as _)
        }
    }

    /// Number of bits
    const BITS: u32;

    /// Lowest value
    const MIN: Self;

    /// Highes value
    const MAX: Self;
}

/// Accumulator summary trait
pub trait Accu<T>: Base {
    /// Convert to base
    fn down(self) -> T;

    /// Converto from base
    fn up(x: T) -> Self;
}

/// Fixed point with F fractional bits
///
/// `F` positive indicates bits right of the decimal point.
/// `F` negative is supported analogously.
///
/// * `Q32<31>` is `(-1..1).step_by(2^-31)`
/// * `Q<i16, _, 20>` is `(-1/32..1/32).step_by(2^-20)`
/// * `Q<u8, _, 4>` is `(0..16).step_by(1/16)`
/// * `Q<u8, _, -2>` is `(0..1024).step_by(4)`
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
    /// Step between distinct numbers
    pub const DELTA: f32 = if F > 0 {
        1.0 / (1u128 << F) as f32
    } else {
        (1u128 << -F) as f32
    };

    /// Create a new fixed point number from a given representation
    pub const fn new(inner: T) -> Self {
        Self {
            _accu: PhantomData,
            inner,
        }
    }
}

impl<T: Base, A, const F: i8> Q<T, A, F> {
    /// Number of bits of the base type
    pub const BITS: u32 = T::BITS;

    /// Lowest value
    pub const MIN: Self = Self::new(T::MIN);

    /// Highest value
    pub const MAX: Self = Self::new(T::MAX);

    /// Convert to a different number of fractional bits (truncating)
    ///
    /// Use this liberally for Add/Sub/Rem with Q's of different F.
    pub fn scale<const F1: i8>(self) -> Q<T, A, F1> {
        Q::new(self.inner.shs(F1 - F))
    }

    /// Return the integer part
    pub fn trunc(self) -> T {
        self.inner.shs(-F)
    }
}

impl<T: Base, A: Accu<T>, const F: i8> Q<T, A, F> {
    /// Number of bits of the accumulator type
    pub const ACCU_BITS: u32 = A::BITS;
}

impl<T: Base, A: Accu<T>, const F: i8> From<(T, i8)> for Q<T, A, F> {
    fn from(value: (T, i8)) -> Self {
        Self::new(value.0.shs(F - value.1))
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
        Self::new((value * const { 1.0 / Self::DELTA }).round().as_())
    }
}

impl<T: 'static + Copy, A, const F: i8> From<f64> for Q<T, A, F>
where
    f64: AsPrimitive<T>,
{
    fn from(value: f64) -> Self {
        Self::new((value * const { 1.0 / Self::DELTA as f64 }).round().as_())
    }
}

impl<T: AsPrimitive<Self>, A, const F: i8> From<Q<T, A, F>> for f32 {
    fn from(value: Q<T, A, F>) -> Self {
        value.inner.as_() * Q::<T, A, F>::DELTA
    }
}

impl<T: AsPrimitive<Self>, A, const F: i8> From<Q<T, A, F>> for f64 {
    fn from(value: Q<T, A, F>) -> Self {
        value.inner.as_() * Q::<T, A, F>::DELTA as Self
    }
}

macro_rules! forward_unop {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<Output = T>, A, const F: i8> $tr for Q<T, A, F> {
            type Output = Self;

            fn $m(self) -> Self::Output {
                Self::new(<T as $tr>::$m(self.inner))
            }
        }
    };
}
forward_unop!(Neg::neg);
forward_unop!(Not::not);

macro_rules! forward_sh_op {
    ($tr:ident::$m:ident) => {
        impl<U, T: $tr<U, Output = T>, A, const F: i8> $tr<U> for Q<T, A, F> {
            type Output = Self;

            fn $m(self, rhs: U) -> Self::Output {
                Self::new(<T as $tr<U>>::$m(self.inner, rhs))
            }
        }
    };
}
forward_sh_op!(Shr::shr);
forward_sh_op!(Shl::shl);

macro_rules! forward_sh_assign_op {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<U>, U, A, const F: i8> $tr<U> for Q<T, A, F> {
            fn $m(&mut self, rhs: U) {
                <T as $tr<U>>::$m(&mut self.inner, rhs)
            }
        }
    };
}
forward_sh_assign_op!(ShrAssign::shr_assign);
forward_sh_assign_op!(ShlAssign::shl_assign);

macro_rules! forward_binop {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T, Output = T>, A, const F: i8> $tr for Q<T, A, F> {
            type Output = Self;

            fn $m(self, rhs: Self) -> Self::Output {
                Self::new(<T as $tr>::$m(self.inner, rhs.inner))
            }
        }
    };
}
forward_binop!(Rem::rem);
forward_binop!(Add::add);
forward_binop!(Sub::sub);
forward_binop!(BitAnd::bitand);
forward_binop!(BitOr::bitor);
forward_binop!(BitXor::bitxor);

/// The crucial exception to https://github.com/rust-lang/rust/pull/93208#issuecomment-1019310634
/// See also the `Mul<Q>,Div<Q> for T` in `impl_q!()`
macro_rules! forward_binop_foreign {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T, Output = T>, A, const F: i8> $tr<T> for Q<T, A, F> {
            type Output = Self;

            fn $m(self, rhs: T) -> Self::Output {
                Self::new(<T as $tr<T>>::$m(self.inner, rhs))
            }
        }
    };
}
forward_binop_foreign!(Mul::mul);
forward_binop_foreign!(Div::div);

macro_rules! forward_assign_op_foreign {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T>, A, const F: i8> $tr<T> for Q<T, A, F> {
            fn $m(&mut self, rhs: T) {
                <T as $tr>::$m(&mut self.inner, rhs)
            }
        }
    };
}
forward_assign_op_foreign!(MulAssign::mul_assign);
forward_assign_op_foreign!(DivAssign::div_assign);

macro_rules! forward_assign_op {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T>, A, const F: i8> $tr for Q<T, A, F> {
            fn $m(&mut self, rhs: Self) {
                <T as $tr>::$m(&mut self.inner, rhs.inner)
            }
        }
        forward_assign_op_foreign!($tr::$m);
    };
}
forward_assign_op!(RemAssign::rem_assign);
forward_assign_op!(AddAssign::add_assign);
forward_assign_op!(SubAssign::sub_assign);
forward_assign_op!(BitAndAssign::bitand_assign);
forward_assign_op!(BitOrAssign::bitor_assign);
forward_assign_op!(BitXorAssign::bitxor_assign);

/// Q *= Q'
impl<T: Copy, A: Accu<T> + Mul<A, Output = A>, const F: i8, const F1: i8> MulAssign<Q<T, A, F1>>
    for Q<T, A, F>
{
    fn mul_assign(&mut self, rhs: Q<T, A, F1>) {
        self.inner = (A::up(self.inner) * A::up(rhs.inner)).shs(-F1).down();
    }
}

/// Q /= Q'
impl<T: Base + Div<T, Output = T>, A: Accu<T> + Div<A, Output = A>, const F: i8, const F1: i8>
    DivAssign<Q<T, A, F1>> for Q<T, A, F>
{
    fn div_assign(&mut self, rhs: Q<T, A, F1>) {
        self.inner = if F1 > 0 {
            (A::up(self.inner).shs(F1) / A::up(rhs.inner)).down()
        } else {
            self.inner.shs(F1) / rhs.inner
        };
    }
}

/// Q*Q -> Q
impl<T: Copy, A: Accu<T> + Mul<A, Output = A>, const F: i8> Mul for Q<T, A, F> {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

/// Q/Q -> Q
impl<T: Base + Div<T, Output = T>, A: Accu<T> + Div<A, Output = A>, const F: i8> Div
    for Q<T, A, F>
{
    type Output = Self;
    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
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

macro_rules! impl_fmt {
    ($tr:path) => {
        impl<T: Base, A, const F: i8> $tr for Q<T, A, F>
        where
            Self: Copy + Into<f64> + Into<f32>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if T::BITS > 24 {
                    <f64 as $tr>::fmt(&(*self).into(), f)
                } else {
                    <f32 as $tr>::fmt(&(*self).into(), f)
                }
            }
        }
    };
}
impl_fmt!(fmt::Display);
impl_fmt!(fmt::UpperExp);
impl_fmt!(fmt::LowerExp);

// TODO: Binary, Octal, UpperHex, LowerHex

macro_rules! impl_q {
    // Primitive
    ($alias:ident<$t:ty, $a:ty>) => {
        impl_q!($alias<$t, $a>, |x| x as _, <$a>::MIN, <$a>::MAX);
    };
    // Newtype
    ($alias:ident<$t:ty, $a:ty>, $wrap:tt) => {
        impl_q!($alias<$wrap<$t>, $wrap<$a>>, |x: $wrap<_>| $wrap(x.0 as _), Self(<$a>::MIN), Self(<$a>::MAX));
    };
    ($alias:ident<$t:ty, $a:ty>, $as:expr, $min:expr, $max:expr) => {
        impl Base for $a {
            const BITS: u32 = <$t as Base>::BITS * 2; // by convention
            const MIN: Self = $min;
            const MAX: Self = $max;
        }

        impl Accu<$t> for $a {
            #[inline(always)]
            fn down(self) -> $t {
                $as(self)
            }

            #[inline(always)]
            fn up(x: $t) -> Self {
                $as(x)
            }
        }

        #[doc = concat!("Fixed point [`", stringify!($t), "`]")]
        pub type $alias<const F: i8 = {<$t as Base>::BITS as i8 - 1}> = Q<$t, $a, F>;

        /// Scale from integer base type
        impl<const F: i8> From<$t> for Q<$t, $a, F> {
            fn from(value: $t) -> Self {
                Self::new(value.shs(F))
            }
        }

        /// I*Q -> I
        impl<const F: i8> Mul<Q<$t, $a, F>> for $t {
            type Output = $t;

            fn mul(self, rhs: Q<$t, $a, F>) -> Self::Output {
                (<$a>::up(self) * <$a>::up(rhs.inner)).shs(-F).down()
            }
        }

        /// I/Q -> I
        impl<const F: i8> Div<Q<$t, $a, F>> for $t {
            type Output = $t;

            fn div(self, rhs: Q<$t, $a, F>) -> Self::Output {
                if F > 0 {
                    (<$a>::up(self).shs(F) / <$a>::up(rhs.inner)).down()
                } else {
                    self.shs(F) / rhs.inner
                }
            }
        }
    };
}
// Signed
impl Base for i8 {
    const BITS: u32 = Self::BITS;
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
}
impl_q!(Q8<i8, i16>);
impl_q!(Q16<i16, i32>);
impl_q!(Q32<i32, i64>);
impl_q!(Q64<i64, i128>);
// Unsigned (_P_ositive)
impl Base for u8 {
    const BITS: u32 = Self::BITS;
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
}
impl_q!(P8<u8, u16>);
impl_q!(P16<u16, u32>);
impl_q!(P32<u32, u64>);
impl_q!(P64<u64, u128>);
// _W_rapping signed
impl Base for Wrapping<i8> {
    const BITS: u32 = i8::BITS;
    const MIN: Self = Self(i8::MIN);
    const MAX: Self = Self(i8::MAX);
}
impl_q!(W8<i8, i16>, Wrapping);
impl_q!(W16<i16, i32>, Wrapping);
impl_q!(W32<i32, i64>, Wrapping);
impl_q!(W64<i64, i128>, Wrapping);
// Wrapping vnsigned
impl Base for Wrapping<u8> {
    const BITS: u32 = u8::BITS;
    const MIN: Self = Self(u8::MIN);
    const MAX: Self = Self(u8::MAX);
}
impl_q!(V8<u8, u16>, Wrapping);
impl_q!(V16<u16, u32>, Wrapping);
impl_q!(V32<u32, u64>, Wrapping);
impl_q!(V64<u64, u128>, Wrapping);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple() {
        assert_eq!(Q32::<5>::from(4) * Q32::<5>::from(3), (3 * 4).into());
        assert_eq!(Q32::<5>::from(12) / Q32::<5>::from(6), 2.into());
        assert_eq!(7 * Q32::<4>::new(0x33), 7 * 3 + ((3 * 7) >> 4));
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", Q32::<9>::new(0x12345)), "145.634765625");
        assert_eq!(format!("{}", Q32::<9>::from(99)), "99");
    }
}
