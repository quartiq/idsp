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
/// Wrapping supports `Sh{lr}<usize>` only.
pub trait Shift: Copy + Shl<usize, Output = Self> + Shr<usize, Output = Self> {
    /// Signed shift (positive: left)
    ///
    /// `x*2**f`
    ///
    /// ```
    /// # use dsp_fixedpoint::Shift;
    /// assert_eq!(1i32.shs(1), 2);
    /// assert_eq!(4i32.shs(-1), 2);
    /// ```
    fn shs(self, f: i8) -> Self;
}

impl<T: Copy + Shl<usize, Output = T> + Shr<usize, Output = T>> Shift for T {
    #[inline(always)]
    fn shs(self, f: i8) -> Self {
        if f >= 0 {
            self << (f as _)
        } else {
            self >> (-f as _)
        }
    }
}

/// Integer bits
pub trait Int {
    /// Number of bits
    ///
    /// ```
    /// # use dsp_fixedpoint::Int;
    /// assert_eq!(i32::BITS, 32);
    /// ```
    const BITS: u32;

    /// Lowest value
    ///
    /// ```
    /// # use dsp_fixedpoint::Int;
    /// assert_eq!(i8::MIN, -128);
    /// ```
    const MIN: Self;

    /// Highes value
    ///
    /// ```
    /// # use dsp_fixedpoint::Int;
    /// assert_eq!(i8::MAX, 127);
    /// ```
    const MAX: Self;
}

/// Conversion trait between base and accumulator type
pub trait Accu<A> {
    /// Cast up to accumulator type
    ///
    /// This is a primitive cast.
    ///
    /// ```
    /// # use dsp_fixedpoint::Accu;
    /// assert_eq!(3i32.up(), 3i64);
    /// ```
    fn up(self) -> A;

    /// Cast down from accumulator type
    ///
    /// This is a primitive cast.
    ///
    /// ```
    /// # use dsp_fixedpoint::Accu;
    /// assert_eq!(i16::down(3i32), 3i16);
    /// ```
    fn down(a: A) -> Self;

    // /// Cast to f32
    // fn as_f32(self) -> f32;
    // /// Cast to f64
    // fn as_f64(self) -> f64;
    // /// Cast from f32
    // fn f32_as(value: f64) -> Self;
    // /// Cast from f64
    // fn f64_as(value: f64) -> Self;
}

/// Fixed point integer
///
/// Generics:
/// * `T`: Base integer
/// * `A`: Accumulator for intermediate results
/// * `F`: Number of fractional bits right of the decimal point
///
/// `F` negative is supported analogously.
///
/// * `Q32<31>` is `(-1..1).step_by(2^-31)`
/// * `Q<i16, _, 20>` is `(-1/32..1/32).step_by(2^-20)`
/// * `Q<u8, _, 4>` is `(0..16).step_by(1/16)`
/// * `Q<u8, _, -2>` is `(0..1024).step_by(4)`
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(Q8::<4>::from(3), Q8::new(3 << 4));
/// assert_eq!(7 * Q8::<4>::from(1.5), 10);
/// assert_eq!(7 / Q8::<4>::from(1.5), 4);
/// ```
#[derive(
    Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
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
    ///
    /// ```
    /// # use dsp_fixedpoint::Q32;
    /// assert_eq!(Q32::<31>::DELTA, 2f32.powi(-31));
    /// assert_eq!(Q32::<-4>::DELTA, 2f32.powi(4));
    /// ```
    pub const DELTA: f32 = if F > 0 {
        1.0 / (1u128 << F) as f32
    } else {
        (1u128 << -F) as f32
    };

    /// Create a new fixed point number from a given representation
    ///
    /// ```
    /// # use dsp_fixedpoint::P8;
    /// assert_eq!(P8::<9>::new(3).inner, 3);
    /// ```
    #[inline(always)]
    pub const fn new(inner: T) -> Self {
        Self {
            _accu: PhantomData,
            inner,
        }
    }
}

impl<T: Int, A, const F: i8> Int for Q<T, A, F> {
    /// Number of bits of the base type
    ///
    /// ```
    /// # use dsp_fixedpoint::{Int, Q32};
    /// assert_eq!(Q32::<7>::BITS, 32);
    /// ```
    const BITS: u32 = T::BITS;

    /// Lowest value
    ///
    /// ```
    /// # use dsp_fixedpoint::{Int, Q8};
    /// assert_eq!(Q8::<4>::MIN, (-16.0).into());
    /// ```
    const MIN: Self = Self::new(T::MIN);

    /// Highest value
    ///
    /// ```
    /// # use dsp_fixedpoint::{Int, Q8};
    /// assert_eq!(Q8::<4>::MAX, (16.0 - 1.0 / 16.0).into());
    /// ```
    const MAX: Self = Self::new(T::MAX);
}

impl<T: Int, A, const F: i8> Q<T, A, F> {
    /// Convert to a different number of fractional bits (truncating)
    ///
    /// Use this liberally for Add/Sub/Rem with Q's of different F.
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// assert_eq!(Q8::<4>::new(32).scale::<0>(), Q8::new(2));
    /// ```
    pub fn scale<const F1: i8>(self) -> Q<T, A, F1>
    where
        T: Shift,
    {
        Q::new(self.inner.shs(F1 - F))
    }

    /// Return the integer part
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// assert_eq!(Q8::<4>::new(0x35).trunc(), 0x3);
    /// ```
    pub fn trunc(self) -> T
    where
        T: Shift,
    {
        self.inner.shs(-F)
    }
}

/// Lossy conversion from a dynamically scaled integer
impl<T: Accu<A> + Shift, A, const F: i8> From<(T, i8)> for Q<T, A, F> {
    fn from(value: (T, i8)) -> Self {
        Self::new(value.0.shs(F - value.1))
    }
}

/// Lossless conversion into a dynamically scaled integer
impl<T, A, const F: i8> From<Q<T, A, F>> for (T, i8) {
    fn from(value: Q<T, A, F>) -> Self {
        (value.inner, F)
    }
}

/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(8 * Q8::<4>::from(0.25f32), 2);
/// ```
impl<T: 'static + Copy, A, const F: i8> From<f32> for Q<T, A, F>
where
    f32: AsPrimitive<T>,
{
    fn from(value: f32) -> Self {
        Self::new((value * const { 1.0 / Self::DELTA }).round().as_())
    }
}

/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(8 * Q8::<4>::from(0.25f64), 2);
/// ```
impl<T: 'static + Copy, A, const F: i8> From<f64> for Q<T, A, F>
where
    f64: AsPrimitive<T>,
{
    fn from(value: f64) -> Self {
        Self::new((value * const { 1.0 / Self::DELTA as f64 }).round().as_())
    }
}

/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(f32::from(Q8::<4>::new(4)), 0.25);
/// ```
impl<T: AsPrimitive<Self>, A, const F: i8> From<Q<T, A, F>> for f32 {
    fn from(value: Q<T, A, F>) -> Self {
        value.inner.as_() * Q::<T, A, F>::DELTA
    }
}

/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(f64::from(Q8::<4>::new(4)), 0.25);
/// ```
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

/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(Q8::<3>::from(3.5) + Q8::from(5.2), Q8::from(8.7));
/// assert_eq!(Q8::<3>::from(4.0) - Q8::from(3.2), Q8::from(0.8));
/// assert_eq!(Q8::<3>::from(3.5) % Q8::from(1), Q8::from(0.5));
/// ```
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

/// The notable exception to standard rules
/// (https://github.com/rust-lang/rust/pull/93208#issuecomment-1019310634)
/// This is for performance reasons
///
/// Q*T -> Q, Q/T -> Q
/// See also the T*Q -> T and T/Q -> T in impl_q!()
///
/// ```
/// # use dsp_fixedpoint::{Q8, Q};
/// assert_eq!(Q8::<3>::new(4) * 2, Q::new(8));
/// assert_eq!(Q8::<3>::new(4) / 2, Q8::new(2));
/// ```

/// Wide multiplication to accumulator
impl<T: Accu<A>, A: Mul<Output = A>, const F: i8> Mul<T> for Q<T, A, F> {
    type Output = Q<A, T, F>;
    #[inline(always)]
    fn mul(self, rhs: T) -> Q<A, T, F> {
        Q::new(self.inner.up() * rhs.up())
    }
}

impl<T: Div<Output = T>, A, const F: i8> Div<T> for Q<T, A, F> {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Q::new(self.inner / rhs)
    }
}

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
    };
}
forward_assign_op!(RemAssign::rem_assign);
forward_assign_op!(AddAssign::add_assign);
forward_assign_op!(SubAssign::sub_assign);
forward_assign_op!(BitAndAssign::bitand_assign);
forward_assign_op!(BitOrAssign::bitor_assign);
forward_assign_op!(BitXorAssign::bitxor_assign);

/// Q *= Q'
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// let mut q = Q8::<4>::from(0.25);
/// q *= Q8::<3>::from(3);
/// assert_eq!(q, Q8::from(0.75));
/// ```
impl<T: Copy + Accu<A>, A: Shift + Mul<A, Output = A>, const F: i8, const F1: i8>
    MulAssign<Q<T, A, F1>> for Q<T, A, F>
{
    fn mul_assign(&mut self, rhs: Q<T, A, F1>) {
        self.inner = T::down((self.inner.up() * rhs.inner.up()).shs(-F1));
    }
}

/// Q /= Q'
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// let mut q = Q8::<4>::from(0.75);
/// q /= Q8::<3>::from(3);
/// assert_eq!(q, Q8::from(0.25));
/// ```
impl<
    T: Copy + Shift + Accu<A> + Div<T, Output = T>,
    A: Shift + Div<A, Output = A>,
    const F: i8,
    const F1: i8,
> DivAssign<Q<T, A, F1>> for Q<T, A, F>
{
    fn div_assign(&mut self, rhs: Q<T, A, F1>) {
        self.inner = if F1 > 0 {
            T::down(self.inner.up().shs(F1) / rhs.inner.up())
        } else {
            self.inner.shs(F1) / rhs.inner
        };
    }
}

/// Q*Q -> Q
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(Q8::<4>::from(0.75) * Q8::from(3), Q8::from(2.25));
/// ```
impl<T, A, const F: i8> Mul for Q<T, A, F>
where
    Self: MulAssign,
{
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

/// Q/Q -> Q
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(Q8::<4>::from(3) / Q8::from(2), Q8::from(1.5));
/// ```
impl<T, A, const F: i8> Div for Q<T, A, F>
where
    Self: DivAssign,
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

/// ```
/// # use dsp_fixedpoint::Q8;
/// let q = Q8::<4>::new(7);
/// assert_eq!(format!("{q} {q:e} {q:E}"), "0.4375 4.375e-1 4.375E-1");
/// ```
macro_rules! impl_fmt {
    ($tr:path) => {
        impl<T: Int, A, const F: i8> $tr for Q<T, A, F>
        where
            Self: Copy + Into<f32> + Into<f64>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if const { T::BITS <= f32::MANTISSA_DIGITS } {
                    <f32 as $tr>::fmt(&(*self).into(), f)
                } else {
                    <f64 as $tr>::fmt(&(*self).into(), f)
                }
            }
        }
    };
}
impl_fmt!(fmt::Display);
impl_fmt!(fmt::UpperExp);
impl_fmt!(fmt::LowerExp);

/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(format!("{:?}", Q8::<4>::new(0x14)), "20");
/// assert_eq!(format!("{:b}", Q8::<4>::new(0x14)), "10100");
/// assert_eq!(format!("{:b}", Q8::<4>::new(-0x14)), "11101100");
/// ```
macro_rules! impl_dot_fmt {
    ($tr:path) => {
        impl<T: $tr, A, const F: i8> $tr for Q<T, A, F> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.inner.fmt(f)
            }
        }
    };
}
impl_dot_fmt!(fmt::Debug);
impl_dot_fmt!(fmt::Binary);
impl_dot_fmt!(fmt::Octal);
impl_dot_fmt!(fmt::UpperHex);
impl_dot_fmt!(fmt::LowerHex);

// TODO: dot format

macro_rules! impl_q {
    // Primitive
    ($alias:ident<$t:ty, $a:ty>) => {
        impl_q!($alias<$t, $a>, |x| x as _, <$t>::MIN, <$t>::MAX, <$t>::BITS);
    };
    // Newtype
    ($alias:ident<$t:ty, $a:ty>, $wrap:tt) => {
        impl_q!($alias<$wrap<$t>, $wrap<$a>>, |x: $wrap<_>| $wrap(x.0 as _), Self(<$t>::MIN), Self(<$t>::MAX), <$t>::BITS);
    };
    ($alias:ident<$t:ty, $a:ty>, $as:expr, $min:expr, $max:expr, $bits:expr) => {
        impl Int for $t {
            const BITS: u32 = $bits;
            const MIN: Self = $min;
            const MAX: Self = $max;
        }

        impl Accu<$a> for $t {
            #[inline(always)]
            fn up(self) -> $a {
                $as(self)
            }
            #[inline(always)]
            fn down(a: $a) -> Self {
                $as(a)
            }
        }

        #[doc = concat!("Fixed point [`", stringify!($t), "`]")]
        pub type $alias<const F: i8 = {<$t as Int>::BITS as _}> = Q<$t, $a, F>;

        /// Scale from integer base type
        impl<const F: i8> From<$t> for Q<$t, $a, F> {
            fn from(value: $t) -> Self {
                Self::new(value.shs(F))
            }
        }

        /// Scale from integer base type
        impl<const F: i8> From<Q<$t, $a, F>> for $t {
            fn from(value: Q<$t, $a, F>) -> Self {
                value.inner.shs(F)
            }
        }

        /// Scale from integer accu type
        impl<const F: i8> From<Q<$a, $t, F>> for $t {
            fn from(value: Q<$a, $t, F>) -> Self {
                <$t>::down(value.inner.shs(-F))
            }
        }

        /// T*Q -> T
        impl<const F: i8> Mul<Q<$t, $a, F>> for $t {
            type Output = $t;

            fn mul(self, rhs: Q<$t, $a, F>) -> Self::Output {
                <$t>::down((self.up() * rhs.inner.up()).shs(-F))
            }
        }

        /// T/Q -> T
        impl<const F: i8> Div<Q<$t, $a, F>> for $t {
            type Output = $t;

            fn div(self, rhs: Q<$t, $a, F>) -> Self::Output {
                if F > 0 {
                    <$t>::down(self.up().shs(F) / rhs.inner.up())
                } else {
                    self.shs(F) / rhs.inner
                }
            }
        }
    };
}
// Signed
// impl Int for i128 {
//     const BITS: u32 = Self::BITS;
//     const MIN: Self = Self::MIN;
//     const MAX: Self = Self::MAX;
// }
impl_q!(Q8<i8, i16>);
impl_q!(Q16<i16, i32>);
impl_q!(Q32<i32, i64>);
impl_q!(Q64<i64, i128>);
// Unsigned (_P_ositive)
// impl Int for u8 {
//     const BITS: u32 = Self::BITS;
//     const MIN: Self = Self::MIN;
//     const MAX: Self = Self::MAX;
// }
impl_q!(P8<u8, u16>);
impl_q!(P16<u16, u32>);
impl_q!(P32<u32, u64>);
impl_q!(P64<u64, u128>);
// _W_rapping signed
// impl Int for Wrapping<i8> {
//     const BITS: u32 = i8::BITS;
//     const MIN: Self = Self(i8::MIN);
//     const MAX: Self = Self(i8::MAX);
// }
impl_q!(W8<i8, i16>, Wrapping);
impl_q!(W16<i16, i32>, Wrapping);
impl_q!(W32<i32, i64>, Wrapping);
impl_q!(W64<i64, i128>, Wrapping);
// Wrapping vnsigned
// impl Int for Wrapping<u8> {
//     const BITS: u32 = u8::BITS;
//     const MIN: Self = Self(u8::MIN);
//     const MAX: Self = Self(u8::MAX);
// }
impl_q!(V8<u8, u16>, Wrapping);
impl_q!(V16<u16, u32>, Wrapping);
impl_q!(V32<u32, u64>, Wrapping);
impl_q!(V64<u64, u128>, Wrapping);

// NonZero<T>, Saturating<T> don't implement Shr/Shl

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
