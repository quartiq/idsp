#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

mod format;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::FloatCore;
use num_traits::{AsPrimitive, ConstOne, ConstZero, One, Zero};

use core::{
    iter,
    marker::PhantomData,
    num::Wrapping,
    ops::{Add, Div, Mul, Neg, Rem, Shl, Shr, Sub},
    ops::{AddAssign, DivAssign, MulAssign, RemAssign, ShlAssign, ShrAssign, SubAssign},
    ops::{BitAnd, BitOr, BitXor, Not},
    ops::{BitAndAssign, BitOrAssign, BitXorAssign},
};

const fn frac_shift(from: i8, to: i8) -> i8 {
    let shift = to as i16 - from as i16;
    assert!(
        shift >= i8::MIN as i16 && shift <= i8::MAX as i16,
        "fractional bit delta must fit in i8"
    );
    shift as i8
}

pub(crate) trait FloatValue: Copy {
    #[cfg(feature = "defmt")]
    fn as_f32(self) -> f32;
    fn as_f64(self) -> f64;
}

macro_rules! impl_float_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl FloatValue for $ty {
                #[cfg(feature = "defmt")]
                #[inline]
                fn as_f32(self) -> f32 {
                    self as f32
                }

                #[inline]
                fn as_f64(self) -> f64 {
                    self as f64
                }
            }

            impl FloatValue for Wrapping<$ty> {
                #[cfg(feature = "defmt")]
                #[inline]
                fn as_f32(self) -> f32 {
                    self.0 as f32
                }

                #[inline]
                fn as_f64(self) -> f64 {
                    self.0 as f64
                }
            }
        )*
    };
}

impl_float_value!(i8, i16, i32, i64, u8, u16, u32, u64);

const fn neg_shift(frac: i8) -> i8 {
    let shift = -(frac as i16);
    assert!(
        shift >= i8::MIN as i16 && shift <= i8::MAX as i16,
        "fractional bit count must be negatable"
    );
    shift as i8
}

const fn shift_magnitude(frac: i8) -> u32 {
    assert!(frac != i8::MIN, "fractional bit count must not be i8::MIN");
    if frac >= 0 {
        frac as u32
    } else {
        (-(frac as i16)) as u32
    }
}

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
            self >> (-(f as i16) as usize)
        }
    }
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
/// assert_eq!(Q8::<4>::from_int(3), Q8::new(3 << 4));
/// assert_eq!(7 * Q8::<4>::from_f32(1.5), 10);
/// assert_eq!(7 / Q8::<4>::from_f32(1.5), 4);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", serde(transparent))]
#[must_use = "fixed-point values are immutable and have no effect unless used"]
pub struct Q<T, A, const F: i8> {
    /// The accumulator type
    _accu: PhantomData<A>,
    /// The inner value representation
    pub inner: T,
}

impl<T: Clone, A, const F: i8> Clone for Q<T, A, F> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            _accu: PhantomData,
            inner: self.inner.clone(),
        }
    }
}

impl<T: Copy, A, const F: i8> Copy for Q<T, A, F> {}

impl<T: PartialEq, A, const F: i8> PartialEq for Q<T, A, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl<T: Eq, A, const F: i8> Eq for Q<T, A, F> where Self: PartialEq {}

impl<T: PartialOrd, A, const F: i8> PartialOrd for Q<T, A, F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

impl<T: Ord, A, const F: i8> Ord for Q<T, A, F>
where
    Self: PartialOrd,
{
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.inner.cmp(&other.inner)
    }
}

impl<T: One + Shift, A, const F: i8> One for Q<T, A, F>
where
    Self: Mul<Output = Self>,
{
    fn one() -> Self {
        const {
            assert!(
                F >= 0,
                "`Q::one()` is only available when 1 is exactly representable"
            );
        }
        Self::new(T::one().shs(F))
    }
}

impl<T: Zero, A, const F: i8> Zero for Q<T, A, F>
where
    Self: Add<Output = Self>,
{
    fn zero() -> Self {
        Self::new(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }
}

impl<T: ConstZero, A, const F: i8> ConstZero for Q<T, A, F> {
    const ZERO: Self = Self::new(T::ZERO);
}

impl<T, A, const F: i8> Q<T, A, F> {
    /// Step between distinct numbers
    ///
    /// ```
    /// # use dsp_fixedpoint::Q32;
    /// assert_eq!(Q32::<31>::DELTA, 2f32.powi(-31));
    /// assert_eq!(Q32::<-4>::DELTA, 2f32.powi(4));
    /// ```
    ///
    /// ```compile_fail
    /// # use dsp_fixedpoint::Q32;
    /// let _ = Q32::<-128>::DELTA;
    /// ```
    pub const DELTA: f32 = if F > 0 {
        1.0 / (1u128 << shift_magnitude(F)) as f32
    } else {
        (1u128 << shift_magnitude(F)) as f32
    };

    /// Create a new fixed point number from a raw representation.
    ///
    /// This is equivalent to [`Q::from_bits`]. Prefer [`Q::from_int`],
    /// [`Q::from_f32`], or [`Q::from_f64`] when constructing from numeric values.
    ///
    /// ```
    /// # use dsp_fixedpoint::P8;
    /// assert_eq!(P8::<9>::new(3).inner, 3);
    /// ```
    #[inline]
    pub const fn new(inner: T) -> Self {
        Self {
            _accu: PhantomData,
            inner,
        }
    }

    /// Create a new fixed point number from a raw representation.
    #[inline]
    pub const fn from_bits(bits: T) -> Self {
        Self::new(bits)
    }

    /// Return the raw representation.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: Shift, A, const F: i8> Q<T, A, F> {
    /// Convert to a different number of fractional bits (truncating)
    ///
    /// Use this liberally for Add/Sub/Rem with Q's of different F.
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// assert_eq!(Q8::<4>::new(32).scale::<0>(), Q8::new(2));
    /// ```
    #[inline]
    pub fn scale<const F1: i8>(self) -> Q<T, A, F1> {
        let shift = const { frac_shift(F, F1) };
        Q::new(self.inner.shs(shift))
    }

    /// Return the integer part
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// assert_eq!(Q8::<4>::new(0x35).trunc(), 0x3);
    /// ```
    #[inline]
    #[must_use]
    pub fn trunc(self) -> T {
        let shift = const { neg_shift(F) };
        self.inner.shs(shift)
    }

    /// Scale from integer base type
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// assert_eq!(Q8::<4>::from_int(7).inner, 7 << 4);
    /// ```
    #[inline]
    pub fn from_int(value: T) -> Self {
        const {
            assert!(
                F != i8::MIN,
                "fractional bit count must not be i8::MIN for scaled construction"
            );
        }
        Self::new(value.shs(F))
    }
}

impl<A: Shift, T: Accu<A>, const F: i8> Q<A, T, F> {
    /// Scale from integer accu type
    ///
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// let q = Q8::<4>::from_f32(0.25);
    /// assert_eq!((q * 7).quantize(), (7.0 * 0.25f32).floor() as _);
    /// ```
    #[inline]
    #[must_use]
    pub fn quantize(self) -> T {
        T::down(self.trunc())
    }
}

/// Lossy conversion from a dynamically scaled integer
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(Q8::<8>::from((1, 3)).inner, 1 << 5);
/// ```
impl<T: Accu<A> + Shift, A, const F: i8> From<(T, i8)> for Q<T, A, F> {
    fn from(value: (T, i8)) -> Self {
        Self::new(value.0.shs(frac_shift(value.1, F)))
    }
}

/// Lossless conversion into a dynamically scaled integer
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// let q: (i8, i8) = Q8::<8>::new(9).into();
/// assert_eq!(q, (9, 8));
/// ```
impl<T, A, const F: i8> From<Q<T, A, F>> for (T, i8) {
    fn from(value: Q<T, A, F>) -> Self {
        (value.inner, F)
    }
}

/// Lossy conversion to and from float
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(8 * Q8::<4>::from_f32(0.25), 2);
/// assert_eq!(8 * Q8::<4>::from_f64(0.25), 2);
/// assert_eq!(Q8::<4>::new(4).as_f32(), 0.25);
/// assert_eq!(Q8::<4>::new(4).as_f64(), 0.25);
/// ```
macro_rules! impl_as_float {
    ($ty:ident) => {
        impl<T: 'static + Copy, A: 'static, const F: i8> AsPrimitive<Q<T, A, F>> for $ty
        where
            $ty: AsPrimitive<T>,
        {
            #[inline]
            fn as_(self) -> Q<T, A, F> {
                Q::new(
                    (self * const { 1.0 / Q::<T, A, F>::DELTA as $ty })
                        .round()
                        .as_(),
                )
            }
        }

        impl<T: AsPrimitive<$ty>, A: 'static, const F: i8> AsPrimitive<$ty> for Q<T, A, F> {
            #[inline]
            fn as_(self) -> $ty {
                self.inner.as_() * Self::DELTA as $ty
            }
        }
    };
}
impl_as_float!(f32);
impl_as_float!(f64);

impl<T, A, const F: i8> Q<T, A, F>
where
    f32: AsPrimitive<Q<T, A, F>>,
    Self: Copy + 'static,
{
    /// Quantize a f32
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        value.as_()
    }
}

impl<T, A, const F: i8> Q<T, A, F>
where
    f64: AsPrimitive<Q<T, A, F>>,
    Self: Copy + 'static,
{
    /// Quantize a f64
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        value.as_()
    }
}

macro_rules! impl_q_as_float {
    ($($ty:ty),* $(,)?) => {
        $(
            impl<A, const F: i8> Q<$ty, A, F> {
                /// Convert lossy to f32
                #[inline]
                #[must_use]
                pub fn as_f32(self) -> f32 {
                    self.inner as f32 * Self::DELTA
                }

                /// Convert lossy to f64
                #[inline]
                #[must_use]
                pub fn as_f64(self) -> f64 {
                    self.inner as f64 * Self::DELTA as f64
                }
            }
        )*
    };
}

macro_rules! impl_newtype_q_as_float {
    ($wrap:ident<$($ty:ty),* $(,)?>) => {
        $(
            impl<A, const F: i8> Q<$wrap<$ty>, A, F> {
                /// Convert lossy to f32
                #[inline]
                #[must_use]
                pub fn as_f32(self) -> f32 {
                    self.inner.0 as f32 * Self::DELTA
                }

                /// Convert lossy to f64
                #[inline]
                #[must_use]
                pub fn as_f64(self) -> f64 {
                    self.inner.0 as f64 * Self::DELTA as f64
                }
            }
        )*
    };
}

impl_q_as_float!(i8, i16, i32, i64, u8, u16, u32, u64);
impl_newtype_q_as_float!(Wrapping<i8, i16, i32, i64, u8, u16, u32, u64>);

impl<T, A, const F: i8> AsPrimitive<Self> for Q<T, A, F>
where
    Self: Copy + 'static,
{
    fn as_(self) -> Self {
        self
    }
}

macro_rules! forward_unop {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<Output = T>, A, const F: i8> $tr for Q<T, A, F> {
            type Output = Self;
            #[inline]
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
            #[inline]
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
            #[inline]
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
/// assert_eq!(
///     Q8::<3>::from_f32(3.5) + Q8::from_f32(5.2),
///     Q8::from_f32(8.7)
/// );
/// assert_eq!(
///     Q8::<3>::from_f32(4.0) - Q8::from_f32(3.2),
///     Q8::from_f32(0.8)
/// );
/// assert_eq!(Q8::<3>::from_f32(3.5) % Q8::from_int(1), Q8::from_f32(0.5));
/// ```
macro_rules! forward_binop {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T, Output = T>, A, const F: i8> $tr for Q<T, A, F> {
            type Output = Self;
            #[inline]
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

// The notable exception to standard rules
// (https://github.com/rust-lang/rust/pull/93208#issuecomment-1019310634)
// This is for performance reasons
//
// Q*T -> A, Q/T -> Q
// See also the T*Q -> T and T/Q -> T in impl_q!()
//

/// Wide multiplication to accumulator
///
/// ```
/// # use dsp_fixedpoint::{Q8, Q};
/// assert_eq!(Q8::<3>::new(4) * 2, Q::new(8));
/// assert_eq!(Q8::<3>::new(4) / 2, Q8::new(2));
/// ```
impl<T: Accu<A>, A: Mul<Output = A>, const F: i8> Mul<T> for Q<T, A, F> {
    type Output = Q<A, T, F>;
    #[inline]
    fn mul(self, rhs: T) -> Q<A, T, F> {
        Q::new(self.inner.up() * rhs.up())
    }
}

impl<T: Div<Output = T>, A, const F: i8> Div<T> for Q<T, A, F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        Q::new(self.inner / rhs)
    }
}

macro_rules! forward_assign_op_foreign {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T>, A, const F: i8> $tr<T> for Q<T, A, F> {
            #[inline]
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
            #[inline]
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
/// let mut q = Q8::<4>::from_f32(0.25);
/// q *= Q8::<3>::from_int(3);
/// assert_eq!(q, Q8::from_f32(0.75));
/// ```
impl<T: Copy + Accu<A>, A: Shift + Mul<A, Output = A>, const F: i8, const F1: i8>
    MulAssign<Q<T, A, F1>> for Q<T, A, F>
{
    #[inline]
    fn mul_assign(&mut self, rhs: Q<T, A, F1>) {
        let shift = const { neg_shift(F1) };
        self.inner = T::down((self.inner.up() * rhs.inner.up()).shs(shift));
    }
}

/// Q /= Q'
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// let mut q = Q8::<4>::from_f32(0.75);
/// q /= Q8::<3>::from_int(3);
/// assert_eq!(q, Q8::from_f32(0.25));
/// ```
impl<
    T: Copy + Shift + Accu<A> + Div<T, Output = T>,
    A: Shift + Div<A, Output = A>,
    const F: i8,
    const F1: i8,
> DivAssign<Q<T, A, F1>> for Q<T, A, F>
{
    #[inline]
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
/// assert_eq!(
///     Q8::<4>::from_f32(0.75) * Q8::from_int(3),
///     Q8::from_f32(2.25)
/// );
/// ```
impl<T, A, const F: i8> Mul for Q<T, A, F>
where
    Self: MulAssign,
{
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

/// Q/Q -> Q
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(Q8::<4>::from_int(3) / Q8::from_int(2), Q8::from_f32(1.5));
/// ```
impl<T, A, const F: i8> Div for Q<T, A, F>
where
    Self: DivAssign,
{
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<T: iter::Sum, A, const F: i8> iter::Sum for Q<T, A, F> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new(iter.map(|i| i.inner).sum())
    }
}

macro_rules! impl_q {
    // Primitive
    ($alias:ident<$t:ty, $a:ty>) => {
        impl_q!($alias<$t, $a>, $t, |x| x as _, core::convert::identity);
    };
    // Newtype
    ($alias:ident<$t:ty, $a:ty>, $wrap:tt) => {
        impl_q!($alias<$wrap<$t>, $wrap<$a>>, $t, |x: $wrap<_>| $wrap(x.0 as _), $wrap);
    };
    // Common
    ($alias:ident<$t:ty, $a:ty>, $inner:ty, $as:expr, $wrap:expr) => {
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

        #[doc = concat!("Fixed point [`", stringify!($t), "`] with [`", stringify!($a), "`] accumulator")]
        pub type $alias<const F: i8> = Q<$t, $a, F>;

        impl<const F: i8> ConstOne for Q<$t, $a, F> {
            const ONE: Self = {
                assert!(
                    F >= 0,
                    "`Q::ONE` is only available when 1 is exactly representable"
                );
                Self::new($wrap(1 << F as usize))
            };
        }

        impl<const F: i8> AsPrimitive<$t> for Q<$a, $t, F> {
            /// Scale from integer accu type
            #[inline]
            fn as_(self) -> $t {
                self.quantize()
            }
        }

        /// T*Q -> T
        impl<const F: i8> Mul<Q<$t, $a, F>> for $t {
            type Output = $t;

            #[inline]
            fn mul(self, rhs: Q<$t, $a, F>) -> Self::Output {
                (rhs * self).quantize()
            }
        }

        /// T/Q -> T
        impl<const F: i8> Div<Q<$t, $a, F>> for $t {
            type Output = $t;

            #[inline]
            fn div(self, rhs: Q<$t, $a, F>) -> Self::Output {
                const {
                    assert!(
                        F != i8::MIN,
                        "fractional bit count must not be i8::MIN for division"
                    );
                }
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
impl_q!(Q8<i8, i16>);
impl_q!(Q16<i16, i32>);
impl_q!(Q32<i32, i64>);
impl_q!(Q64<i64, i128>);
// Unsigned (_P_ositive)
impl_q!(P8<u8, u16>);
impl_q!(P16<u16, u32>);
impl_q!(P32<u32, u64>);
impl_q!(P64<u64, u128>);
// _W_rapping signed
impl_q!(W8<i8, i16>, Wrapping);
impl_q!(W16<i16, i32>, Wrapping);
impl_q!(W32<i32, i64>, Wrapping);
impl_q!(W64<i64, i128>, Wrapping);
// Wrapping vnsigned
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
        assert_eq!(
            Q32::<5>::from_int(4) * Q32::<5>::from_int(3),
            Q32::from_int(3 * 4)
        );
        assert_eq!(
            Q32::<5>::from_int(12) / Q32::<5>::from_int(6),
            Q32::from_int(2)
        );
        assert_eq!(7 * Q32::<4>::new(0x33), 7 * 3 + ((3 * 7) >> 4));
    }
}
