#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

mod format;
mod num_traits_impl;
mod ops;

use num_traits::{AsPrimitive, ConstOne};

use core::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    num::Wrapping,
    ops::{Div, Mul, Shl, Shr},
};

/// Helper trait to unify over missing impl AsPrimitive<f*> for Wrapping<T>
pub(crate) trait AsFloat: Copy {
    fn as_f32(self) -> f32;
    fn as_f64(self) -> f64;
}

macro_rules! impl_as_float {
    ($($ty:ty),* $(,)?) => {
        $(
            impl AsFloat for $ty {
                #[inline]
                fn as_f32(self) -> f32 {
                    self as f32
                }

                #[inline]
                fn as_f64(self) -> f64 {
                    self as f64
                }
            }

            impl AsFloat for Wrapping<$ty> {
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

impl_as_float!(i8, i16, i32, i64, u8, u16, u32, u64);

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

    /// Const signed shift
    #[inline(always)]
    fn shsc<const F: i8>(self) -> Self {
        const { assert!(F > i8::MIN, "shift must not be i8::MIN") }
        self.shs(F)
    }
}

impl<T: Copy + Shl<usize, Output = T> + Shr<usize, Output = T>> Shift for T {
    #[inline(always)]
    fn shs(self, f: i8) -> Self {
        debug_assert!(f > i8::MIN, "shift must not be i8::MIN");
        if f >= 0 {
            self << (f as _)
        } else {
            self >> (-f as _)
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
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck::Pod, bytemuck::TransparentWrapper, bytemuck::Zeroable),
    transparent(T)
)]
#[must_use]
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

impl<T: Hash, A, const F: i8> Hash for Q<T, A, F> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

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
        1.0 / (1u128 << F) as f32
    } else {
        (1u128 << -F) as f32
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
        Q::new(self.inner.shs(const { F1 - F }))
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
        self.inner.shs(const { -F })
    }

    /// Scale from integer base type
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// assert_eq!(Q8::<4>::from_int(7).inner, 7 << 4);
    /// ```
    #[inline]
    pub fn from_int(value: T) -> Self {
        Self::new(value.shsc::<F>())
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
        Self::new(value.0.shs(F - value.1))
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

impl<T, A, const F: i8> Q<T, A, F>
where
    f32: AsPrimitive<Q<T, A, F>>,
    Self: Copy + 'static,
{
    /// Quantize a f32
    #[inline]
    #[must_use]
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
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        value.as_()
    }
}

#[allow(private_bounds)]
impl<T: AsFloat, A, const F: i8> Q<T, A, F> {
    /// Convert lossy to f32
    #[inline]
    #[must_use]
    pub fn as_f32(self) -> f32 {
        self.inner.as_f32() * Self::DELTA
    }

    /// Convert lossy to f64
    #[inline]
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.inner.as_f64() * Self::DELTA as f64
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
                Self::new($wrap(1 << F as usize))
            };
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
                if F > 0 {
                    <$t>::down(self.up().shs(F) / rhs.inner.up())
                } else {
                    self.shsc::<F>() / rhs.inner
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
    use num_traits::{Bounded, FromPrimitive, Signed, ToPrimitive};

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

    #[test]
    fn numeric_traits() {
        assert_eq!(Q8::<4>::min_value().inner, i8::MIN);
        assert_eq!(Q8::<4>::max_value().inner, i8::MAX);
        assert_eq!(Q8::<4>::from_f32(1.5).to_i32(), Some(1));
        assert_eq!(Q8::<4>::from_f32(1.5).to_f64(), Some(1.5));
        assert_eq!(Q8::<4>::from_i32(3), Some(Q8::<4>::from_int(3)));
        assert_eq!(Q8::<4>::from_f32(-1.5).abs(), Q8::<4>::from_f32(1.5));
        assert_eq!(Q8::<4>::from_f32(-1.5).signum(), Q8::<4>::from_int(-1));
        assert!(Q8::<4>::from_f32(-1.5).is_negative());
    }

    #[cfg(feature = "bytemuck")]
    #[test]
    fn bytemuck_traits() {
        use bytemuck::TransparentWrapper;

        let q = Q8::<4>::from_int(3);
        assert_eq!(q.into_inner(), 48);
        assert_eq!(Q8::<4>::wrap(48i8), q);
        assert_eq!(*Q8::<4>::wrap_ref(&48i8), q);
    }
}
