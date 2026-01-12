use super::{atan2, cossin};
use core::num::Wrapping;
use core::ops::{Add, Mul, Sub};
use dsp_fixedpoint::{P32, Q32, W32};
use num_traits::AsPrimitive;

/// A complex number in cartesian coordinates
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
    bytemuck::Zeroable,
    bytemuck::Pod,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Complex<T>(
    /// Real and imaginary parts
    pub [T; 2],
);

impl<T: Copy> Complex<T> {
    /// Create a new `Complex<T>`
    pub const fn new(re: T, im: T) -> Self {
        Self([re, im])
    }

    /// The real part
    pub fn re(&self) -> T {
        self.0[0]
    }

    /// The imaginary part
    pub fn im(&self) -> T {
        self.0[1]
    }
}

impl<T: Copy + core::ops::Neg<Output = T>> Complex<T> {
    /// Conjugate
    pub fn conj(self) -> Self {
        Self([self.0[0], -self.0[1]])
    }
}

macro_rules! fwd_binop {
    ($tr:ident::$meth:ident) => {
        impl<T: Copy + core::ops::$tr<Output = T>> core::ops::$tr for Complex<T> {
            type Output = Self;
            fn $meth(self, rhs: Self) -> Self {
                Self([self.0[0].$meth(rhs.0[0]), self.0[1].$meth(rhs.0[1])])
            }
        }
    };
}
fwd_binop!(Add::add);
fwd_binop!(Sub::sub);
fwd_binop!(BitAnd::bitand);
fwd_binop!(BitOr::bitor);
fwd_binop!(BitXor::bitxor);

macro_rules! fwd_binop_inner {
    ($tr:ident::$meth:ident) => {
        impl<T: Copy + core::ops::$tr<Output = T>> core::ops::$tr<T> for Complex<T> {
            type Output = Self;
            fn $meth(self, rhs: T) -> Self {
                Self([self.0[0].$meth(rhs), self.0[1].$meth(rhs)])
            }
        }
    };
}
fwd_binop_inner!(Mul::mul);
fwd_binop_inner!(Div::div);
fwd_binop_inner!(Rem::rem);
fwd_binop_inner!(BitAnd::bitand);
fwd_binop_inner!(BitOr::bitor);
fwd_binop_inner!(BitXor::bitxor);

macro_rules! fwd_unop {
    ($tr:ident::$meth:ident) => {
        impl<T: Copy + core::ops::$tr<Output = T>> core::ops::$tr for Complex<T> {
            type Output = Self;
            fn $meth(self) -> Self {
                Self([self.0[0].$meth(), self.0[1].$meth()])
            }
        }
    };
}
fwd_unop!(Not::not);
fwd_unop!(Neg::neg);

impl<T: 'static + Copy + Mul<Output = A>, A: Add<Output = A> + Sub<Output = A> + AsPrimitive<T>> Mul
    for Complex<T>
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self([
            (self.0[0] * rhs.0[0] - self.0[1] * rhs.0[1]).as_(),
            (self.0[0] * rhs.0[1] + self.0[1] * rhs.0[0]).as_(),
        ])
    }
}

impl<T> core::iter::Sum for Complex<T>
where
    Self: Default + Add<Output = Self>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Default::default(), |c, i| c + i)
    }
}

impl<T> core::iter::Product for Complex<T>
where
    Self: Default + Mul<Output = Self>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Default::default(), |c, i| c * i)
    }
}

impl Complex<Q32<31>> {
    /// Return a Complex on the unit circle given an angle.
    ///
    /// Example:
    ///
    /// ```
    /// use core::num::Wrapping as W;
    /// use dsp_fixedpoint::W32;
    /// use idsp::Complex;
    /// Complex::<_>::from_angle(W32::new(W(0)));
    /// Complex::<_>::from_angle(W32::new(W(1 << 30))); // pi/2
    /// Complex::<_>::from_angle(W32::new(W(-1 << 30))); // -pi/2
    /// ```
    pub fn from_angle(angle: W32<32>) -> Self {
        let (c, s) = cossin(angle.inner.0);
        Self::new(Q32::new(c), Q32::new(s))
    }
}

impl Complex<i32> {
    /// Return the absolute square (the squared magnitude).
    ///
    /// Note(panic): This will panic for `Complex(i32::MIN, i32::MIN)`
    ///
    /// Example:
    ///
    /// ```
    /// use dsp_fixedpoint::{P32, Q32};
    /// use idsp::Complex;
    /// assert_eq!(Complex::new(i32::MIN, 0).norm_sqr(), P32::new(1 << 31));
    /// assert_eq!(
    ///     Complex::new(i32::MAX, i32::MAX).norm_sqr(),
    ///     P32::new(u32::MAX - 3)
    /// );
    /// assert_eq!(
    ///     Complex::new(i32::MIN, i32::MAX).norm_sqr(),
    ///     P32::new(u32::MAX - 1)
    /// );
    /// ```
    pub fn norm_sqr(&self) -> P32<31> {
        let [x, y] = self.0.map(|x| x as i64 * x as i64);
        P32::new(((x + y) >> 31) as _)
    }

    /// trunc(log2(power)) re full scale (approximation)
    ///
    /// TODO: scale up, interpolate
    ///
    /// Example:
    ///
    /// ```
    /// use idsp::Complex;
    /// assert_eq!(Complex::new(i32::MIN, i32::MIN).log2(), 0);
    /// assert_eq!(Complex::new(i32::MAX, i32::MAX).log2(), -1);
    /// assert_eq!(Complex::new(i32::MIN, 0).log2(), -1);
    /// assert_eq!(Complex::new(i32::MAX, 0).log2(), -2);
    /// assert_eq!(Complex::new(-1, 0).log2(), -63);
    /// assert_eq!(Complex::new(1, 0).log2(), -63);
    /// assert_eq!(Complex::new(0, 0).log2(), -64);
    /// ```
    pub fn log2(&self) -> i32 {
        let [x, y] = self.0.map(|x| x as i64 * x as i64);
        -(x.wrapping_add(y).leading_zeros() as i32)
    }

    /// Return the angle.
    ///
    /// Note: Normalization is `1 << 31 == pi`.
    ///
    /// Example:
    ///
    /// ```
    /// use core::num::Wrapping as W;
    /// use dsp_fixedpoint::W32;
    /// use idsp::Complex;
    /// assert_eq!(Complex::new(0, 0).arg(), W32::new(W(0)));
    /// assert_eq!(Complex::new(0, 1).arg(), W32::new(W((1 << 30) - 1)));
    /// ```
    pub fn arg(&self) -> W32<32> {
        W32::new(Wrapping(atan2(self.im(), self.re())))
    }
}
