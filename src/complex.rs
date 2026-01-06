use super::{atan2, cossin};

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

macro_rules! fwd_binop {
    ($tr:ident, $meth:ident) => {
        impl<T: Copy + core::ops::$tr<Output = T>> core::ops::$tr for Complex<T> {
            type Output = Self;
            fn $meth(self, rhs: Self) -> Self {
                Self([self.0[0].$meth(rhs.0[0]), self.0[1].$meth(rhs.0[1])])
            }
        }
    };
}
fwd_binop!(Add, add);
fwd_binop!(Sub, sub);

macro_rules! fwd_binop_inner {
    ($tr:ident, $meth:ident) => {
        impl<T: Copy + core::ops::$tr<Output = T>> core::ops::$tr<T> for Complex<T> {
            type Output = Self;
            fn $meth(self, rhs: T) -> Self {
                Self([self.0[0].$meth(rhs), self.0[1].$meth(rhs)])
            }
        }
    };
}
fwd_binop_inner!(Mul, mul);
fwd_binop_inner!(Div, div);
fwd_binop_inner!(Rem, rem);

impl<T> core::iter::Sum for Complex<T>
where
    Self: Default + core::ops::Add<Output = Self>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Default::default(), |c, i| c + i)
    }
}

impl<T> core::iter::Product for Complex<T>
where
    Self: Default + core::ops::Mul<Output = Self>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Default::default(), |c, i| c * i)
    }
}

/// Complex extension trait offering DSP (fast, good accuracy) functionality.
pub trait ComplexExt<T, U> {
    /// Unit magnitude from angle
    fn from_angle(angle: T) -> Self;
    /// Square of magnitude
    fn norm_sqr(&self) -> U;
    /// Log2 approximation
    fn log2(&self) -> T;
    /// Angle
    fn arg(&self) -> T;
}

impl ComplexExt<i32, u32> for Complex<i32> {
    /// Return a Complex on the unit circle given an angle.
    ///
    /// Example:
    ///
    /// ```
    /// use idsp::{Complex, ComplexExt};
    /// Complex::<i32>::from_angle(0);
    /// Complex::<i32>::from_angle(1 << 30); // pi/2
    /// Complex::<i32>::from_angle(-1 << 30); // -pi/2
    /// ```
    fn from_angle(angle: i32) -> Self {
        let (c, s) = cossin(angle);
        Self::new(c, s)
    }

    /// Return the absolute square (the squared magnitude).
    ///
    /// Note: Normalization is `1 << 32`, i.e. U0.32.
    ///
    /// Note(panic): This will panic for `Complex(i32::MIN, i32::MIN)`
    ///
    /// Example:
    ///
    /// ```
    /// use idsp::{Complex, ComplexExt};
    /// assert_eq!(Complex::new(i32::MIN, 0).norm_sqr(), 1 << 31);
    /// assert_eq!(Complex::new(i32::MAX, i32::MAX).norm_sqr(), u32::MAX - 3);
    /// ```
    fn norm_sqr(&self) -> u32 {
        let [x, y] = self.0.map(|x| x as i64 * x as i64);
        ((x + y) >> 31) as _
    }

    /// log2(power) re full scale approximation
    ///
    /// TODO: scale up, interpolate
    ///
    /// Example:
    ///
    /// ```
    /// use idsp::{Complex, ComplexExt};
    /// assert_eq!(Complex::new(i32::MIN, i32::MIN).log2(), 0);
    /// assert_eq!(Complex::new(i32::MAX, i32::MAX).log2(), -1);
    /// assert_eq!(Complex::new(i32::MIN, 0).log2(), -1);
    /// assert_eq!(Complex::new(i32::MAX, 0).log2(), -2);
    /// assert_eq!(Complex::new(-1, 0).log2(), -63);
    /// assert_eq!(Complex::new(1, 0).log2(), -63);
    /// assert_eq!(Complex::new(0, 0).log2(), -64);
    /// ```
    fn log2(&self) -> i32 {
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
    /// use idsp::{Complex, ComplexExt};
    /// assert_eq!(Complex::new(0, 0).arg(), 0);
    /// ```
    fn arg(&self) -> i32 {
        atan2(self.im(), self.re())
    }
}

/// Full scale fixed point multiplication.
pub trait MulScaled<T> {
    /// Scaled multiplication for fixed point
    fn mul_scaled(self, other: T) -> Self;
}

impl MulScaled<Complex<i32>> for Complex<i32> {
    fn mul_scaled(self, other: Self) -> Self {
        let a = self.re() as i64;
        let b = self.im() as i64;
        let c = other.re() as i64;
        let d = other.im() as i64;
        Complex::new(
            ((a * c - b * d) >> 31) as i32,
            ((b * c + a * d) >> 31) as i32,
        )
    }
}

impl MulScaled<i32> for Complex<i32> {
    fn mul_scaled(self, other: i32) -> Self {
        Complex::new(
            ((other as i64 * self.re() as i64) >> 31) as i32,
            ((other as i64 * self.im() as i64) >> 31) as i32,
        )
    }
}

impl MulScaled<i16> for Complex<i32> {
    fn mul_scaled(self, other: i16) -> Self {
        Complex::new(
            (other as i32 * (self.re() >> 16) + (1 << 14)) >> 15,
            (other as i32 * (self.im() >> 16) + (1 << 14)) >> 15,
        )
    }
}
