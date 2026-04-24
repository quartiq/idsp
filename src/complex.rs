use super::{atan2, cossin};
use core::num::Wrapping;
use core::ops::{Add, Deref, DerefMut, Mul, Sub};
use dsp_fixedpoint::{Accu, Q, Shift};
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

impl<T> Deref for Complex<T> {
    type Target = [T; 2];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Complex<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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

impl<T: Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T>> Mul for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self([
            self.0[0] * rhs.0[0] - self.0[1] * rhs.0[1],
            self.0[0] * rhs.0[1] + self.0[1] * rhs.0[0],
        ])
    }
}

impl<
    T: 'static + Copy,
    A: Copy + Add<Output = A> + Sub<Output = A> + AsPrimitive<T>,
    B,
    const F: i8,
> Mul<Complex<T>> for Complex<Q<T, B, F>>
where
    Q<T, B, F>: Mul<T, Output = A>,
{
    type Output = Complex<T>;

    fn mul(self, rhs: Complex<T>) -> Complex<T> {
        Complex([
            (self.0[0] * rhs.0[0] - self.0[1] * rhs.0[1]).as_(),
            (self.0[0] * rhs.0[1] + self.0[1] * rhs.0[0]).as_(),
        ])
    }
}

impl<
    T: 'static + Copy,
    A: Copy + Add<Output = A> + Sub<Output = A> + AsPrimitive<T>,
    B,
    const F: i8,
> Mul<Complex<Q<T, B, F>>> for Complex<T>
where
    T: Mul<Q<T, B, F>, Output = A>,
{
    type Output = Complex<T>;

    fn mul(self, rhs: Complex<Q<T, B, F>>) -> Complex<T> {
        Complex([
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

impl Complex<f32> {
    /// Return a unit complex number from a floating-point angle in radians.
    pub fn from_angle_rad(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s)
    }
}

impl Complex<f64> {
    /// Return a unit complex number from a floating-point angle in radians.
    pub fn from_angle_rad(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, s)
    }
}

impl Complex<i32> {
    /// Return the absolute square (the squared magnitude).
    ///
    /// Panics: on `Complex::new(i32::MIN, i32::MIN)`
    /// Example:
    ///
    /// ```
    /// use idsp::Complex;
    /// assert_eq!(Complex::new(i32::MIN, 0).norm_sqr(), 1 << 62);
    /// ```
    pub fn norm_sqr(&self) -> i64 {
        let [x, y] = self.0.map(|x| x as i64 * x as i64);
        x + y
    }

    /// Return the integer log base 2
    ///
    /// Panics: on [i32::MIN; 2]
    pub fn ilog2(&self) -> u32 {
        self.norm_sqr().ilog2()
    }

    /// Return a Complex on the unit circle given an angle.
    ///
    /// Example:
    ///
    /// ```
    /// use core::num::Wrapping as W;
    /// use idsp::Complex;
    /// Complex::<_>::from_angle(W(0));
    /// Complex::<_>::from_angle(W(1 << 30)); // pi/2
    /// Complex::<_>::from_angle(W(-1 << 30)); // -pi/2
    /// ```
    pub fn from_angle(angle: Wrapping<i32>) -> Self {
        let (c, s) = cossin(angle.0);
        Self::new(c, s)
    }

    /// Return the angle.
    ///
    /// Note: Normalization is `1 << 31 == pi`.
    ///
    /// Example:
    ///
    /// ```
    /// use core::num::Wrapping as W;
    /// use idsp::Complex;
    /// assert_eq!(Complex::new(0, 0).arg(), W(0));
    /// assert_eq!(Complex::new(0, 1).arg(), W(0x3fff_ffff));
    /// ```
    pub fn arg(&self) -> Wrapping<i32> {
        Wrapping(atan2(self.im(), self.re()))
    }
}

impl<A: Shift, T: Accu<A> + Copy, const F: i8> Complex<Q<A, T, F>> {
    /// Quantize each lane back to the base integer type.
    pub fn quantize(self) -> Complex<T> {
        Complex::new(self.re().quantize(), self.im().quantize())
    }
}

impl<T: Copy, A, const F: i8> Complex<Q<T, A, F>> {
    /// Return the raw fixed-point representation lane-wise.
    pub fn into_bits(self) -> Complex<T> {
        Complex::new(self.re().into_inner(), self.im().into_inner())
    }

    /// Convert from raw to fixedpoint
    pub fn from_bits(value: Complex<T>) -> Self {
        Self::new(Q::new(value.re()), Q::new(value.im()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use dsp_fixedpoint::Q32;

    #[test]
    fn fixedpoint_into_bits_exposes_raw_representation() {
        let x = Complex::new(Q32::<32>::new(0x1234_5678), Q32::<32>::new(-0x2345_6789));
        assert_eq!(x.into_bits(), Complex::new(0x1234_5678, -0x2345_6789));
    }

    #[test]
    fn fixedpoint_arg_matches_i32_arg() {
        let z = Complex::new(
            Q::<i64, i32, 32>::new(1 << 34),
            Q::<i64, i32, 32>::new(1 << 34),
        );
        assert_eq!(z.quantize().arg(), Complex::new(4, 4).arg());
    }
}
