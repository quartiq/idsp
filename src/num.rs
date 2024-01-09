use core::{
    iter::Sum,
    ops::{Add, Neg, Sub},
};
use num_traits::{AsPrimitive, Float};

/// Helper trait unifying fixed point and floating point coefficients/samples
pub trait FilterNum:
    Copy
    + PartialEq
    + Neg<Output = Self>
    + Sub<Self, Output = Self>
    + Add<Self, Output = Self>
    + Sum<Self>
where
    // + AsPrimitive<Self::ACCU>,
    Self: 'static,
{
    /// Multiplicative identity
    const ONE: Self;
    /// Negative multiplicative identity, equal to `-Self::ONE`.
    const NEG_ONE: Self;
    /// Additive identity
    const ZERO: Self;
    /// Lowest value
    const MIN: Self;
    /// Highest value
    const MAX: Self;
    // type ACCU: AsPrimitive<Self>;
    /// Multiply-accumulate `self + sum(x*a)`
    ///
    /// Proper scaling and potentially using a wide accumulator.
    fn macc(self, xa: impl Iterator<Item = (Self, Self)>) -> Self;
    /// Multiplication (scaled)
    fn mul(self, other: Self) -> Self;
    /// Division (scaled)
    fn div(self, other: Self) -> Self;
    /// Clamp `self` such that `min <= self <= max`.
    ///
    /// # Panic
    /// May panics in debug mode if `max < min`.
    fn clamp(self, min: Self, max: Self) -> Self;
    /// Scale and quantize a floating point value.
    fn quantize<C>(value: C) -> Self
    where
        Self: AsPrimitive<C>,
        C: Float + AsPrimitive<Self>;
}

macro_rules! impl_float {
    ($T:ty) => {
        impl FilterNum for $T {
            const ONE: Self = 1.0;
            const NEG_ONE: Self = -Self::ONE;
            const ZERO: Self = 0.0;
            const MIN: Self = Self::NEG_INFINITY;
            const MAX: Self = Self::INFINITY;
            fn macc(self, xa: impl Iterator<Item = (Self, Self)>) -> Self {
                xa.fold(self, |y, (a, x)| a.mul_add(x, y))
                // xa.fold(self, |y, (a, x)| y + a * x)
            }
            fn clamp(self, min: Self, max: Self) -> Self {
                <$T>::clamp(self, min, max)
            }
            fn div(self, other: Self) -> Self {
                self / other
            }
            fn mul(self, other: Self) -> Self {
                self * other
            }
            fn quantize<C: Float + AsPrimitive<Self>>(value: C) -> Self {
                value.as_()
            }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

macro_rules! impl_int {
    ($T:ty, $A:ty, $Q:literal) => {
        impl FilterNum for $T {
            const ONE: Self = 1 << $Q;
            const NEG_ONE: Self = -1 << $Q;
            const ZERO: Self = 0;
            // Need to avoid `$T::MIN*$T::MIN` overflow.
            const MIN: Self = -<$T>::MAX;
            const MAX: Self = <$T>::MAX;
            fn macc(self, xa: impl Iterator<Item = (Self, Self)>) -> Self {
                self + (xa.fold(1 << ($Q - 1), |y, (a, x)| y + a as $A * x as $A) >> $Q) as Self
            }
            fn clamp(self, min: Self, max: Self) -> Self {
                Ord::clamp(self, min, max)
            }
            fn div(self, other: Self) -> Self {
                (((self as $A) << $Q) / other as $A) as Self
            }
            fn mul(self, other: Self) -> Self {
                (((1 << ($Q - 1)) + self as $A * other as $A) >> $Q) as Self
            }
            fn quantize<C>(value: C) -> Self
            where
                Self: AsPrimitive<C>,
                C: Float + AsPrimitive<Self>,
            {
                (value * (1 << $Q).as_()).round().as_()
            }
        }
    };
}
// Q2.X chosen to be able to exactly and inclusively represent -2 as `-1 << X + 1`
impl_int!(i8, i16, 6);
impl_int!(i16, i32, 14);
impl_int!(i32, i64, 30);
impl_int!(i64, i128, 62);
