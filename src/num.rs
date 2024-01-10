use core::{
    iter::Sum,
    ops::{Add, Mul, Neg},
};
use num_traits::{AsPrimitive, Float};

/// Helper trait unifying fixed point and floating point coefficients/samples
pub trait FilterNum:
    'static + Copy + Neg<Output = Self> + Add<Self, Output = Self> + Sum<Self> + AsPrimitive<Self::ACCU>
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
    /// Accumulator type
    type ACCU: AsPrimitive<Self>
        + Add<Self::ACCU, Output = Self::ACCU>
        + Mul<Self::ACCU, Output = Self::ACCU>
        + Sum<Self::ACCU>;

    /// Proper scaling and potentially using a wide accumulator.
    /// Clamp `self` such that `min <= self <= max`.
    /// Undefined result if `max < min`.
    fn macc(
        xa: impl Iterator<Item = (Self, Self)>,
        u: Self,
        e1: Self,
        min: Self,
        max: Self,
    ) -> (Self, Self);

    fn clip(self, min: Self, max: Self) -> Self;

    /// Multiplication (scaled)
    fn mul(self, other: Self) -> Self;

    /// Division (scaled)
    fn div(self, other: Self) -> Self;

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
            const NEG_ONE: Self = -1.0;
            const ZERO: Self = 0.0;
            const MIN: Self = <$T>::NEG_INFINITY;
            const MAX: Self = <$T>::INFINITY;
            type ACCU = Self;

            fn macc(
                xa: impl Iterator<Item = (Self, Self)>,
                u: Self,
                _e1: Self,
                min: Self,
                max: Self,
            ) -> (Self, Self) {
                (xa.fold(u, |u, (x, a)| u + x * a).clip(min, max), 0.0)
            }

            fn clip(self, min: Self, max: Self) -> Self {
                // <$T>::clamp() is slow and checks
                // this calls fminf/fmaxf
                // self.max(min).min(max)
                if self < min {
                    min
                } else if self > max {
                    max
                } else {
                    self
                }
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
    ($T:ty, $U:ty, $A:ty, $Q:literal) => {
        impl FilterNum for $T {
            const ONE: Self = 1 << $Q;
            const NEG_ONE: Self = -1 << $Q;
            const ZERO: Self = 0;
            // Need to avoid `$T::MIN*$T::MIN` overflow.
            const MIN: Self = -<$T>::MAX;
            const MAX: Self = <$T>::MAX;
            type ACCU = $A;

            fn macc(
                xa: impl Iterator<Item = (Self, Self)>,
                u: Self,
                e1: Self,
                min: Self,
                max: Self,
            ) -> (Self, Self) {
                const S: usize = core::mem::size_of::<$T>() * 8;
                // Guard bits
                const G: usize = S - $Q;
                // Combine offset (u << $Q) with previous quantization error e1
                let s0 = (((u >> G) as $A) << S) | (((u << $Q) | e1) as $U as $A);
                let s = xa.fold(s0, |s, (x, a)| s + x as $A * a as $A);
                let sh = (s >> S) as $T;
                // Ord::clamp() is slow and checks
                // This clamping truncates the lowest G bits of the value and the limits.
                let y = if sh < min >> G {
                    min
                } else if sh > max >> G {
                    max
                } else {
                    (s >> $Q) as $T
                };
                // Quantization error
                let e = (s & ((1 << $Q) - 1)) as $T;
                (y, e)
            }

            fn clip(self, min: Self, max: Self) -> Self {
                // Ord::clamp() is slow and checks
                if self < min {
                    min
                } else if self > max {
                    max
                } else {
                    self
                }
            }

            fn div(self, other: Self) -> Self {
                (((self as $A) << $Q) / other as $A) as $T
            }

            fn mul(self, other: Self) -> Self {
                (((1 << ($Q - 1)) + self as $A * other as $A) >> $Q) as $T
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
// This is necessary to meet a1 = -2
// It also create 2 guard bits for clamping in the accumulator which is often enough.
impl_int!(i8, u8, i16, 6);
impl_int!(i16, u16, i32, 14);
impl_int!(i32, u32, i64, 30);
impl_int!(i64, u64, i128, 62);
