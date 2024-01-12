use num_traits::{AsPrimitive, Float, Num};

/// Helper trait unifying fixed point and floating point coefficients/samples
pub trait FilterNum: 'static + Copy + Num + AsPrimitive<Self::ACCU> {
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
    type ACCU: AsPrimitive<Self> + Num;

    /// Proper scaling and potentially using a wide accumulator.
    /// Clamp `self` such that `min <= self <= max`.
    /// Undefined result if `max < min`.
    fn macc(self, s: Self::ACCU, min: Self, max: Self, e1: Self) -> (Self, Self);

    fn clip(self, min: Self, max: Self) -> Self;

    /// Multiplication (scaled)
    fn mul_scaled(self, other: Self) -> Self;

    /// Division (scaled)
    fn div_scaled(self, other: Self) -> Self;

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

            #[inline]
            fn macc(self, s: Self::ACCU, min: Self, max: Self, _e1: Self) -> (Self, Self) {
                ((self + s).clip(min, max), 0.0)
            }

            #[inline]
            fn clip(self, min: Self, max: Self) -> Self {
                // <$T>::clamp() is slow and checks
                self.max(min).min(max)
            }

            #[inline]
            fn div_scaled(self, other: Self) -> Self {
                self / other
            }

            #[inline]
            fn mul_scaled(self, other: Self) -> Self {
                self * other
            }

            #[inline]
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
            const MIN: Self = <$T>::MIN;
            const MAX: Self = <$T>::MAX;
            type ACCU = $A;

            #[inline]
            fn macc(self, mut s: Self::ACCU, min: Self, max: Self, e1: Self) -> (Self, Self) {
                const S: usize = core::mem::size_of::<$T>() * 8;
                // Guard bits
                const G: usize = S - $Q;
                // Combine offset (u << $Q) with previous quantization error e1
                s += (((self >> G) as $A) << S) | (((self << $Q) | e1) as $U as $A);
                // Ord::clamp() is slow and checks
                // This clamping truncates the lowest G bits of the value and the limits.
                debug_assert_eq!(min & ((1 << G) - 1), 0);
                debug_assert_eq!(max & ((1 << G) - 1), (1 << G) - 1);
                let y0 = if (s >> S) as $T < (min >> G) {
                    min
                } else if (s >> S) as $T > (max >> G) {
                    max
                } else {
                    (s >> $Q) as $T
                };
                // Quantization error
                let e0 = s as $T & ((1 << $Q) - 1);
                (y0, e0)
            }

            #[inline]
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

            #[inline]
            fn div_scaled(self, other: Self) -> Self {
                (((self as $A) << $Q) / other as $A) as $T
            }

            #[inline]
            fn mul_scaled(self, other: Self) -> Self {
                (((1 << ($Q - 1)) + self as $A * other as $A) >> $Q) as $T
            }

            #[inline]
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
