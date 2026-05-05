use core::num::Wrapping;
use core::ops::{Add, Mul};
use num_traits::{
    AsPrimitive, Bounded, ConstZero, FromPrimitive, Num, One, Signed, ToPrimitive, Zero,
};

#[cfg(not(any(feature = "std")))]
#[allow(unused_imports)]
use num_traits::Float; // .round()

use crate::{Accu, AsFloat, Q, Shift};

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
        Self::new(T::one().shsc::<F>())
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

macro_rules! impl_as_float {
    ($ty:ident) => {
        impl<T, A, const F: i8> AsPrimitive<Q<T, A, F>> for $ty
        where
            $ty: AsPrimitive<T>,
            T: 'static + Copy,
            A: 'static,
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

        impl<T, A, const F: i8> AsPrimitive<$ty> for Q<T, A, F>
        where
            T: AsPrimitive<$ty>,
            A: 'static,
        {
            #[inline]
            fn as_(self) -> $ty {
                self.inner.as_() * Self::DELTA as $ty
            }
        }
    };
}

impl_as_float!(f32);
impl_as_float!(f64);

impl<T, A, const F: i8> AsPrimitive<Self> for Q<T, A, F>
where
    Self: Copy + 'static,
{
    #[inline]
    fn as_(self) -> Self {
        self
    }
}

macro_rules! impl_accu_as_primitive {
    ($($t:ty => $a:ty),* $(,)?) => {
        $(
            impl<const F: i8> AsPrimitive<$t> for Q<$a, $t, F> {
                #[inline]
                fn as_(self) -> $t {
                    self.quantize()
                }
            }
        )*
    };
}

impl_accu_as_primitive!(
    i8 => i16,
    i16 => i32,
    i32 => i64,
    i64 => i128,
    u8 => u16,
    u16 => u32,
    u32 => u64,
    u64 => u128,
    Wrapping<i8> => Wrapping<i16>,
    Wrapping<i16> => Wrapping<i32>,
    Wrapping<i32> => Wrapping<i64>,
    Wrapping<i64> => Wrapping<i128>,
    Wrapping<u8> => Wrapping<u16>,
    Wrapping<u16> => Wrapping<u32>,
    Wrapping<u32> => Wrapping<u64>,
    Wrapping<u64> => Wrapping<u128>,
);

impl<T: Bounded, A, const F: i8> Bounded for Q<T, A, F> {
    #[inline]
    fn min_value() -> Self {
        Self::new(T::min_value())
    }

    #[inline]
    fn max_value() -> Self {
        Self::new(T::max_value())
    }
}

impl<T, A, const F: i8> Num for Q<T, A, F>
where
    T: Num + Shift + Accu<A> + Copy + core::ops::Div<T, Output = T>,
    A: Shift + Copy + core::ops::Div<A, Output = A>,
    Self: One + Zero,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(str, radix).map(Self::from_int)
    }
}

impl<T: Shift + ToPrimitive + AsFloat, A, const F: i8> ToPrimitive for Q<T, A, F> {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.trunc().to_i64()
    }

    #[inline]
    fn to_i128(&self) -> Option<i128> {
        self.trunc().to_i128()
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.trunc().to_u64()
    }

    #[inline]
    fn to_u128(&self) -> Option<u128> {
        self.trunc().to_u128()
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        Some((*self).as_f32())
    }

    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some((*self).as_f64())
    }
}

impl<T, A, const F: i8> FromPrimitive for Q<T, A, F>
where
    T: 'static + Copy + FromPrimitive + Shift,
    A: 'static,
    f32: AsPrimitive<Q<T, A, F>>,
    f64: AsPrimitive<Q<T, A, F>>,
    Self: Copy + 'static,
{
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        T::from_i64(n).map(Self::from_int)
    }

    #[inline]
    fn from_i128(n: i128) -> Option<Self> {
        T::from_i128(n).map(Self::from_int)
    }

    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        T::from_u64(n).map(Self::from_int)
    }

    #[inline]
    fn from_u128(n: u128) -> Option<Self> {
        T::from_u128(n).map(Self::from_int)
    }

    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        Some(Self::from_f32(n))
    }

    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::from_f64(n))
    }
}

macro_rules! impl_signed_q {
    ($($ty:ty),* $(,)?) => {
        $(
            impl<A, const F: i8> Signed for Q<$ty, A, F>
            where
                Self: Num + core::ops::Neg<Output = Self>,
            {
                #[inline]
                fn abs(&self) -> Self {
                    Self::new(self.inner.abs())
                }

                #[inline]
                fn abs_sub(&self, other: &Self) -> Self {
                    Self::new(self.inner.abs_sub(&other.inner))
                }

                #[inline]
                fn signum(&self) -> Self {
                    match Signed::signum(&self.inner) {
                        1 => Self::one(),
                        -1 => -Self::one(),
                        _ => Self::zero(),
                    }
                }

                #[inline]
                fn is_positive(&self) -> bool {
                    self.inner > 0
                }

                #[inline]
                fn is_negative(&self) -> bool {
                    self.inner < 0
                }
            }

            impl<A, const F: i8> Signed for Q<Wrapping<$ty>, A, F>
            where
                Self: Num + core::ops::Neg<Output = Self>,
            {
                #[inline]
                fn abs(&self) -> Self {
                    Self::new(Wrapping(self.inner.0.abs()))
                }

                #[inline]
                fn abs_sub(&self, other: &Self) -> Self {
                    Self::new(self.inner.abs_sub(&other.inner))
                }

                #[inline]
                fn signum(&self) -> Self {
                    match Signed::signum(&self.inner) {
                        Wrapping(1) => Self::one(),
                        Wrapping(-1) => -Self::one(),
                        Wrapping(_) => Self::zero(),
                    }
                }

                #[inline]
                fn is_positive(&self) -> bool {
                    self.inner.0.is_positive()
                }

                #[inline]
                fn is_negative(&self) -> bool {
                    self.inner.0.is_negative()
                }
            }
        )*
    };
}

impl_signed_q!(i8, i16, i32, i64);
