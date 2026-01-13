use dsp_fixedpoint::{Q, Shift};
use num_traits::{ConstOne, ConstZero, clamp};

/// Constants
pub trait Clamp: ConstOne + ConstZero + PartialOrd {
    /// Lowest value
    ///
    /// Negative infinity for floating point values.
    ///
    /// ```
    /// # use idsp::Clamp;
    /// assert_eq!(<i8 as Clamp>::MIN, -128);
    /// assert_eq!(<f32 as Clamp>::MIN, f32::NEG_INFINITY);
    /// ```
    const MIN: Self;

    /// Highest value
    ///
    /// Positive infinity for floating point values.
    ///
    /// ```
    /// # use idsp::Clamp;
    /// assert_eq!(<i8 as Clamp>::MAX, 127);
    /// ```
    const MAX: Self;

    /// Clamp
    fn clamp(self, min: Self, max: Self) -> Self {
        clamp(self, min, max)
    }
}

macro_rules! impl_const_float {
    ($ty:ident) => {
        impl Clamp for $ty {
            const MIN: Self = <$ty>::NEG_INFINITY;
            const MAX: Self = <$ty>::INFINITY;
        }
    };
}
impl_const_float!(f32);
impl_const_float!(f64);

macro_rules! impl_foreign {
    ($($ty:ident),*) => {$(
        impl Clamp for $ty {
            const MIN: Self = <$ty>::MIN;
            const MAX: Self = <$ty>::MAX;
        }
    )*};
}
impl_foreign!(
    i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, isize, usize
);

impl<T: Clamp + Shift + Copy + PartialOrd, A, const F: i8> Clamp for Q<T, A, F>
where
    Self: ConstOne,
{
    /// Lowest value
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// # use idsp::Clamp;
    /// # use num_traits::AsPrimitive;
    /// assert_eq!(Q8::<4>::MIN, (-16f32).as_());
    /// ```
    const MIN: Self = Self::new(T::MIN);

    /// Highest value
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// # use idsp::Clamp;
    /// # use num_traits::AsPrimitive;
    /// assert_eq!(Q8::<4>::MAX, (16.0f32 - 1.0 / 16.0).as_());
    /// ```
    const MAX: Self = Self::new(T::MAX);
}
