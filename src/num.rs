use dsp_fixedpoint::{Q, Shift};
use num_traits::{ConstOne, ConstZero};

/// Constants
pub trait Const: ConstOne + ConstZero {
    /// Lowest value
    ///
    /// Negative infinity for floating point values.
    ///
    /// ```
    /// # use idsp::Const;
    /// assert_eq!(<i8 as Const>::MIN, -128);
    /// assert_eq!(<f32 as Const>::MIN, f32::NEG_INFINITY);
    /// ```
    const MIN: Self;

    /// Highest value
    ///
    /// Positive infinity for floating point values.
    ///
    /// ```
    /// # use idsp::Const;
    /// assert_eq!(<i8 as Const>::MAX, 127);
    /// ```
    const MAX: Self;
}

macro_rules! impl_const_float {
    ($ty:ident) => {
        impl Const for $ty {
            const MIN: Self = <$ty>::NEG_INFINITY;
            const MAX: Self = <$ty>::INFINITY;
        }
    };
}
impl_const_float!(f32);
impl_const_float!(f64);

impl<T: Const + Shift + Copy, A, const F: i8> Const for Q<T, A, F>
where
    Self: ConstOne,
{
    /// Lowest value
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// # use idsp::Const;
    /// # use num_traits::AsPrimitive;
    /// assert_eq!(Q8::<4>::MIN, (-16f32).as_());
    /// ```
    const MIN: Self = Self::new(T::MIN);

    /// Highest value
    ///
    /// ```
    /// # use dsp_fixedpoint::Q8;
    /// # use idsp::Const;
    /// # use num_traits::AsPrimitive;
    /// assert_eq!(Q8::<4>::MAX, (16.0f32 - 1.0 / 16.0).as_());
    /// ```
    const MAX: Self = Self::new(T::MAX);
}

macro_rules! impl_foreign {
    ($($ty:ident),*) => {$(
        impl Const for $ty {
            const MIN: Self = <$ty>::MIN;
            const MAX: Self = <$ty>::MAX;
        }
    )*};
}
impl_foreign!(
    i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, isize, usize
);
