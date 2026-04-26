use core::{
    iter,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign},
    ops::{Neg, Not, Shl, ShlAssign, Shr, ShrAssign},
};

use crate::{Accu, Q, Shift};

macro_rules! forward_unop {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<Output = T>, A, const F: i8> $tr for Q<T, A, F> {
            type Output = Self;
            #[inline]
            fn $m(self) -> Self::Output {
                Self::new(<T as $tr>::$m(self.inner))
            }
        }
    };
}
forward_unop!(Neg::neg);
forward_unop!(Not::not);

macro_rules! forward_sh_op {
    ($tr:ident::$m:ident) => {
        impl<U, T: $tr<U, Output = T>, A, const F: i8> $tr<U> for Q<T, A, F> {
            type Output = Self;
            #[inline]
            fn $m(self, rhs: U) -> Self::Output {
                Self::new(<T as $tr<U>>::$m(self.inner, rhs))
            }
        }
    };
}
forward_sh_op!(Shr::shr);
forward_sh_op!(Shl::shl);

macro_rules! forward_sh_assign_op {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<U>, U, A, const F: i8> $tr<U> for Q<T, A, F> {
            #[inline]
            fn $m(&mut self, rhs: U) {
                <T as $tr<U>>::$m(&mut self.inner, rhs)
            }
        }
    };
}
forward_sh_assign_op!(ShrAssign::shr_assign);
forward_sh_assign_op!(ShlAssign::shl_assign);

/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(
///     Q8::<3>::from_f32(3.5) + Q8::from_f32(5.2),
///     Q8::from_f32(8.7)
/// );
/// assert_eq!(
///     Q8::<3>::from_f32(4.0) - Q8::from_f32(3.2),
///     Q8::from_f32(0.8)
/// );
/// assert_eq!(Q8::<3>::from_f32(3.5) % Q8::from_int(1), Q8::from_f32(0.5));
/// ```
macro_rules! forward_binop {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T, Output = T>, A, const F: i8> $tr for Q<T, A, F> {
            type Output = Self;
            #[inline]
            fn $m(self, rhs: Self) -> Self::Output {
                Self::new(<T as $tr>::$m(self.inner, rhs.inner))
            }
        }
    };
}
forward_binop!(Rem::rem);
forward_binop!(Add::add);
forward_binop!(Sub::sub);
forward_binop!(BitAnd::bitand);
forward_binop!(BitOr::bitor);
forward_binop!(BitXor::bitxor);

// The notable exception to standard rules
// (https://github.com/rust-lang/rust/pull/93208#issuecomment-1019310634)
// This is for performance reasons
//
// Q*T -> A, Q/T -> Q
// See also the T*Q -> T and T/Q -> T in impl_q!()
//

/// Wide multiplication to accumulator
///
/// ```
/// # use dsp_fixedpoint::{Q, Q8};
/// assert_eq!(Q8::<3>::new(4) * 2, Q::new(8));
/// assert_eq!(Q8::<3>::new(4) / 2, Q8::new(2));
/// ```
impl<T: Accu<A>, A: Mul<Output = A>, const F: i8> Mul<T> for Q<T, A, F> {
    type Output = Q<A, T, F>;
    #[inline]
    fn mul(self, rhs: T) -> Q<A, T, F> {
        Q::new(self.inner.up() * rhs.up())
    }
}

impl<T: Div<Output = T>, A, const F: i8> Div<T> for Q<T, A, F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self {
        Q::new(self.inner / rhs)
    }
}

macro_rules! forward_assign_op_foreign {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T>, A, const F: i8> $tr<T> for Q<T, A, F> {
            #[inline]
            fn $m(&mut self, rhs: T) {
                <T as $tr>::$m(&mut self.inner, rhs)
            }
        }
    };
}
forward_assign_op_foreign!(MulAssign::mul_assign);
forward_assign_op_foreign!(DivAssign::div_assign);

macro_rules! forward_assign_op {
    ($tr:ident::$m:ident) => {
        impl<T: $tr<T>, A, const F: i8> $tr for Q<T, A, F> {
            #[inline]
            fn $m(&mut self, rhs: Self) {
                <T as $tr>::$m(&mut self.inner, rhs.inner)
            }
        }
    };
}
forward_assign_op!(RemAssign::rem_assign);
forward_assign_op!(AddAssign::add_assign);
forward_assign_op!(SubAssign::sub_assign);
forward_assign_op!(BitAndAssign::bitand_assign);
forward_assign_op!(BitOrAssign::bitor_assign);
forward_assign_op!(BitXorAssign::bitxor_assign);

/// Q *= Q'
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// let mut q = Q8::<4>::from_f32(0.25);
/// q *= Q8::<3>::from_int(3);
/// assert_eq!(q, Q8::from_f32(0.75));
/// ```
impl<T: Copy + Accu<A>, A: Shift + Mul<A, Output = A>, const F: i8, const F1: i8>
    MulAssign<Q<T, A, F1>> for Q<T, A, F>
{
    #[inline]
    fn mul_assign(&mut self, rhs: Q<T, A, F1>) {
        let shift = const { -F1 };
        self.inner = T::down((self.inner.up() * rhs.inner.up()).shs(shift));
    }
}

/// Q /= Q'
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// let mut q = Q8::<4>::from_f32(0.75);
/// q /= Q8::<3>::from_int(3);
/// assert_eq!(q, Q8::from_f32(0.25));
/// ```
impl<
    T: Copy + Shift + Accu<A> + Div<T, Output = T>,
    A: Shift + Div<A, Output = A>,
    const F: i8,
    const F1: i8,
> DivAssign<Q<T, A, F1>> for Q<T, A, F>
{
    #[inline]
    fn div_assign(&mut self, rhs: Q<T, A, F1>) {
        self.inner = if F1 > 0 {
            T::down(self.inner.up().shs(F1) / rhs.inner.up())
        } else {
            self.inner.shs(F1) / rhs.inner
        };
    }
}

/// Q*Q -> Q
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(
///     Q8::<4>::from_f32(0.75) * Q8::from_int(3),
///     Q8::from_f32(2.25)
/// );
/// ```
impl<T, A, const F: i8> Mul for Q<T, A, F>
where
    Self: MulAssign,
{
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

/// Q/Q -> Q
///
/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(Q8::<4>::from_int(3) / Q8::from_int(2), Q8::from_f32(1.5));
/// ```
impl<T, A, const F: i8> Div for Q<T, A, F>
where
    Self: DivAssign,
{
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<T: iter::Sum, A, const F: i8> iter::Sum for Q<T, A, F> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::new(iter.map(|i| i.inner).sum())
    }
}
