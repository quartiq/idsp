use core::{
    cmp::{Ordering, PartialOrd},
    ops::{BitAnd, Shr},
};
use num_traits::{
    Bounded, Signed,
    cast::AsPrimitive,
    identities::Zero,
    ops::wrapping::{WrappingAdd, WrappingSub},
};
use serde::{Deserialize, Serialize};

/// Subtract `y - x` with signed overflow.
///
/// This is very similar to `i32::overflowing_sub(y, x)` except that the
/// overflow indicator is not a boolean but the signum of the overflow.
/// Additionally it's typically faster.
///
/// Returns:
/// A tuple containg the (wrapped) difference `y - x` and the signum of the
/// overflow.
#[inline(always)]
pub fn overflowing_sub<T>(y: T, x: T) -> (T, Wrap)
where
    T: WrappingSub + Zero + PartialOrd,
{
    let delta = y.wrapping_sub(&x);
    let wrap = (delta >= T::zero()).cmp(&(y >= x)).into();
    (delta, wrap)
}

/// Combine high and low i32 into a single downscaled i32, saturating monotonically.
///
/// Args:
/// `lo`: LSB i32 to scale down by `shift` and range-extend with `hi`
/// `hi`: MSB i32 to scale up and extend `lo` with. Output will be clipped if
///     `hi` exceeds the output i32 range.
/// `shift`: Downscale `lo` by that many bits. Values from 1 to 32 inclusive
///     are valid.
pub fn saturating_scale(lo: i32, hi: i32, shift: u32) -> i32 {
    debug_assert!(shift > 0);
    debug_assert!(shift <= 32);
    let hi_range = -1 << (shift - 1);
    if hi <= hi_range {
        i32::MIN - hi_range
    } else if -hi <= hi_range {
        hi_range - i32::MIN
    } else {
        (lo >> shift) + (hi << (32 - shift))
    }
}

/// Overflow unwrapper.
///
/// This is unwrapping as in the phase and overflow unwrapping context, not
/// unwrapping as in the `Result`/`Option` context.
#[derive(Copy, Clone, Default, Deserialize, Serialize)]
pub struct Unwrapper<Q> {
    /// current output
    y: Q,
}

impl<Q> Unwrapper<Q>
where
    Q: 'static + WrappingAdd + Copy,
{
    /// Feed a new sample..
    ///
    /// Args:
    /// * `x`: New sample
    ///
    /// Returns:
    /// The (wrapped) difference `x - x_old`
    pub fn update<P>(&mut self, x: P) -> P
    where
        P: 'static + WrappingSub + Copy + AsPrimitive<Q>,
        Q: AsPrimitive<P>,
    {
        let dx = x.wrapping_sub(&self.y.as_());
        self.y = self.y.wrapping_add(&dx.as_());
        dx
    }

    /// The current number of wraps
    pub fn wraps<P, const S: u32>(&self) -> P
    where
        Q: AsPrimitive<P> + Shr<u32, Output = Q>,
        P: 'static + Copy + WrappingAdd + Signed + BitAnd<u32, Output = P>,
    {
        (self.y >> S)
            .as_()
            .wrapping_add(&((self.y >> (S - 1)).as_() & 1))
    }

    /// The current phase
    pub fn phase<P>(&self) -> P
    where
        P: 'static + Copy,
        Q: AsPrimitive<P>,
    {
        self.y.as_()
    }

    /// Current output including wraps
    pub fn y(&self) -> Q {
        self.y
    }
}

/// Wrap classification
#[repr(i8)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Wrap {
    /// A wrap occured in the negative direction
    Negative = -1,
    /// No wrap, the wrapped difference between successive values
    /// has the same sign as their comparison
    #[default]
    None = 0,
    /// A wrap occurred in the positive direction
    Positive = 1,
}

impl From<Wrap> for Ordering {
    fn from(value: Wrap) -> Self {
        match value {
            Wrap::Negative => Self::Less,
            Wrap::None => Self::Equal,
            Wrap::Positive => Self::Greater,
        }
    }
}

impl From<Ordering> for Wrap {
    fn from(value: Ordering) -> Self {
        match value {
            Ordering::Less => Self::Negative,
            Ordering::Equal => Self::None,
            Ordering::Greater => Self::Positive,
        }
    }
}

impl core::ops::Add for Wrap {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match self as i32 + rhs as i32 {
            ..0 => Self::Negative,
            0 => Self::None,
            1.. => Self::Positive,
        }
    }
}

/// Maps wraps to saturation
///
/// Clamps output to the value range on wraps and only un-clamps on
/// (one corresponding) un-wrap.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Clamp<Q> {
    /// Last input value
    pub x0: Q,
    /// Clamp indicator
    pub clamp: Wrap,
}

impl<Q> Clamp<Q>
where
    Q: 'static + Zero + PartialOrd + WrappingSub + Copy + Bounded,
{
    /// Update the clamp with a new input
    ///
    /// IF (positive wrap and negative clamp,
    /// OR negative wrap and positive clamp,
    /// OR no wrap and no clamp): output the input.
    /// ELSE IF negative wrap: clamp minimum,
    /// ELSE IF positive wrap: clamp maximum.
    pub fn update(&mut self, x: Q) -> Q {
        let (_dx, wrap) = overflowing_sub(x, self.x0);
        self.x0 = x;
        match self.clamp as i32 + wrap as i32 {
            ..0 => {
                self.clamp = Wrap::Negative;
                Q::min_value()
            }
            1.. => {
                self.clamp = Wrap::Positive;
                Q::max_value()
            }
            0 => {
                self.clamp = Wrap::None;
                x
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overflowing_sub_correctness() {
        for (x0, x1, v) in [
            (0i32, 0i32, Wrap::None),
            (0, 1, Wrap::None),
            (0, -1, Wrap::None),
            (1, 0, Wrap::None),
            (-1, 0, Wrap::None),
            (0, 0x7fff_ffff, Wrap::None),
            (-1, 0x7fff_ffff, Wrap::Negative),
            (-2, 0x7fff_ffff, Wrap::Negative),
            (-1, -0x8000_0000, Wrap::None),
            (0, -0x8000_0000, Wrap::None),
            (1, -0x8000_0000, Wrap::Positive),
            (-0x6000_0000, 0x6000_0000, Wrap::Negative),
            (0x6000_0000, -0x6000_0000, Wrap::Positive),
            (-0x4000_0000, 0x3fff_ffff, Wrap::None),
            (-0x4000_0000, 0x4000_0000, Wrap::Negative),
            (-0x4000_0000, 0x4000_0001, Wrap::Negative),
            (0x4000_0000, -0x3fff_ffff, Wrap::None),
            (0x4000_0000, -0x4000_0000, Wrap::None),
            (0x4000_0000, -0x4000_0001, Wrap::Positive),
        ]
        .iter()
        {
            let (dx, w) = overflowing_sub(*x1, *x0);
            assert_eq!(*v, w, " = overflowing_sub({:#x}, {:#x})", *x0, *x1);
            let (dx0, w0) = x1.overflowing_sub(*x0);
            assert_eq!(w0, w != Wrap::None);
            assert_eq!(dx, dx0);
        }
    }

    #[test]
    fn saturating_scale_correctness() {
        let shift = 8;
        for (lo, hi, res) in [
            (0i32, 0i32, 0i32),
            (0, 1, 0x0100_0000),
            (0, -1, -0x0100_0000),
            (0x100, 0, 1),
            (-1 << 31, 0, -1 << 23),
            (0x7fffffff, 0, 0x007f_ffff),
            (0x7fffffff, 1, 0x0017f_ffff),
            (-0x7fffffff, -1, -0x0180_0000),
            (0x1234_5600, 0x7f, 0x7f12_3456),
            (0x1234_5600, -0x7f, -0x7f00_0000 + 0x12_3456),
            (0, 0x7f, 0x7f00_0000),
            (0, 0x80, 0x7fff_ff80),
            (0, -0x7f, -0x7f00_0000),
            (0, -0x80, -0x7fff_ff80),
            (0x7fff_ffff, 0x7f, 0x7f7f_ffff),
            (-0x8000_0000, 0x7f, 0x7e80_0000),
            (-0x8000_0000, -0x7f, -0x7f80_0000),
            (0x7fff_ffff, -0x7f, -0x7e80_0001),
            (0x100, 0x7f, 0x7f00_0001),
            (0, -0x80, -0x7fff_ff80),
            (-1 << 31, 0x80, 0x7fff_ff80),
            (-1 << 31, -0x80, -0x7fff_ff80),
        ]
        .iter()
        {
            let s = saturating_scale(*lo, *hi, shift);
            assert_eq!(
                *res, s,
                "{:#x} != {:#x} = saturating_scale({:#x}, {:#x}, {:#x})",
                *res, s, *lo, *hi, shift
            );
        }
    }
}
