use core::{fmt, fmt::Write, num::Wrapping};

use crate::{FloatValue, Q};

/// ```
/// # use dsp_fixedpoint::Q8;
/// let q = Q8::<4>::new(7);
/// assert_eq!(format!("{q} {q:e} {q:E}"), "0.4375 4.375e-1 4.375E-1");
/// ```
macro_rules! impl_fmt {
    ($tr:path) => {
        impl<T, A, const F: i8> $tr for Q<T, A, F>
        where
            T: FloatValue,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                <f64 as $tr>::fmt(&(self.inner.as_f64() * Self::DELTA as f64), f)
            }
        }
    };
}
impl_fmt!(fmt::Display);
impl_fmt!(fmt::UpperExp);
impl_fmt!(fmt::LowerExp);

#[cfg(feature = "defmt")]
impl<T, A, const F: i8> defmt::Format for Q<T, A, F>
where
    T: FloatValue,
{
    fn format(&self, fmt: defmt::Formatter<'_>) {
        defmt::write!(fmt, "{=f32}", self.inner.as_f32() * Self::DELTA);
    }
}

/// ```
/// # use dsp_fixedpoint::Q8;
/// assert_eq!(format!("{:?}", Q8::<4>::new(0x14)), "20");
/// assert_eq!(format!("{:#b}", Q8::<3>::new(0b01101001)), "0b1101.001");
/// assert_eq!(format!("{:x}", Q8::<-2>::new(3)), "c.");
/// assert_eq!(format!("{:x}", Q8::<4>::new(-0x14)), "-1.4");
/// ```
impl<T: fmt::Debug, A, const F: i8> fmt::Debug for Q<T, A, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[derive(Copy, Clone)]
struct Radix {
    bits: u8,
    prefix: &'static str,
    digits: &'static [u8],
}

impl Radix {
    const BINARY: Self = Self {
        bits: 1,
        prefix: "0b",
        digits: b"01",
    };
    const OCTAL: Self = Self {
        bits: 3,
        prefix: "0o",
        digits: b"01234567",
    };
    const LOWER_HEX: Self = Self {
        bits: 4,
        prefix: "0x",
        digits: b"0123456789abcdef",
    };
    const UPPER_HEX: Self = Self {
        bits: 4,
        prefix: "0X",
        digits: b"0123456789ABCDEF",
    };

    #[inline]
    fn mask(self) -> u8 {
        (1u8 << self.bits) - 1
    }

    #[inline]
    fn digit(self, value: u8) -> u8 {
        self.digits[value as usize]
    }
}

trait RadixValue: Copy {
    fn is_negative(self) -> bool;
    fn magnitude(self) -> u64;
}

macro_rules! impl_unsigned_radix_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl RadixValue for $ty {
                #[inline]
                fn is_negative(self) -> bool {
                    false
                }

                #[inline]
                fn magnitude(self) -> u64 {
                    self.into()
                }
            }
        )*
    };
}

macro_rules! impl_signed_radix_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl RadixValue for $ty {
                #[inline]
                fn is_negative(self) -> bool {
                    self < 0
                }

                #[inline]
                fn magnitude(self) -> u64 {
                    self.unsigned_abs().into()
                }
            }
        )*
    };
}

macro_rules! impl_wrapping_unsigned_radix_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl RadixValue for Wrapping<$ty> {
                #[inline]
                fn is_negative(self) -> bool {
                    false
                }

                #[inline]
                fn magnitude(self) -> u64 {
                    self.0.into()
                }
            }
        )*
    };
}

macro_rules! impl_wrapping_signed_radix_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl RadixValue for Wrapping<$ty> {
                #[inline]
                fn is_negative(self) -> bool {
                    self.0 < 0
                }

                #[inline]
                fn magnitude(self) -> u64 {
                    self.0.unsigned_abs().into()
                }
            }
        )*
    };
}

impl_unsigned_radix_value!(u8, u16, u32, u64);
impl_signed_radix_value!(i8, i16, i32, i64);
impl_wrapping_unsigned_radix_value!(u8, u16, u32, u64);
impl_wrapping_signed_radix_value!(i8, i16, i32, i64);

#[inline]
const fn ceil_digits(bits: usize, group: u8) -> usize {
    match group {
        1 => bits,
        3 => bits.div_ceil(3),
        4 => bits.div_ceil(4),
        _ => unreachable!(),
    }
}

#[inline]
const fn split_bits(bits: u8, group: u8) -> (usize, u8) {
    match group {
        1 => (bits as usize, 0),
        3 => ((bits / 3) as usize, bits % 3),
        4 => ((bits / 4) as usize, bits % 4),
        _ => unreachable!(),
    }
}

#[inline]
fn shifted_digit(magnitude: u64, shift: u8, radix: Radix, index: usize) -> u8 {
    let mask = u64::from(radix.mask());
    let offset = index * radix.bits as usize;
    if offset >= shift as usize {
        let right = offset - shift as usize;
        if right >= u64::BITS as usize {
            0
        } else {
            ((magnitude >> right) & mask) as u8
        }
    } else {
        ((magnitude << (shift as usize - offset)) & mask) as u8
    }
}

fn pad(f: &mut fmt::Formatter<'_>, fill: char, count: usize) -> fmt::Result {
    for _ in 0..count {
        f.write_char(fill)?;
    }
    Ok(())
}

fn format_fixed(
    negative: bool,
    magnitude: u64,
    frac_bits: i8,
    f: &mut fmt::Formatter<'_>,
    radix: Radix,
) -> fmt::Result {
    let magnitude_bits = (u64::BITS - magnitude.leading_zeros()) as usize;
    let body_len = if frac_bits > 0 {
        let frac_bits = frac_bits as usize;
        let frac_digits = ceil_digits(frac_bits, radix.bits);
        let effective_digits = if magnitude == 0 {
            0
        } else {
            ceil_digits(
                magnitude_bits + (frac_digits * radix.bits as usize - frac_bits),
                radix.bits,
            )
        };
        effective_digits.saturating_sub(frac_digits).max(1) + 1 + frac_digits
    } else {
        let (zero_digits, shift) = split_bits(frac_bits.unsigned_abs(), radix.bits);
        if magnitude == 0 {
            2
        } else {
            ceil_digits(magnitude_bits + shift as usize, radix.bits) + zero_digits + 1
        }
    };
    let sign = if negative {
        "-"
    } else if f.sign_plus() {
        "+"
    } else {
        ""
    };
    let prefix = if f.alternate() { radix.prefix } else { "" };
    let total_len = sign.len() + prefix.len() + body_len;
    let pad_len = f.width().map_or(0, |width| width.saturating_sub(total_len));
    let align = f.align();
    let fill = f.fill();
    let zero_pad = if f.sign_aware_zero_pad() && align.is_none() {
        pad_len
    } else {
        0
    };
    let align = align.unwrap_or(fmt::Alignment::Right);
    let (left_pad, right_pad) = if zero_pad != 0 {
        (0, 0)
    } else {
        match align {
            fmt::Alignment::Left => (0, pad_len),
            fmt::Alignment::Center => (pad_len / 2, pad_len - pad_len / 2),
            fmt::Alignment::Right => (pad_len, 0),
        }
    };

    pad(f, fill, left_pad)?;
    f.write_str(sign)?;
    f.write_str(prefix)?;
    pad(f, '0', zero_pad)?;

    if frac_bits > 0 {
        let frac_bits = frac_bits as usize;
        let frac_digits = ceil_digits(frac_bits, radix.bits);
        let shift = (frac_digits * radix.bits as usize - frac_bits) as u8;
        let effective_digits = if magnitude == 0 {
            0
        } else {
            ceil_digits(magnitude_bits + shift as usize, radix.bits)
        };
        if effective_digits <= frac_digits {
            f.write_char('0')?;
        } else {
            for index in (frac_digits..effective_digits).rev() {
                f.write_char(radix.digit(shifted_digit(magnitude, shift, radix, index)) as char)?;
            }
        }
        f.write_char('.')?;
        for index in (0..frac_digits).rev() {
            f.write_char(radix.digit(shifted_digit(magnitude, shift, radix, index)) as char)?;
        }
    } else {
        let (zero_digits, shift) = split_bits(frac_bits.unsigned_abs(), radix.bits);
        if magnitude == 0 {
            f.write_char('0')?;
        } else {
            let digits = ceil_digits(magnitude_bits + shift as usize, radix.bits);
            for index in (0..digits).rev() {
                f.write_char(radix.digit(shifted_digit(magnitude, shift, radix, index)) as char)?;
            }
            for _ in 0..zero_digits {
                f.write_char(radix.digit(0) as char)?;
            }
        }
        f.write_char('.')?;
    }

    pad(f, fill, right_pad)
}

macro_rules! impl_radix_fmt {
    ($tr:path, $radix:expr) => {
        impl<T: RadixValue, A, const F: i8> $tr for Q<T, A, F> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                const {
                    assert!(
                        F != i8::MIN,
                        "fractional bit count must not be i8::MIN for formatting"
                    );
                }
                format_fixed(
                    self.inner.is_negative(),
                    self.inner.magnitude(),
                    F,
                    f,
                    $radix,
                )
            }
        }
    };
}

impl_radix_fmt!(fmt::Binary, Radix::BINARY);
impl_radix_fmt!(fmt::Octal, Radix::OCTAL);
impl_radix_fmt!(fmt::LowerHex, Radix::LOWER_HEX);
impl_radix_fmt!(fmt::UpperHex, Radix::UPPER_HEX);

#[cfg(test)]
mod test {
    #[cfg(feature = "defmt")]
    #[test]
    fn defmt_format_impls_exist() {
        fn assert_defmt<T: defmt::Format>() {}

        assert_defmt::<crate::Q8<4>>();
        assert_defmt::<crate::P8<4>>();
        assert_defmt::<crate::W8<4>>();
        assert_defmt::<crate::V8<4>>();
    }

    #[cfg(feature = "std")]
    #[test]
    fn display() {
        use crate::Q32;
        use std::format;

        assert_eq!(format!("{}", Q32::<9>::new(0x12345)), "145.634765625");
        assert_eq!(format!("{}", Q32::<9>::from_int(99)), "99");
    }

    #[cfg(feature = "std")]
    #[test]
    fn float_accessors_cover_wrapping_types() {
        use crate::{V8, W8};
        use core::num::Wrapping;

        assert_eq!(W8::<4>::new(Wrapping(-4)).as_f32(), -0.25);
        assert_eq!(V8::<4>::new(Wrapping(4)).as_f64(), 0.25);
    }

    #[cfg(feature = "std")]
    #[test]
    fn radix_dot_examples() {
        use crate::Q8;
        use std::format;

        assert_eq!(format!("{:#b}", Q8::<3>::new(0b01101001)), "0b1101.001");
        assert_eq!(format!("{:x}", Q8::<3>::new(0b01101001)), "d.2");
        assert_eq!(format!("{:o}", Q8::<5>::new(1)), "0.02");
        assert_eq!(format!("{:x}", Q8::<-2>::new(3)), "c.");
    }

    #[cfg(feature = "std")]
    #[test]
    fn radix_dot_leading_zero_and_zero_value() {
        use crate::Q8;
        use std::format;

        assert_eq!(format!("{:b}", Q8::<3>::new(1)), "0.001");
        assert_eq!(format!("{:x}", Q8::<7>::new(1)), "0.02");
        assert_eq!(format!("{:#x}", Q8::<7>::new(1)), "0x0.02");
        assert_eq!(format!("{:b}", Q8::<5>::new(0)), "0.00000");
        assert_eq!(format!("{:x}", Q8::<-5>::new(0)), "0.");
    }

    #[cfg(feature = "std")]
    #[test]
    fn radix_dot_signed_values_are_magnitude_based() {
        use crate::{Q8, W8};
        use core::num::Wrapping;
        use std::format;

        assert_eq!(format!("{:b}", Q8::<3>::new(-0x14)), "-10.100");
        assert_eq!(format!("{:#x}", Q8::<4>::new(-0x14)), "-0x1.4");
        assert_eq!(format!("{:o}", Q8::<0>::new(-1)), "-1.");
        assert_eq!(format!("{:x}", Q8::<4>::new(i8::MIN)), "-8.0");
        assert_eq!(format!("{:#b}", W8::<3>::new(Wrapping(-0x14))), "-0b10.100");
    }

    #[cfg(feature = "std")]
    #[test]
    fn radix_dot_unsigned_and_wrapping_unsigned() {
        use crate::{P8, V8};
        use core::num::Wrapping;
        use std::format;

        assert_eq!(format!("{:x}", P8::<4>::new(u8::MAX)), "f.f");
        assert_eq!(
            format!("{:b}", V8::<3>::new(Wrapping(0b1111_1111))),
            "11111.111"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn radix_dot_handles_large_positive_and_negative_f() {
        use crate::{Q8, Q64};
        use std::format;

        assert_eq!(format!("{:b}", Q8::<7>::new(i8::MAX)), "0.1111111");
        assert_eq!(format!("{:b}", Q8::<-7>::new(1)), "10000000.");
        assert_eq!(
            format!("{:x}", Q64::<63>::new(i64::MAX)),
            "0.fffffffffffffffe"
        );
        assert_eq!(format!("{:x}", Q64::<-63>::new(1)), "8000000000000000.");
        assert_eq!(
            format!("{:b}", Q64::<-63>::new(1)),
            "1\
000000000000000000000000000000000000000000000000000000000000000."
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn radix_dot_handles_zero_fractional_bits() {
        use crate::Q8;
        use std::format;

        assert_eq!(format!("{:b}", Q8::<0>::new(0b1010)), "1010.");
        assert_eq!(format!("{:#x}", Q8::<0>::new(0x2a)), "0x2a.");
    }

    #[cfg(feature = "std")]
    #[test]
    fn radix_dot_respects_width_alignment_and_zero_fill() {
        use crate::Q8;
        use std::format;

        assert_eq!(format!("{:>10x}", Q8::<4>::new(0x14)), "       1.4");
        assert_eq!(format!("{:#010x}", Q8::<4>::new(0x14)), "0x000001.4");
        assert_eq!(format!("{:#010x}", Q8::<4>::new(-0x14)), "-0x00001.4");
        assert_eq!(format!("{:<010x}", Q8::<4>::new(0x14)), "1.4       ");
        assert_eq!(format!("{:^010x}", Q8::<4>::new(0x14)), "   1.4    ");
    }

    #[cfg(feature = "std")]
    #[test]
    fn debug_stays_raw() {
        use crate::Q8;
        use std::format;

        assert_eq!(format!("{:?}", Q8::<3>::new(-0x14)), "-20");
        assert_eq!(format!("{:b}", Q8::<3>::new(-0x14)), "-10.100");
    }
}
