#[cfg(not(feature = "std"))]
use num_traits::float::FloatCore as _;

use crate::process::{Inplace, Process, Split};

/// Two port adapter architecture
///
/// Each architecture is a nibble in the const generic of [`Wdf`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Tpa {
    /// terminate
    Z = 0x0,
    /// 1 > g > 1/2: a = g - 1
    A = 0xa,
    /// 1/2 >= g > 0: a = -g
    B = 0xb,
    /// alternative to B
    B1 = 0xE,
    /// g = 0
    X = 0x1,
    /// -1/2 <= g < 0: a = g
    C = 0xC,
    /// alternative to C
    C1 = 0xF,
    /// -1 < g < -1/2: a = -(1 + g)
    D = 0xD,
}

impl From<u8> for Tpa {
    #[inline]
    fn from(value: u8) -> Self {
        match value {
            0xa => Tpa::A,
            0xb => Tpa::B,
            0xe => Tpa::B1,
            0x1 => Tpa::X,
            0xc => Tpa::C,
            0xf => Tpa::C1,
            0xd => Tpa::D,
            _ => Tpa::Z,
        }
    }
}

impl Tpa {
    fn quantize(self, g: f64) -> Option<i32> {
        // Use negative -0.5 <= a <= 0 instead of the usual positive
        // as -0.5 just fits the Q32 fixed point range.
        let a = match self {
            Self::Z => 0.0,
            Self::A => g - 1.0,
            Self::B | Self::B1 => -g,
            Self::X => 0.0,
            Self::C | Self::C1 => g,
            Self::D => -1.0 - g,
        };
        (-0.5..=0.0)
            .contains(&a)
            .then_some((a * (1u64 << 32) as f64).round() as _)
    }

    #[inline]
    fn adapt(&self, x: [i32; 2], a: i32) -> [i32; 2] {
        fn mul(x: i32, y: i32) -> i32 {
            ((x as i64 * y as i64) >> 32) as i32
        }

        match self {
            Tpa::A => {
                let c = x[1] - x[0];
                let y = mul(a, c).wrapping_add(x[1]);
                [y.wrapping_add(c), y]
            }
            Tpa::B => {
                let c = x[0] - x[1];
                let y = mul(a, c).wrapping_add(x[1]);
                [y, y.wrapping_add(c)]
            }
            Tpa::B1 => {
                let c = x[0] - x[1];
                let y = mul(a, c);
                [y.wrapping_add(x[1]), y.wrapping_add(x[0])]
            }
            Tpa::X => [x[1], x[0]],
            Tpa::C => {
                let c = x[1] - x[0];
                let y = mul(a, c).wrapping_sub(x[1]);
                [y, y.wrapping_add(c)]
            }
            Tpa::C1 => {
                let c = x[1] - x[0];
                let y = mul(a, c);
                [y.wrapping_sub(x[1]), y.wrapping_sub(x[0])]
            }
            Tpa::D => {
                let c = x[0] - x[1];
                let y = mul(a, c).wrapping_sub(x[1]);
                [y.wrapping_add(c), y]
            }
            Tpa::Z => x,
        }
    }
}

/// Wave digital filter, order N, configuration M
///
/// Allpass
///
/// The M const generic enforces compile time knowledge about
/// the two port adapter architecture. Each nibble is one TPA.
#[derive(Debug, Clone)]
pub struct Wdf<const N: usize, const M: u32> {
    /// Filter coefficient
    pub a: [i32; N],
}

impl<const N: usize, const M: u32> Default for Wdf<N, M> {
    fn default() -> Self {
        Self { a: [0; N] }
    }
}

impl<const N: usize, const M: u32> Wdf<N, M> {
    /// Quantize and scale filter coefficients
    ///
    /// The coefficients are the allpass poles.
    /// The type (configuration nibbles M) must match the
    /// optimal scaled architecture, see [`Tpa`].
    pub fn quantize(g: &[f64; N]) -> Option<Self> {
        let mut a = [0; N];
        let mut m = M;
        for (a, g) in a.iter_mut().zip(g) {
            *a = Tpa::from((m & 0xf) as u8).quantize(*g)?;
            m >>= 4;
        }
        (m == 0).then_some(Self { a })
    }
}

/// Wave digital filter state, order N
#[derive(Clone, Debug)]
pub struct WdfState<const N: usize> {
    /// Filter state
    pub z: [i32; N],
}

impl<const N: usize> Default for WdfState<N> {
    fn default() -> Self {
        Self { z: [0; N] }
    }
}

impl<const N: usize, const M: u32> Process<i32> for Split<&Wdf<N, M>, &mut WdfState<N>> {
    #[inline]
    fn process(&mut self, x0: i32) -> i32 {
        let mut y = 0;
        let (_, x, z) = self.config.a.iter().zip(self.state.z.iter_mut()).fold(
            (M, x0, &mut y),
            |(m, x, y), (a, z0)| {
                let z1;
                [*y, z1] = Tpa::from((m & 0xf) as u8).adapt([x, *z0], *a);
                (m >> 4, z1, z0)
            },
        );
        *z = x;
        y
    }
}

impl<const N: usize, const M: u32> Inplace<i32> for Split<&Wdf<N, M>, &mut WdfState<N>> {}
