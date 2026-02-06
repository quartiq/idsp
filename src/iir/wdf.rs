//! Wave digital filters

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::FloatCore as _;

use dsp_fixedpoint::Q32;
use dsp_process::{SplitInplace, SplitProcess};

/// Two port adapter architecture
///
/// Each architecture is a nibble in the const generic of [`Wdf`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Tpa {
    /// terminate
    Z = 0x0,
    /// 1 > g > 1/2: a = g - 1
    A = 0xA,
    /// 1/2 >= g > 0: a = -g
    B = 0xB,
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
    fn quantize(self, g: f64) -> Option<Q32<32>> {
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
        (-0.5..=0.0).contains(&a).then_some(Q32::from_f64(a))
    }

    #[inline]
    fn adapt(&self, x: [i32; 2], a: Q32<32>) -> [i32; 2] {
        match self {
            Tpa::A => {
                let c = x[1] - x[0];
                let y = (c * a).wrapping_add(x[1]);
                [y.wrapping_add(c), y]
            }
            Tpa::B => {
                let c = x[0] - x[1];
                let y = (c * a).wrapping_add(x[1]);
                [y, y.wrapping_add(c)]
            }
            Tpa::B1 => {
                let c = x[0] - x[1];
                let y = c * a;
                [y.wrapping_add(x[1]), y.wrapping_add(x[0])]
            }
            Tpa::X => [x[1], x[0]],
            Tpa::C => {
                let c = x[1] - x[0];
                let y = (c * a).wrapping_sub(x[1]);
                [y, y.wrapping_add(c)]
            }
            Tpa::C1 => {
                let c = x[1] - x[0];
                let y = c * a;
                [y.wrapping_sub(x[1]), y.wrapping_sub(x[0])]
            }
            Tpa::D => {
                let c = x[0] - x[1];
                let y = (c * a).wrapping_sub(x[1]);
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
    pub a: [Q32<32>; N],
}

impl<const N: usize, const M: u32> Default for Wdf<N, M> {
    fn default() -> Self {
        Self {
            a: [Default::default(); N],
        }
    }
}

impl<const N: usize, const M: u32> Wdf<N, M> {
    /// Quantize and scale filter coefficients
    ///
    /// The coefficients are the allpass poles.
    /// The type (configuration nibbles M) must match the
    /// optimal scaled architecture, see [`Tpa`].
    pub fn quantize(g: &[f64; N]) -> Option<Self> {
        let mut a = [Default::default(); N];
        let mut m = M;
        for (a, g) in a.iter_mut().zip(g) {
            *a = Tpa::from((m & 0xf) as u8).quantize(*g)?;
            m >>= 4;
        }
        debug_assert_eq!(m, 0);
        Some(Self { a })
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

impl<const N: usize, const M: u32> SplitProcess<i32, i32, WdfState<N>> for Wdf<N, M> {
    #[inline]
    fn process(&self, state: &mut WdfState<N>, x: i32) -> i32 {
        let mut y = 0;
        let (m, x, z) =
            self.a
                .iter()
                .zip(state.z.iter_mut())
                .fold((M, x, &mut y), |(m, mut x, y), (a, z)| {
                    [*y, x] = Tpa::from((m & 0xf) as u8).adapt([x, *z], *a);
                    (m >> 4, x, z)
                });
        debug_assert_eq!(m, 0);
        *z = x;
        y
    }
}

impl<const N: usize, const M: u32> SplitInplace<i32, WdfState<N>> for Wdf<N, M> {}
