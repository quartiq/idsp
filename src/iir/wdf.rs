#[cfg(not(feature = "std"))]
use num_traits::float::FloatCore as _;

use super::{Process, StatefulRef};

/// Two port adapter
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum Tpa {
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
    fn quantize(self, g: f64) -> Result<i32, Self> {
        // Use negative -0.5 <= a <= 0 instead of the usual positive
        // as -0.5 just fits the Q32 fixed point range.
        const S: f64 = (1u64 << 32) as _;
        let a = match self {
            Self::Z => 0.0,
            Self::A => g - 1.0,
            Self::B | Self::B1 => -g,
            Self::X => 0.0,
            Self::C | Self::C1 => g,
            Self::D => -1.0 - g,
        };
        if (-0.5..=0.0).contains(&a) {
            Ok((a * S).round() as _)
        } else {
            Err(self)
        }
    }

    #[inline]
    fn adapt(&self, x: [i32; 2], a: i32) -> [i32; 2] {
        #[inline]
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
    pub fn quantize(g: &[f64; N]) -> Result<Self, f64> {
        let mut a = [0; N];
        let mut m = M;
        for (a, g) in a.iter_mut().zip(g) {
            *a = Tpa::from((m & 0xf) as u8).quantize(*g).or(Err(*g))?;
            m >>= 4;
        }
        Ok(Self { a })
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

impl<const N: usize, const M: u32> Process for StatefulRef<'_, Wdf<N, M>, WdfState<N>> {
    fn process(&mut self, x0: i32) -> i32 {
        let mut y = 0;
        let (m, x, z) =
            self.0
                .a
                .iter()
                .zip(self.1.z.iter_mut())
                .fold((M, x0, &mut y), |(m, x, y), (a, z)| {
                    let yx = Tpa::from((m & 0xf) as u8).adapt([x, *z], *a);
                    *y = yx[0];
                    (m >> 4, yx[1], z)
                });
        debug_assert_eq!(m, 0);
        *z = x;
        y
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Gazsi 1985, Example 5
    pub struct Nineteen {
        a: (
            Wdf<2, 0x1c>,
            Wdf<2, 0x1d>,
            Wdf<2, 0x1d>,
            Wdf<2, 0x1d>,
            Wdf<2, 0x01>,
        ),
        b: (
            Wdf<2, 0x1c>,
            Wdf<2, 0x1c>,
            Wdf<2, 0x1d>,
            Wdf<2, 0x1d>,
            Wdf<2, 0x1d>,
        ),
    }

    impl Process for StatefulRef<'_, Nineteen, [WdfState<2>; 10]> {
        fn process(&mut self, x0: i32) -> i32 {
            let mut xa = StatefulRef(&self.0.a.0, &mut self.1[0]).process(x0);
            xa = StatefulRef(&self.0.a.1, &mut self.1[1]).process(xa);
            xa = StatefulRef(&self.0.a.2, &mut self.1[2]).process(xa);
            xa = StatefulRef(&self.0.a.3, &mut self.1[3]).process(xa);
            xa = StatefulRef(&self.0.a.4, &mut self.1[4]).process(xa);
            let mut xb = StatefulRef(&self.0.b.0, &mut self.1[5]).process(x0);
            xb = StatefulRef(&self.0.b.1, &mut self.1[6]).process(xb);
            xb = StatefulRef(&self.0.b.2, &mut self.1[7]).process(xb);
            xb = StatefulRef(&self.0.b.3, &mut self.1[8]).process(xb);
            xb = StatefulRef(&self.0.b.4, &mut self.1[9]).process(xb);
            xa + xb
        }
    }

    impl Nineteen {
        pub fn inplace(&self, s: &mut [WdfState<2>; 10], xy0: &mut [i32; 8]) {
            StatefulRef(self, s).process_in_place(xy0);
        }

        pub fn block(&self, s: &mut [WdfState<2>; 10], xy0: &mut [[i32; 8]; 2]) {
            xy0[1] = xy0[0].clone();
            StatefulRef(&self.a.0, &mut s[0]).process_in_place(&mut xy0[1]);
            StatefulRef(&self.a.1, &mut s[1]).process_in_place(&mut xy0[1]);
            StatefulRef(&self.a.2, &mut s[2]).process_in_place(&mut xy0[1]);
            StatefulRef(&self.a.3, &mut s[3]).process_in_place(&mut xy0[1]);
            StatefulRef(&self.a.4, &mut s[4]).process_in_place(&mut xy0[1]);
            StatefulRef(&self.b.0, &mut s[5]).process_in_place(&mut xy0[0]);
            StatefulRef(&self.b.1, &mut s[6]).process_in_place(&mut xy0[0]);
            StatefulRef(&self.b.2, &mut s[7]).process_in_place(&mut xy0[0]);
            StatefulRef(&self.b.3, &mut s[8]).process_in_place(&mut xy0[0]);
            StatefulRef(&self.b.4, &mut s[9]).process_in_place(&mut xy0[0]);
            let [a, b] = xy0.each_mut();
            for (a, b) in a.iter_mut().zip(b) {
                (*a, *b) = (*a + *b, *a - *b);
            }
        }
    }
}
