use num_traits::Float;

use super::{Process, StatefulRef};

/// Two port adapter
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
            0xc => Tpa::C,
            0xd => Tpa::D,
            0xe => Tpa::B1,
            0xf => Tpa::C1,
            0x0 => Tpa::Z,
            _ => Tpa::X,
        }
    }
}

impl Tpa {
    fn quantize(self, g: f64) -> Result<i32, Self> {
        const S: f64 = (1u64 << 32) as _;
        let (t, a) = match g {
            0.0 => (Self::X, g),
            0.5..1.0 => (Self::A, g - 1.0),
            0.0..0.5 => (Self::B, -g),
            -0.5..0.0 => (Self::C, g),
            -1.0..-0.5 => (Self::D, -1.0 - g),
            _ => (Self::Z, 0.0),
        };
        assert!(a <= 0.0);
        assert!(a >= -0.5);
        if t == self {
            Ok((a * S).round() as _)
        } else {
            Err(t)
        }
    }

    #[inline]
    fn adapt(&self, x: [i32; 2], a: i32) -> [i32; 2] {
        #[inline]
        fn mm(x: i32, y: i32) -> i32 {
            ((x as i64 * y as i64) >> 32) as i32
        }

        match self {
            Tpa::A => {
                let c = x[1] - x[0];
                let y = mm(a, c) + x[1];
                [y + c, y]
            }
            Tpa::B => {
                let c = x[0] - x[1];
                let y = mm(a, c) + x[1];
                [y, y + c]
            }
            Tpa::B1 => {
                let c = x[0] - x[1];
                let y = mm(a, c);
                [y + x[1], y + x[0]]
            }
            Tpa::X => [x[1], x[0]],
            Tpa::C => {
                let c = x[1] - x[0];
                let y = mm(a, c) - x[1];
                [y, y + c]
            }
            Tpa::C1 => {
                let c = x[1] - x[0];
                let y = mm(a, c);
                [y - x[1], y - x[0]]
            }
            Tpa::D => {
                let c = x[0] - x[1];
                let y = mm(a, c) - x[1];
                [y + c, y]
            }
            Tpa::Z => x,
        }
    }
}

/// Wave digital filter
#[derive(Default, Debug, Clone)]
pub struct Wdf1<const M: u8> {
    /// Filter coefficient
    pub a: i32,
}

/// Wave digital filter state
pub struct Wdf1State {
    /// Filter state
    pub z: i32,
}

impl<const M: u8> Process for StatefulRef<'_, Wdf1<M>, Wdf1State> {
    fn process(&mut self, x0: i32) -> i32 {
        let y = Tpa::from(M & 0xf).adapt([x0, self.1.z], self.0.a);
        self.1.z = y[1];
        y[0]
    }
}

/// Wave digital filter
///
/// This enforces compile-time knowledge about the two TPA architectures
/// in the two nibbles of the const generic.
#[derive(Default, Debug, Clone)]
pub struct Wdf2<const M: u8> {
    /// Filter coefficients
    pub a: [i32; 2],
}

impl<const M: u8> Wdf2<M> {
    /// Quantize and scale filter coefficients
    pub fn quantize(g: [f64; 2]) -> Result<Self, Tpa> {
        Ok(Self {
            a: [
                Tpa::from((M >> 0) & 0xf).quantize(g[0])?,
                Tpa::from((M >> 4) & 0xf).quantize(g[1])?,
            ],
        })
    }
}

/// Wave digital filter state
#[derive(Default, Debug, Clone)]
pub struct Wdf2State {
    /// Filter state
    pub z: [i32; 2],
}

impl<const M: u8> Process for StatefulRef<'_, Wdf2<M>, Wdf2State> {
    fn process(&mut self, x0: i32) -> i32 {
        let y0 = Tpa::from((M >> 0) & 0xf).adapt([x0, self.1.z[0]], self.0.a[0]);
        self.1.z = Tpa::from((M >> 4) & 0xf).adapt([y0[1], self.1.z[1]], self.0.a[1]);
        y0[0]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Gazsi 1985, Example 5
    pub struct Nineteen {
        a: (Wdf2<0x1c>, Wdf2<0x1d>, Wdf2<0x1d>, Wdf2<0x1d>, Wdf2<0x01>),
        b: (Wdf2<0x1c>, Wdf2<0x1c>, Wdf2<0x1d>, Wdf2<0x1d>, Wdf2<0x1d>),
    }

    impl Process for StatefulRef<'_, Nineteen, [Wdf2State; 10]> {
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
        pub fn inplace(&self, s: &mut [Wdf2State; 10], xy0: &mut [i32; 8]) {
            StatefulRef(self, s).process_in_place(xy0);
        }

        pub fn block(&self, s: &mut [Wdf2State; 10], xy0: &mut [[i32; 8]; 2]) {
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
