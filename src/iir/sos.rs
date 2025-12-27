use dsp_fixedpoint::Q32;
use dsp_process::{SplitInplace, SplitProcess};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::FloatCore as _;

/// Second-order-section
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Sos<const F: i8 = 29> {
    /// Coefficients
    ///
    /// `[b0, b1, b2, a1, a2]`
    /// Such that:
    /// `y0 = (b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2)/(1 << F)`
    ///
    /// Note the a1, a2 sign:
    /// `H = (b0 + b1*z^-1 + b2*z^-2)/((1 << F) - a2*z^-1 - a2*z^-2)`
    pub ba: [Q32<F>; 5],
}

/// Second-order-section with offset and clamp
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SosClamp<const F: i8 = 29> {
    /// Coefficients
    pub coeff: Sos<F>,
    /// Summing junction offset
    pub u: i32,
    /// Summing junction min clamp
    pub min: i32,
    /// Summing junction min clamp
    pub max: i32,
}

impl<const F: i8> Default for SosClamp<F> {
    fn default() -> Self {
        Self {
            coeff: Default::default(),
            u: 0,
            min: i32::MIN,
            max: i32::MAX,
        }
    }
}

impl<const F: i8> SosClamp<F> {
    /// Forward gain
    pub fn k(&self) -> Q32<F> {
        self.coeff.ba[..3].iter().copied().sum()
    }

    /// Summing junction offset referred to input
    pub fn input_offset(&self) -> i32 {
        self.u / self.k()
    }

    /// Summing junction offset referred to input
    pub fn set_input_offset(&mut self, i: i32) {
        self.u = self.k() * i;
    }
}

/// SOS state
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct SosState {
    /// X,Y state
    ///
    /// `[x1, x2, y1, y2]`
    pub xy: [i32; 4],
}

/// SOS state with wide Y
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct SosStateWide {
    /// X state
    ///
    /// `[x1, x2]`
    pub x: [i32; 2],
    /// Y state
    ///
    /// `[y1, y2]`
    pub y: [i64; 2],
}

/// SOS state with first order error feedback
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct SosStateDither {
    /// X,Y state
    ///
    /// `[x1, x2, y1, y2]`
    pub xy: [i32; 4],
    /// Error feedback
    pub e: u32,
}

impl<const F: i8> SplitProcess<i32, i32, SosState> for Sos<F> {
    fn process(&self, state: &mut SosState, x0: i32) -> i32 {
        let xy = &mut state.xy;
        let ba = &self.ba;
        let mut acc = 0;
        acc += x0 as i64 * ba[0].inner as i64;
        acc += xy[0] as i64 * ba[1].inner as i64;
        acc += xy[1] as i64 * ba[2].inner as i64;
        acc += xy[2] as i64 * ba[3].inner as i64;
        acc += xy[3] as i64 * ba[4].inner as i64;
        let y0 = (acc >> F) as i32;
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const F: i8> SplitProcess<i32, i32, SosState> for SosClamp<F> {
    fn process(&self, state: &mut SosState, x0: i32) -> i32 {
        let xy = &mut state.xy;
        let ba = &self.coeff.ba;
        let mut acc = 0;
        acc += x0 as i64 * ba[0].inner as i64;
        acc += xy[0] as i64 * ba[1].inner as i64;
        acc += xy[1] as i64 * ba[2].inner as i64;
        acc += xy[2] as i64 * ba[3].inner as i64;
        acc += xy[3] as i64 * ba[4].inner as i64;
        let y0 = ((acc >> F) as i32 + self.u).clamp(self.min, self.max);
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const F: i8> SplitProcess<i32, i32, SosStateWide> for Sos<F> {
    fn process(&self, state: &mut SosStateWide, x0: i32) -> i32 {
        let x = &mut state.x;
        let y = &mut state.y;
        let ba = &self.ba;
        let mut acc = 0;
        acc += x0 as i64 * ba[0].inner as i64;
        acc += x[0] as i64 * ba[1].inner as i64;
        acc += x[1] as i64 * ba[2].inner as i64;
        *x = [x0, x[0]];
        acc += (y[0] as u32 as i64 * ba[3].inner as i64) >> 32;
        acc += (y[0] >> 32) * ba[3].inner as i64;
        acc += (y[1] as u32 as i64 * ba[4].inner as i64) >> 32;
        acc += (y[1] >> 32) * ba[4].inner as i64;
        acc <<= 32 - F;
        *y = [acc, y[0]];
        (acc >> 32) as _
    }
}

impl<const F: i8> SplitProcess<i32, i32, SosStateWide> for SosClamp<F> {
    fn process(&self, state: &mut SosStateWide, x0: i32) -> i32 {
        let x = &mut state.x;
        let y = &mut state.y;
        let ba = &self.coeff.ba;
        let mut acc = 0;
        acc += x0 as i64 * ba[0].inner as i64;
        acc += x[0] as i64 * ba[1].inner as i64;
        acc += x[1] as i64 * ba[2].inner as i64;
        *x = [x0, x[0]];
        acc += (y[0] as u32 as i64 * ba[3].inner as i64) >> 32;
        acc += (y[0] >> 32) * ba[3].inner as i64;
        acc += (y[1] as u32 as i64 * ba[4].inner as i64) >> 32;
        acc += (y[1] >> 32) * ba[4].inner as i64;
        acc <<= 32 - F;
        let y0 = ((acc >> 32) as i32 + self.u).clamp(self.min, self.max);
        *y = [((y0 as i64) << 32) | acc as u32 as i64, y[0]];
        y0
    }
}

impl<const F: i8> SplitProcess<i32, i32, SosStateDither> for Sos<F> {
    fn process(&self, state: &mut SosStateDither, x0: i32) -> i32 {
        let xy = &mut state.xy;
        let e = &mut state.e;
        let ba = &self.ba;
        let mut acc = *e as i64;
        acc += x0 as i64 * ba[0].inner as i64;
        acc += xy[0] as i64 * ba[1].inner as i64;
        acc += xy[1] as i64 * ba[2].inner as i64;
        acc += xy[2] as i64 * ba[3].inner as i64;
        acc += xy[3] as i64 * ba[4].inner as i64;
        acc <<= 32 - F;
        *e = acc as _;
        let y0 = (acc >> 32) as i32;
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const F: i8> SplitProcess<i32, i32, SosStateDither> for SosClamp<F> {
    fn process(&self, state: &mut SosStateDither, x0: i32) -> i32 {
        let xy = &mut state.xy;
        let e = &mut state.e;
        let ba = &self.coeff.ba;
        let mut acc = *e as i64;
        acc += x0 as i64 * ba[0].inner as i64;
        acc += xy[0] as i64 * ba[1].inner as i64;
        acc += xy[1] as i64 * ba[2].inner as i64;
        acc += xy[2] as i64 * ba[3].inner as i64;
        acc += xy[3] as i64 * ba[4].inner as i64;
        acc <<= 32 - F;
        *e = acc as _;
        let y0 = ((acc >> 32) as i32 + self.u).clamp(self.min, self.max);
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const F: i8> SplitInplace<i32, SosState> for Sos<F> {}
impl<const F: i8> SplitInplace<i32, SosStateWide> for Sos<F> {}
impl<const F: i8> SplitInplace<i32, SosStateDither> for Sos<F> {}
impl<const F: i8> SplitInplace<i32, SosState> for SosClamp<F> {}
impl<const F: i8> SplitInplace<i32, SosStateWide> for SosClamp<F> {}
impl<const F: i8> SplitInplace<i32, SosStateDither> for SosClamp<F> {}

fn quantize<T: From<f64>>(ba: &[[f64; 3]; 2]) -> [T; 5] {
    let a0 = 1.0 / ba[1][0];
    [
        (ba[0][0] * a0).into(),
        (ba[0][1] * a0).into(),
        (ba[0][2] * a0).into(),
        (-ba[1][1] * a0).into(),
        (-ba[1][2] * a0).into(),
    ]
}

impl<const F: i8> From<&[[f64; 3]; 2]> for Sos<F> {
    fn from(ba: &[[f64; 3]; 2]) -> Self {
        Self { ba: quantize(ba) }
    }
}

impl<const F: i8> From<&[[f64; 3]; 2]> for SosClamp<F> {
    fn from(ba: &[[f64; 3]; 2]) -> Self {
        Self {
            coeff: ba.into(),
            ..Default::default()
        }
    }
}

impl<const F: i8> From<[i32; 5]> for Sos<F> {
    fn from(mut ba: [i32; 5]) -> Self {
        ba[3] *= -1;
        ba[4] *= -1;
        Self {
            ba: ba.map(Q32::new),
        }
    }
}

impl<const F: i8> From<[i32; 5]> for SosClamp<F> {
    fn from(ba: [i32; 5]) -> Self {
        Self {
            coeff: ba.into(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(dead_code)]
    use super::*;
    use dsp_process::SplitInplace;
    // No manual tuning needed here.
    // Compiler knows best how and when:
    //   unroll loops
    //   cache on stack
    //   handle alignment
    //   register allocate variables
    //   manage pipeline and insn issue

    // cargo asm idsp::iir::sos::pnm --rust --target thumbv7em-none-eabihf --lib --target-cpu cortex-m7 --color --mca -M=-iterations=1 -M=-timeline -M=-skip-unsupported-instructions=lack-sched | less -R

    pub fn pnm(config: &[Sos<29>; 4], state: &mut [SosState; 4], xy0: &mut [i32; 8]) {
        config.inplace(state, xy0);
    }

    // ~20 insn/sample/sos on skylake, >200 MS/s
    #[test]
    #[ignore]
    fn sos_insn() {
        let cfg = [
            [[1., 3., 5.], [19., -9., 9.]],
            [[3., 3., 5.], [21., -11., 11.]],
            [[1., 3., 5.], [55., -17., 17.]],
            [[1., 8., 5.], [77., -7., 7.]],
        ]
        .map(|c| Sos::from(&c));
        let mut state = Default::default();
        let mut x = [977371917; 1 << 6];
        for _ in 0..1 << 20 {
            for x in x.as_chunks_mut().0 {
                pnm(&cfg, &mut state, x);
            }
            x[9] = x[63];
        }
    }
}
