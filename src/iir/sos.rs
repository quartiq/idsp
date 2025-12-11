use super::{Process, ProcessorRef};
use miniconf::Tree;
#[cfg(not(feature = "std"))]
use num_traits::float::FloatCore as _;

/// Second-order-section
#[derive(Clone, Debug, Default, Tree)]
#[tree(meta(doc, typename))]
pub struct Sos<const Q: u8> {
    /// Coefficients
    ///
    /// `[b0, b1, b2, a1, a2]`
    /// Such that:
    /// `y0 = (b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2)/(1 << Q)`
    ///
    /// Note the a1, a2 sign:
    /// `H = (b0 + b1*z^-1 + b2*z^-2)/((1 << Q) - a2*z^-1 - a2*z^-2)`
    #[tree(with=miniconf::leaf)]
    pub ba: [i32; 5],
}

/// Second-order-section with offset and clamp
#[derive(Clone, Debug, Tree)]
#[tree(meta(doc, typename))]
pub struct SosClamp<const Q: u8> {
    /// Coefficients
    ///
    /// `[b0, b1, b2, a1, a2]`
    /// Such that:
    /// `y0 = (b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2)/(1 << Q)`
    ///
    /// Note the a1, a2 sign:
    /// `H = (b0 + b1*z^-1 + b2*z^-2)/((1 << Q) - a2*z^-1 - a2*z^-2)`
    #[tree(with=miniconf::leaf)]
    pub ba: [i32; 5],
    /// Summing junction offset
    pub u: i32,
    /// Summing junction min clamp
    pub min: i32,
    /// Summing junction min clamp
    pub max: i32,
}

impl<const Q: u8> Default for SosClamp<Q> {
    fn default() -> Self {
        Self {
            ba: Default::default(),
            u: 0,
            min: i32::MIN,
            max: i32::MAX,
        }
    }
}

impl<const Q: u8> SosClamp<Q> {
    /// DC gain
    pub fn k(&self) -> i32 {
        self.ba[..3].iter().sum::<i32>()
    }

    /// Summing junction offset referred to input
    pub fn input_offset(&self) -> i32 {
        (((self.u as i64) << Q) / self.k() as i64) as _
    }

    /// Summing junction offset referred to input
    pub fn set_input_offset(&mut self, i: i32) {
        self.u = ((i as i64 * self.k() as i64) >> Q) as _;
    }
}

/// SOS state
#[derive(Clone, Debug, Default)]
pub struct SosState {
    /// X,Y state
    ///
    /// `[x1, x2, y1, y2]`
    pub xy: [i32; 4],
}

/// SOS state with wide Y
#[derive(Clone, Debug, Default)]
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
#[derive(Clone, Debug, Default)]
pub struct SosStateDither {
    /// X,Y state
    ///
    /// `[x1, x2, y1, y2]`
    pub xy: [i32; 4],
    /// Error feedback
    pub e: u32,
}

impl<const Q: u8> Process<i32> for ProcessorRef<'_, Sos<Q>, SosState> {
    fn process(&mut self, x0: &i32) -> i32 {
        let xy = &mut self.state.xy;
        let ba = &self.config.ba;
        let mut acc = 0;
        acc += *x0 as i64 * ba[0] as i64;
        acc += xy[0] as i64 * ba[1] as i64;
        acc += xy[1] as i64 * ba[2] as i64;
        acc += xy[2] as i64 * ba[3] as i64;
        acc += xy[3] as i64 * ba[4] as i64;
        let y0 = (acc >> Q) as i32;
        *xy = [*x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const Q: u8> Process<i32> for ProcessorRef<'_, SosClamp<Q>, SosState> {
    fn process(&mut self, x0: &i32) -> i32 {
        let xy = &mut self.state.xy;
        let ba = &self.config.ba;
        let mut acc = 0;
        acc += *x0 as i64 * ba[0] as i64;
        acc += xy[0] as i64 * ba[1] as i64;
        acc += xy[1] as i64 * ba[2] as i64;
        acc += xy[2] as i64 * ba[3] as i64;
        acc += xy[3] as i64 * ba[4] as i64;
        let mut y0 = (acc >> Q) as i32 + self.config.u;
        // clamp() is slower
        if y0 < self.config.min {
            y0 = self.config.min;
        } else if y0 > self.config.max {
            y0 = self.config.max;
        }
        *xy = [*x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const Q: u8> Process<i32> for ProcessorRef<'_, Sos<Q>, SosStateWide> {
    fn process(&mut self, x0: &i32) -> i32 {
        let x = &mut self.state.x;
        let y = &mut self.state.y;
        let ba = &self.config.ba;
        let mut acc = 0;
        acc += *x0 as i64 * ba[0] as i64;
        acc += x[0] as i64 * ba[1] as i64;
        acc += x[1] as i64 * ba[2] as i64;
        *x = [*x0, x[0]];
        acc += (y[0] as u32 as i64 * ba[3] as i64) >> 32;
        acc += (y[0] >> 32) * ba[3] as i64;
        acc += (y[1] as u32 as i64 * ba[4] as i64) >> 32;
        acc += (y[1] >> 32) * ba[4] as i64;
        acc <<= 32 - Q;
        *y = [acc, y[0]];
        (acc >> 32) as _
    }
}

impl<const Q: u8> Process<i32> for ProcessorRef<'_, SosClamp<Q>, SosStateWide> {
    fn process(&mut self, x0: &i32) -> i32 {
        let x = &mut self.state.x;
        let y = &mut self.state.y;
        let ba = &self.config.ba;
        let mut acc = 0;
        acc += *x0 as i64 * ba[0] as i64;
        acc += x[0] as i64 * ba[1] as i64;
        acc += x[1] as i64 * ba[2] as i64;
        *x = [*x0, x[0]];
        acc += (y[0] as u32 as i64 * ba[3] as i64) >> 32;
        acc += (y[0] >> 32) * ba[3] as i64;
        acc += (y[1] as u32 as i64 * ba[4] as i64) >> 32;
        acc += (y[1] >> 32) * ba[4] as i64;
        acc <<= 32 - Q;
        let mut y0 = (acc >> 32) as i32 + self.config.u;
        // clamp() is slower
        if y0 < self.config.min {
            y0 = self.config.min;
        } else if y0 > self.config.max {
            y0 = self.config.max;
        }
        *y = [((y0 as i64) << 32) | acc as u32 as i64, y[0]];
        y0
    }
}

impl<const Q: u8> Process<i32> for ProcessorRef<'_, Sos<Q>, SosStateDither> {
    fn process(&mut self, x0: &i32) -> i32 {
        let xy = &mut self.state.xy;
        let e = &mut self.state.e;
        let ba = &self.config.ba;
        let mut acc = *e as i64;
        acc += *x0 as i64 * ba[0] as i64;
        acc += xy[0] as i64 * ba[1] as i64;
        acc += xy[1] as i64 * ba[2] as i64;
        acc += xy[2] as i64 * ba[3] as i64;
        acc += xy[3] as i64 * ba[4] as i64;
        acc <<= 32 - Q;
        *e = acc as _;
        let y0 = (acc >> 32) as i32;
        *xy = [*x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const Q: u8> Process<i32> for ProcessorRef<'_, SosClamp<Q>, SosStateDither> {
    fn process(&mut self, x0: &i32) -> i32 {
        let xy = &mut self.state.xy;
        let e = &mut self.state.e;
        let ba = &self.config.ba;
        let mut acc = *e as i64;
        acc += *x0 as i64 * ba[0] as i64;
        acc += xy[0] as i64 * ba[1] as i64;
        acc += xy[1] as i64 * ba[2] as i64;
        acc += xy[2] as i64 * ba[3] as i64;
        acc += xy[3] as i64 * ba[4] as i64;
        acc <<= 32 - Q;
        *e = acc as _;
        let mut y0 = (acc >> 32) as i32 + self.config.u;
        if y0 < self.config.min {
            y0 = self.config.min;
        } else if y0 > self.config.max {
            y0 = self.config.max;
        }
        *xy = [*x0, xy[0], y0, xy[2]];
        y0
    }
}

fn quantize(ba: &[[f64; 3]; 2], q: u8) -> [i32; 5] {
    let a0 = (1u64 << q) as f64 / ba[1][0];
    [
        (ba[0][0] * a0).round() as i32,
        (ba[0][1] * a0).round() as i32,
        (ba[0][2] * a0).round() as i32,
        (ba[1][1] * a0).round() as i32,
        (ba[1][2] * a0).round() as i32,
    ]
}

impl<const Q: u8> From<&[[f64; 3]; 2]> for Sos<Q> {
    fn from(ba: &[[f64; 3]; 2]) -> Self {
        quantize(ba, Q).into()
    }
}

impl<const Q: u8> From<&[[f64; 3]; 2]> for SosClamp<Q> {
    fn from(ba: &[[f64; 3]; 2]) -> Self {
        quantize(ba, Q).into()
    }
}

impl<const Q: u8> From<[i32; 5]> for Sos<Q> {
    fn from(mut ba: [i32; 5]) -> Self {
        ba[3] *= -1;
        ba[4] *= -1;
        Self { ba }
    }
}

impl<const Q: u8> From<[i32; 5]> for SosClamp<Q> {
    fn from(mut ba: [i32; 5]) -> Self {
        ba[3] *= -1;
        ba[4] *= -1;
        Self {
            ba,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(dead_code)]
    use super::*;
    // No manual tuning needed here.
    // Compiler knows best how and when:
    //   unroll loops
    //   cache on stack
    //   handle alignment
    //   register allocate variables
    //   manage pipeline and insn issue

    // cargo asm idsp::iir::sos::pnm --rust --target thumbv7em-none-eabihf --lib --target-cpu cortex-m7 --color --mca -M=-iterations=1 -M=-timeline -M=-skip-unsupported-instructions=lack-sched | less -R

    pub struct Casc([Sos<29>; 4]);
    impl Casc {
        pub fn block(&self, state: &mut [SosState; 4], xy0: &mut [i32; 8]) {
            ProcessorRef::new(&self.0, state).in_place(xy0);
        }
    }
}
