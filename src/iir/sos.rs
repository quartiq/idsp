/// Processing block
/// Single input, single output, one value
pub trait Block {
    /// The operating state
    type State;
    /// Update the state with a new sample and obtain an output sample
    fn process(&self, state: &mut Self::State, x0: i32) -> i32;
}

/// Second-order-section
#[derive(Clone, Debug, Default)]
pub struct Sos<const Q: u8> {
    /// Coefficients
    ///
    /// `[b0, b1, b2, a1, a2]`
    /// Such that:
    /// `y0 = (b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2)/(1 << Q)`
    ///
    /// Note the a1, a2 sign:
    /// `H = (b0 + b1*z^-1 + b2*z^-2)/((1 << Q) - a2*z^-1 - a2*z^-2)`
    pub ba: [i32; 5],
}

/// SOS state
#[derive(Clone, Debug, Default)]
pub struct State {
    /// X,Y state
    ///
    /// `[x1, x2, y1, y2]`
    pub xy: [i32; 4],
}

impl<const Q: u8> Block for Sos<Q> {
    type State = State;

    fn process(&self, state: &mut Self::State, x0: i32) -> i32 {
        let mut acc = 0;
        acc += x0 as i64 * self.ba[0] as i64;
        acc += state.xy[0] as i64 * self.ba[1] as i64;
        acc += state.xy[1] as i64 * self.ba[2] as i64;
        // no subtraction/negation for performance, apply that to coefficients
        acc += state.xy[2] as i64 * self.ba[3] as i64;
        acc += state.xy[3] as i64 * self.ba[4] as i64;
        let y0 = (acc >> Q) as i32;
        state.xy = [x0, state.xy[0], y0, state.xy[2]];
        y0
    }
}

/// Second-order-section
#[derive(Clone, Debug, Default)]
pub struct SosR<const Q: u8> {
    /// Coefficients
    ///
    /// `[b0, b1, b2, a1, a2]`
    /// Such that:
    /// `y0 = (b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2)/(1 << Q)`
    ///
    /// Note the a1, a2 sign:
    /// `H = (b0 + b1*z^-1 + b2*z^-2)/((1 << Q) - a2*z^-1 - a2*z^-2)`
    pub ba: [i32; 5],
    /// Summing junction offset
    pub u: i32,
    /// Output min clamp
    pub min: i32,
    /// Output min clamp
    pub max: i32,
}

/// SOS state
#[derive(Clone, Debug, Default)]
pub struct StateR {
    /// X state
    ///
    /// `[x1, x2]`
    pub x: [i32; 2],
    /// Y state
    ///
    /// `[y1, y2]``
    pub y: [i64; 2],
}

impl<const Q: u8> Block for SosR<Q> {
    type State = StateR;

    fn process(&self, state: &mut Self::State, x0: i32) -> i32 {
        let StateR { x, y } = state;
        let ba = &self.ba;
        let mut acc = 0;
        acc += x0 as i64 * ba[0] as i64;
        acc += x[0] as i64 * ba[1] as i64;
        acc += x[1] as i64 * ba[2] as i64;
        *x = [x0, x[0]];
        acc += (y[0] as u32 as i64 * ba[3] as i64) >> 32;
        acc += (y[0] >> 32) * ba[3] as i64;
        acc += (y[1] as u32 as i64 * ba[4] as i64) >> 32;
        acc += (y[1] >> 32) * ba[4] as i64;
        acc <<= 32 - Q;
        let mut y0 = (acc >> 32) as i32 + self.u;
        if y0 < self.min {
            y0 = self.min;
        } else if y0 > self.max {
            y0 = self.max;
        }
        *y = [((y0 as i64) << 32) | acc as u32 as i64, y[0]];
        y0
    }
}

// pub fn pnmr(biquad: &mut [(SosR<29>, StateR); 4], xy0: &mut [i32; 8]) {
//     for (biquad, state) in biquad.iter_mut() {
//         for xy0 in xy0.iter_mut() {
//             *xy0 = biquad.process(state, *xy0);
//         }
//     }
// }

// No manual tuning needed here.
// Compiler knows best how and when:
//   unroll loops
//   cache on stack
//   handle alignment
//   register allocate variables
//   manage pipeline and insn issue

// cargo asm idsp::iir::sos::pnm --rust --target thumbv7em-none-eabihf --lib --target-cpu cortex-m7 --color --mca -M=-iterations=1 -M=-timeline -M=-skip-unsupported-instructions=lack-sched | less -R

//pub fn pnm(biquad: &[Sos<29>; 4], state: &mut [State; 4], xy0: &mut [i32; 8]) {
//    for (biquad, state) in biquad.iter().zip(state) {
// pub fn pnm(biquad: &mut [(Sos<29>, State); 4], xy0: &mut [i32; 8]) {
//     for (biquad, state) in biquad.iter_mut() {
//         for xy0 in xy0.iter_mut() {
//             *xy0 = biquad.process(state, *xy0);
//         }
//     }
// }
