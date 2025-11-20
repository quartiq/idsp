use num_traits::Float as _;

/// Processing block
/// Single input, single output, one value
pub trait Process {
    /// Update the state with a new sample and obtain an output sample
    fn process(&mut self, x: i32) -> i32;

    /// Process a block of samples
    #[inline]
    fn process_block(&mut self, x: &[i32], y: &mut [i32]) {
        for (x, y) in x.iter().zip(y) {
            *y = self.process(*x);
        }
    }

    /// Process a block of samples inplace
    #[inline]
    fn process_in_place(&mut self, xy: &mut [i32]) {
        for xy in xy {
            *xy = self.process(*xy);
        }
    }
}

/// A cascade of several filter blocks of the same type
#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Cascade<T>(pub T);
impl<T, const N: usize> Process for Cascade<[T; N]>
where
    T: Process,
{
    fn process(&mut self, x: i32) -> i32 {
        self.0.iter_mut().fold(x, |x, filter| filter.process(x))
    }

    fn process_block(&mut self, x: &[i32], y: &mut [i32]) {
        if let Some(filter) = self.0.first_mut() {
            filter.process_block(x, y);
        }
        for filter in self.0.iter_mut().skip(1) {
            filter.process_in_place(y);
        }
    }

    fn process_in_place(&mut self, xy: &mut [i32]) {
        for filter in self.0.iter_mut() {
            filter.process_in_place(xy);
        }
    }
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

/// Second-order-section with offset and clamp
#[derive(Clone, Debug)]
pub struct SosClamp<const Q: u8> {
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
pub struct State {
    /// X,Y state
    ///
    /// `[x1, x2, y1, y2]`
    pub xy: [i32; 4],
}

/// SOS state with wide Y
#[derive(Clone, Debug, Default)]
pub struct StateWide {
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
pub struct StateDither {
    /// X,Y state
    ///
    /// `[x1, x2, y1, y2]`
    pub xy: [i32; 4],
    /// Error feedback
    pub e: u32,
}

/// A stateful processor
#[derive(Debug, Clone, Default)]
pub struct Stateful<F, S>(pub F, pub S);

/// Stateful processor by reference
#[derive(Debug)]
pub struct StatefulRef<'a, F, S>(pub &'a F, pub &'a mut S);

impl<F, S> Process for Stateful<F, S>
where
    for<'a> StatefulRef<'a, F, S>: Process,
{
    #[inline]
    fn process(&mut self, x: i32) -> i32 {
        StatefulRef(&self.0, &mut self.1).process(x)
    }

    #[inline]
    fn process_block(&mut self, x: &[i32], y: &mut [i32]) {
        StatefulRef(&self.0, &mut self.1).process_block(x, y)
    }

    #[inline]
    fn process_in_place(&mut self, xy: &mut [i32]) {
        StatefulRef(&self.0, &mut self.1).process_in_place(xy)
    }
}

impl<const Q: u8> Process for StatefulRef<'_, Sos<Q>, State> {
    fn process(&mut self, x0: i32) -> i32 {
        let xy = &mut self.1.xy;
        let ba = &self.0.ba;
        let mut acc = 0;
        acc += x0 as i64 * ba[0] as i64;
        acc += xy[0] as i64 * ba[1] as i64;
        acc += xy[1] as i64 * ba[2] as i64;
        acc += xy[2] as i64 * ba[3] as i64;
        acc += xy[3] as i64 * ba[4] as i64;
        let y0 = (acc >> Q) as i32;
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const Q: u8> Process for StatefulRef<'_, SosClamp<Q>, State> {
    fn process(&mut self, x0: i32) -> i32 {
        let xy = &mut self.1.xy;
        let ba = &self.0.ba;
        let mut acc = 0;
        acc += x0 as i64 * ba[0] as i64;
        acc += xy[0] as i64 * ba[1] as i64;
        acc += xy[1] as i64 * ba[2] as i64;
        acc += xy[2] as i64 * ba[3] as i64;
        acc += xy[3] as i64 * ba[4] as i64;
        let mut y0 = (acc >> Q) as i32 + self.0.u;
        if y0 < self.0.min {
            y0 = self.0.min;
        } else if y0 > self.0.max {
            y0 = self.0.max;
        }
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const Q: u8> Process for StatefulRef<'_, Sos<Q>, StateWide> {
    fn process(&mut self, x0: i32) -> i32 {
        let x = &mut self.1.x;
        let y = &mut self.1.y;
        let ba = &self.0.ba;
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
        *y = [acc, y[0]];
        (acc >> 32) as _
    }
}

impl<const Q: u8> Process for StatefulRef<'_, SosClamp<Q>, StateWide> {
    fn process(&mut self, x0: i32) -> i32 {
        let x = &mut self.1.x;
        let y = &mut self.1.y;
        let ba = &self.0.ba;
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
        let mut y0 = (acc >> 32) as i32 + self.0.u;
        if y0 < self.0.min {
            y0 = self.0.min;
        } else if y0 > self.0.max {
            y0 = self.0.max;
        }
        *y = [((y0 as i64) << 32) | acc as u32 as i64, y[0]];
        y0
    }
}

impl<const Q: u8> Process for StatefulRef<'_, Sos<Q>, StateDither> {
    fn process(&mut self, x0: i32) -> i32 {
        let xy = &mut self.1.xy;
        let e = &mut self.1.e;
        let ba = &self.0.ba;
        let mut acc = *e as i64;
        acc += x0 as i64 * ba[0] as i64;
        acc += xy[0] as i64 * ba[1] as i64;
        acc += xy[1] as i64 * ba[2] as i64;
        acc += xy[2] as i64 * ba[3] as i64;
        acc += xy[3] as i64 * ba[4] as i64;
        acc <<= 32 - Q;
        *e = acc as _;
        let y0 = (acc >> 32) as i32;
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const Q: u8> Process for StatefulRef<'_, SosClamp<Q>, StateDither> {
    fn process(&mut self, x0: i32) -> i32 {
        let xy = &mut self.1.xy;
        let e = &mut self.1.e;
        let ba = &self.0.ba;
        let mut acc = *e as i64;
        acc += x0 as i64 * ba[0] as i64;
        acc += xy[0] as i64 * ba[1] as i64;
        acc += xy[1] as i64 * ba[2] as i64;
        acc += xy[2] as i64 * ba[3] as i64;
        acc += xy[3] as i64 * ba[4] as i64;
        acc <<= 32 - Q;
        *e = acc as _;
        let mut y0 = (acc >> 32) as i32 + self.0.u;
        if y0 < self.0.min {
            y0 = self.0.min;
        } else if y0 > self.0.max {
            y0 = self.0.max;
        }
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

// No manual tuning needed here.
// Compiler knows best how and when:
//   unroll loops
//   cache on stack
//   handle alignment
//   register allocate variables
//   manage pipeline and insn issue

// cargo asm idsp::iir::sos::pnm --rust --target thumbv7em-none-eabihf --lib --target-cpu cortex-m7 --color --mca -M=-iterations=1 -M=-timeline -M=-skip-unsupported-instructions=lack-sched | less -R

// pub fn pnm(biquad: &[Sos<29>; 4], state: &mut [State; 4], xy0: &mut [i32; 8]) {
//     for p in biquad.iter().zip(state) {
//         Processor(p.0, p.1).process_in_place(xy0);
//     }
//     // pub fn pnm(biquad: &mut [(Sos<29>, State); 4], xy0: &mut [i32; 8]) {
//     //     for (biquad, state) in biquad.iter_mut() {
//     //     for xy0 in xy0.iter_mut() {
//     //         *xy0 = biquad.process(state, *xy0);
//     //     }
//     // }
// }

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
