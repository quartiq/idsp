use serde::{Deserialize, Serialize};

/// Generic vector for integer IIR filter.
/// This struct is used to hold the x/y input/output data vector or the b/a coefficient
/// vector.
///
/// Integer biquad IIR
///
/// See [`crate::iir::Biquad`] for general implementation details.
/// Offset and limiting disabled to suit lowpass applications.
/// Coefficient scaling fixed and optimized such that -2 is representable.
/// Tailored to low-passes, PID, II etc, where the integration rule is [1, -2, 1].
/// Since the relevant coefficients `a1` and `a2` are negated, we also negate the
/// stored `y1` and `y2` in the state.
/// Note that `xy` contains the negative `y1` and `y2`, such that `-a1`
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd)]
pub struct Biquad {
    ba: [i32; 5],
    u: i32,
    min: i32,
    max: i32,
}

impl Default for Biquad {
    fn default() -> Self {
        Self {
            ba: [0; 5],
            u: 0,
            // Due to the negation of the stored `y1`, `y2`
            // values, we need to avoid `i32::MIN`.
            min: -i32::MAX,
            max: i32::MAX,
        }
    }
}

impl From<[i32; 5]> for Biquad {
    fn from(value: [i32; 5]) -> Self {
        Self {
            ba: value,
            ..Default::default()
        }
    }
}

/// A filter with the given proportional gain at all frequencies
///
/// ```
/// # use idsp::iir_int::*;
/// let x0 = 3;
/// let k = 5;
/// let y0 = Biquad::from(proportional(k << 20)).update(&mut [0; 5], x0 << 20);
/// assert_eq!(y0, (x0 * k) << (20 + 20 - Biquad::SHIFT));
/// ```
pub fn proportional(k: i32) -> [i32; 5] {
    [k, 0, 0, 0, 0]
}

/// A unit gain filter
///
/// ```
/// # use idsp::iir_int::*;
/// let x0 = 3;
/// let y0 = Biquad::from(identity()).update(&mut [0; 5], x0);
/// assert_eq!(y0, x0);
/// ```
pub fn identity() -> [i32; 5] {
    proportional(Biquad::ONE)
}

/// A "hold" filter that ingests input and maintains output
///
/// ```
/// # use idsp::iir_int::*;
/// let mut xy = [0, 1, 2, 3, 4];
/// let x0 = 7;
/// let y0 = Biquad::from(hold()).update(&mut xy, x0);
/// assert_eq!(y0, -2);
/// assert_eq!(xy, [x0, 0, -y0, -y0, 3]);
/// ```
pub fn hold() -> [i32; 5] {
    [0, 0, 0, -Biquad::ONE, 0]
}

/// Lowpass biquad filter using cutoff and sampling frequencies.  Taken from:
/// https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
///
/// # Args
/// * `f` - Corner frequency, or 3dB cutoff frequency (in units of sample rate).
///         This is only accurate for low corner frequencies less than ~0.01.
/// * `q` - Quality factor (1/sqrt(2) for critical).
/// * `k` - DC gain.
///
/// # Returns
/// Biquad IIR filter.
pub fn lowpass(f: f64, q: f64, k: f64) -> [i32; 5] {
    // 3rd order Taylor approximation of sin and cos.
    let f = f * core::f64::consts::TAU;
    let f2 = f * f * 0.5;
    let fcos = 1. - f2;
    let fsin = f * (1. - f2 / 3.);
    let alpha = fsin / (2. * q);
    let a0 = Biquad::ONE as f64 / (1. + alpha);
    let b = (k / 2. * (1. - fcos) * a0 + 0.5) as i32;
    let a1 = (2. * fcos * a0 + 0.5) as i32;
    let a2 = ((alpha - 1.) * a0 + 0.5) as i32;

    [b, 2 * b, b, -a1, -a2]
}

impl Biquad {
    /// Filter coefficients
    ///
    /// IIR filter tap gains (`ba`) are an array `[b0, b1, b2, -a1, -a2]` such that
    /// [`Biquad::update(&mut xy, x0)`] with `xy = [x1, x2, -y1, -y2, -y3]` returns
    /// `y0 = clamp(b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2 + u, min, max)`.
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// assert_eq!(Biquad::from(identity()).ba()[0], Biquad::ONE);
    /// assert_eq!(Biquad::from(hold()).ba()[3], -Biquad::ONE);
    /// ```
    pub fn ba(&self) -> &[i32; 5] {
        &self.ba
    }

    /// Mutable reference to the filter coefficients.
    ///
    /// See [`Biquad::ba()`].
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// let mut i = Biquad::default();
    /// i.ba_mut()[0] = Biquad::ONE;
    /// assert_eq!(i, Biquad::from(identity()));
    /// ```
    pub fn ba_mut(&mut self) -> &mut [i32; 5] {
        &mut self.ba
    }

    /// Summing junction offset
    ///
    /// This offset is applied to the output `y0` summing junction
    /// on top of the feed-forward (`b`) and feed-back (`a`) terms.
    /// The feedback samples are taken at the summing junction and
    /// thus also include (and feed back) this offset.
    pub fn u(&self) -> i32 {
        self.u
    }

    /// Set the summing junction offset
    ///
    /// See [`Biquad::u()`].
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// let mut i = Biquad::default();
    /// i.set_u(5);
    /// assert_eq!(i.update(&mut [0; 5], 0), 5);
    /// ```
    pub fn set_u(&mut self, u: i32) {
        self.u = u;
    }

    /// Lower output limit
    ///
    /// Guaranteed minimum output value.
    /// The value is inclusive.
    /// The clamping also cleanly affects the feedback terms.
    ///
    /// Note: `i32::MIN` should not be passed as the `y` samples stored in
    /// the filter state are negated. Instead use `-i32::MAX` as the lowest
    /// possible limit.
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// assert_eq!(Biquad::default().min(), -i32::MAX);
    /// ```
    pub fn min(&self) -> i32 {
        self.min
    }

    /// Set the lower output limit
    ///
    /// See [`Biquad::min()`].
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// let mut i = Biquad::default();
    /// i.set_min(5);
    /// assert_eq!(i.update(&mut [0; 5], 0), 5);
    /// ```
    pub fn set_min(&mut self, min: i32) {
        self.min = min;
    }

    /// Upper output limit
    ///
    /// Guaranteed maximum output value.
    /// The value is inclusive.
    /// The clamping also cleanly affects the feedback terms.
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// assert_eq!(Biquad::default().max(), i32::MAX);
    /// ```
    pub fn max(&self) -> i32 {
        self.max
    }

    /// Set the upper output limit
    ///
    /// See [`Biquad::max()`].
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// let mut i = Biquad::default();
    /// i.set_max(-5);
    /// assert_eq!(i.update(&mut [0; 5], 0), -5);
    /// ```
    pub fn set_max(&mut self, max: i32) {
        self.max = max;
    }

    /// Compute the overall (DC/proportional feed-forward) gain.
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// assert_eq!(Biquad::from(proportional(3)).forward_gain(), 3);
    /// ```
    ///
    /// # Returns
    /// The sum of the `b` feed-forward coefficients.
    pub fn forward_gain(&self) -> i32 {
        self.ba.iter().take(3).sum()
    }

    /// Compute input-referred (`x`) offset.
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// let mut i = Biquad::from(proportional(3));
    /// i.set_u(3);
    /// assert_eq!(i.input_offset(), Biquad::ONE);
    /// ```
    pub fn input_offset(&self) -> i32 {
        (((self.u as i64) << Self::SHIFT) / self.forward_gain() as i64) as i32
    }

    /// Convert input (`x`) offset to equivalent summing junction offset (`u`) and apply.
    ///
    /// In the case of a "PID" controller the response behavior of the controller
    /// to the offset is "stabilizing", and not "tracking": its frequency response
    /// is exclusively according to the lowest non-zero [`PidAction`] gain.
    /// There is no high order ("faster") response as would be the case for a "tracking"
    /// controller.
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// let mut i = Biquad::from(proportional(5 << 20));
    /// i.set_input_offset(3 << 20);
    /// assert_eq!(i.u(), 15 << (20 + 20 - Biquad::SHIFT));
    /// ```
    ///
    /// # Arguments
    /// * `offset`: Input (`x`) offset.
    pub fn set_input_offset(&mut self, offset: i32) {
        self.u = (((1 << (Self::SHIFT - 1)) + offset as i64 * self.forward_gain() as i64)
            >> Self::SHIFT) as i32;
    }

    /// Coefficient fixed point format: signed Q2.30
    pub const SHIFT: u32 = 30;
    pub const ONE: i32 = 1 << Self::SHIFT;

    /// Feed a new input value into the filter, update the filter state, and
    /// return the new output. Only the state `xy` is modified.
    ///
    /// ```
    /// # use idsp::iir_int::*;
    /// let i = Biquad::from(identity());
    /// let mut xy = [0, 1, 2, 3, 4];
    /// let x0 = 5;
    /// let y0 = i.update(&mut xy, x0);
    /// assert_eq!(y0, x0);
    /// assert_eq!(xy, [x0, 0, -y0, 2, 3]);
    /// ```
    ///
    /// # Arguments
    /// * `xy` - Current filter state.
    ///   On entry: `[x1, x2, -y1, -y2, -y3]`
    ///   On exit:  `[x0, x1, -y0, -y1, -y2]`
    /// * `x0` - New input.
    ///
    /// # Returns
    /// The new output `y0 = clamp(b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2 + u, min, max)`
    pub fn update(&self, xy: &mut [i32; 5], x0: i32) -> i32 {
        // `xy` contains       x0 x1 -y0 -y1 -y2
        // Increment time      x1 x2 -y1 -y2 -y3
        // Shift               x1 x1  x2 -y1 -y2
        // This unrolls better than xy.rotate_right(1)
        xy.copy_within(0..4, 1);
        // Store x0            x0 x1  x2 -y1 -y2
        xy[0] = x0;
        // Compute y0 by multiply-accumulate
        let y0 = (xy
            .iter()
            .zip(self.ba.iter())
            .fold(1 << (Self::SHIFT - 1), |y0, (xy, ba)| {
                y0 + *xy as i64 * *ba as i64
            })
            >> Self::SHIFT) as i32
            + self.u;
        // Limit y0
        let y0 = y0.clamp(self.min, self.max);
        // Store y0            x0 x1 -y0 -y1 -y2
        xy[2] = -y0;
        y0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lowpass_gen() {
        let ba = Biquad::from(lowpass(5e-6, 0.5f64.sqrt(), 2.));
        println!("{:?}", ba);
    }
}
