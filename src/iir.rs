use serde::{Deserialize, Serialize};

use num_traits::{clamp, Float};

/// # Coefficients and state
///
/// `[T; 5]` is both the IIR state and coefficients type.
///
/// To represent the IIR state (input and output memory) during [`Biquad::update()`]
/// this contains the three inputs `[x0, x1, x2]` and the two outputs `[y1, y2]`
/// concatenated. Lower indices correspond to more recent samples.
/// To represent the IIR coefficients, this contains the feed-forward
/// coefficients `[b0, b1, b2]` followd by the negated feed-back coefficients
/// `[a1, a2]`, all five normalized such that `a0 = 1`.
/// Note that between filter [`Biquad::update()`] the `xy` state contains
/// `[x0, x1, y0, y1, y2]`.
///
/// The IIR coefficients can be mapped to other transfer function
/// representations, for example as described in <https://arxiv.org/abs/1508.06319>
///
/// # IIR filter as PID controller
///
/// Contains the coeeficients `ba`, the summing junction offset `u`, and the
/// output limits `min` and `max`. Data is represented in floating-point
/// for all internal signals, input and output.
///
/// This implementation achieves several important properties:
///
/// * Its transfer function is universal in the sense that any biquadratic
///   transfer function can be implemented (high-passes, gain limits, second
///   order integrators with inherent anti-windup, notches etc) without code
///   changes preserving all features.
/// * It inherits a universal implementation of "integrator anti-windup", also
///   and especially in the presence of set-point changes and in the presence
///   of proportional or derivative gain without any back-off that would reduce
///   steady-state output range.
/// * It has universal derivative-kick (undesired, unlimited, and un-physical
///   amplification of set-point changes by the derivative term) avoidance.
/// * An offset at the input of an IIR filter (a.k.a. "set-point") is
///   equivalent to an offset at the summing junction (in output units).
///   They are related by the overall (DC feed-forward) gain of the filter.
/// * It stores only previous outputs and inputs. These have direct and
///   invariant interpretation (independent of coefficients and offset).
///   Therefore it can trivially implement bump-less transfer between any
///   coefficients/offset sets.
/// * Cascading multiple IIR filters allows stable and robust
///   implementation of transfer functions beyond bequadratic terms.
///
/// See also <https://hackmd.io/IACbwcOTSt6Adj3_F9bKuw>.
#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Biquad<T> {
    ba: [T; 5],
    u: T,
    min: T,
    max: T,
}

impl<T: Float> Default for Biquad<T> {
    fn default() -> Self {
        Self {
            ba: [T::zero(); 5],
            u: T::zero(),
            min: T::neg_infinity(),
            max: T::infinity(),
        }
    }
}

impl<T: Float> From<[T; 5]> for Biquad<T> {
    fn from(ba: [T; 5]) -> Self {
        Self {
            ba,
            ..Default::default()
        }
    }
}

impl<T: Float> Biquad<T> {
    /// Filter coefficients
    ///
    /// IIR filter tap gains (`ba`) are an array `[b0, b1, b2, a1, a2]` such that
    /// `y0 = clamp(b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2 + u, min, max)`.
    pub fn ba(&self) -> &[T; 5] {
        &self.ba
    }

    /// Mutable reference to the filter coefficients.
    ///
    /// See [`Biquad::ba()`].
    pub fn ba_mut(&mut self) -> &mut [T; 5] {
        &mut self.ba
    }

    /// Summing junction offset
    ///
    /// This offset is applied to the output `y0` summing junction
    /// on top of the feed-forward (`b`) and feed-back (`a`) terms.
    /// The feedback samples are taken at the summing junction and
    /// thus also include (and feed back) this offset.
    pub fn u(&self) -> T {
        self.u
    }

    /// Set the summing junction offset
    ///
    /// See [`Biquad::u()`].
    pub fn set_u(&mut self, u: T) {
        self.u = u;
    }

    /// Lower output limit
    ///
    /// Guaranteed minimum output value.
    /// The value is inclusive.
    /// The clamping also cleanly affects the feedback terms.
    pub fn min(&self) -> T {
        self.min
    }

    /// Set the lower output limit
    ///
    /// See [`Biquad::min()`].
    pub fn set_min(&mut self, min: T) {
        self.max = min;
    }

    /// Upper output limit
    ///
    /// Guaranteed maximum output value.
    /// The value is inclusive.
    /// The clamping also cleanly affects the feedback terms.
    pub fn max(&self) -> T {
        self.max
    }

    /// Set the upper output limit
    ///
    /// See [`Biquad::max()`].
    pub fn set_max(&mut self, max: T) {
        self.max = max;
    }

    /// A unit gain filter
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let x0 = 3.0;
    /// let y0 = Biquad::identity().update(&mut [0.0; 5], x0);
    /// assert_eq!(y0, x0);
    /// ```
    pub fn identity() -> Self {
        Self::proportional(T::one())
    }

    /// A filter with the given proportional gain at all frequencies
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let x0 = 2.0;
    /// let k = 5.0;
    /// let y0 = Biquad::proportional(k).update(&mut [0.0; 5], x0);
    /// assert_eq!(y0, x0 * k);
    /// ```
    pub fn proportional(k: T) -> Self {
        let mut s = Self::default();
        s.ba[0] = k;
        s
    }

    /// A "hold" filter that ingests input and maintains output
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = core::array::from_fn(|i| i as _);
    /// let x0 = 7.0;
    /// let y0 = Biquad::hold().update(&mut xy, x0);
    /// assert_eq!(y0, 2.0);
    /// assert_eq!(xy, [x0, 0.0, y0, y0, 3.0]);
    /// ```
    pub fn hold() -> Self {
        let mut s = Self::default();
        s.ba[3] = T::one();
        s
    }

    // TODO
    // lowpass1
    // highpass1
    // butterworth
    // elliptic
    // chebychev1/2
    // bessel
    // invert // high-to-low/low-to-high
    // notch
    // PI-notch
    // SOS cascades thereoff

    /// Return a builder for a "PID" controller
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let i = Biquad::<f32>::pid().build().unwrap();
    /// assert_eq!(i, Biquad::default());
    /// ```
    pub fn pid() -> PidBuilder<T> {
        PidBuilder::default()
    }

    /// Compute the overall (DC/proportional feed-forward) gain.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// assert_eq!(Biquad::proportional(3.0).forward_gain(), 3.0);
    /// ```
    ///
    /// # Returns
    /// The sum of the `b` feed-forward coefficients.
    pub fn forward_gain(&self) -> T {
        self.ba[0] + self.ba[1] + self.ba[2]
    }

    /// Compute input-referred (`x`) offset.
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::proportional(3.0);
    /// i.set_input_offset(2.0);
    /// assert_eq!(i.input_offset(), 2.0);
    /// ```
    pub fn input_offset(&self) -> T {
        self.u / self.forward_gain()
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
    /// # use idsp::iir::*;
    /// let mut i = Biquad::proportional(3.0);
    /// i.set_input_offset(2.0);
    /// assert_eq!(i.input_offset(), 2.0);
    /// let x0 = 0.5;
    /// let y0 = i.update(&mut [0.0; 5], x0);
    /// assert_eq!(y0, (x0 + i.input_offset()) * i.forward_gain());
    /// ```
    ///
    /// # Arguments
    /// * `offset`: Input (`x`) offset.
    pub fn set_input_offset(&mut self, offset: T) {
        self.u = offset * self.forward_gain();
    }

    /// Ingest a new input value into the filter, update the filter state, and
    /// return the new output. Only the state `xy` is modified.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = core::array::from_fn(|i| i as _);
    /// let x0 = 3.0;
    /// let y0 = Biquad::identity().update(&mut xy, x0);
    /// assert_eq!(y0, x0);
    /// assert_eq!(xy, [x0, 0.0, y0, 2.0, 3.0]);
    /// ```
    ///
    /// # Arguments
    /// * `xy` - Current filter state.
    /// * `x0` - New input.
    ///
    /// # Returns
    /// The new output `y0`.
    ///
    /// # Panics
    /// Panics in debug mode if `!(self.min <= self.max)`.
    pub fn update(&self, xy: &mut [T; 5], x0: T) -> T {
        // `xy` contains    x0 x1 y0 y1 y2
        // Increment time   x1 x2 y1 y2 y3
        // Shift            x1 x1 x2 y1 y2
        xy.copy_within(0..4, 1);
        // Store x0         x0 x1 x2 y1 y2
        xy[0] = x0;
        let y0 = xy
            .iter()
            .zip(self.ba.iter())
            .fold(self.u, |y, (x, a)| y + *x * *a);
        let y0 = clamp(y0, self.min, self.max);
        // Store y0         x0 x1 y0 y1 y2
        xy[2] = y0;
        y0
    }
}

/// PID controller builder
///
/// Builds `Biquad` from action gains, gain limits, input offset and output limits.
///
/// ```
/// # use idsp::iir::*;
/// let b = Biquad::pid()
///     .period(1e-3)
///     .gain(PidAction::Ki, 1e-3)
///     .gain(PidAction::Kp, 1.0)
///     .gain(PidAction::Kd, 1e2)
///     .limit(PidAction::Ki, 1e3)
///     .limit(PidAction::Kd, 1e1)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PidBuilder<T> {
    period: T,
    gains: [T; 5],
    limits: [T; 5],
}

impl<T: Float> Default for PidBuilder<T> {
    fn default() -> Self {
        Self {
            period: T::one(),
            gains: [T::zero(); 5],
            limits: [T::infinity(); 5],
        }
    }
}

/// [`PidBuilder::build()`] errors
#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PidError {
    /// The action gains cover more than three successive orders
    OrderRange,
}

/// PID action
///
/// This enumerates the five possible PID style actions of a [`Biquad`]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum PidAction {
    /// Double integrating, -40 dB per decade
    Kii = 0,
    /// Integrating, -20 dB per decade
    Ki = 1,
    /// Proportional
    Kp = 2,
    /// Derivative=, 20 dB per decade
    Kd = 3,
    /// Double derivative, 40 dB per decade
    Kdd = 4,
}

impl<T: Float> PidBuilder<T> {
    /// Sample period
    ///
    /// # Arguments
    /// * `period`: Sample period in some units, e.g. SI seconds
    pub fn period(mut self, period: T) -> Self {
        self.period = period;
        self
    }

    /// Gain for a given action
    ///
    /// Gain units are `output/input * time.powi(order)` where
    /// * `output` are output (`y`) units
    /// * `input` are input (`x`) units
    /// * `time` are sample period units, e.g. SI seconds
    /// * `order` is the action order: the frequency exponent
    ///    (`-1` for integrating, `0` for proportional, etc.)
    ///
    /// Note that inverse time units correspond to angular frequency units.
    /// Gains are accurate in the low frequency limit. Towards Nyquist, the
    /// frequency response is wrapped.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let tau = 1e-3;
    /// let ki = 1e-4;
    /// let i = Biquad::pid()
    ///     .period(tau)
    ///     .gain(PidAction::Ki, ki)
    ///     .build()
    ///     .unwrap();
    /// let x0 = 5.0;
    /// let y0 = i.update(&mut [0.0; 5], x0);
    /// assert_eq!(y0, x0 * ki / tau);
    /// ```
    ///
    /// # Arguments
    /// * `action`: Action to control
    /// * `gain`: Gain value
    pub fn gain(mut self, action: PidAction, gain: T) -> Self {
        self.gains[action as usize] = gain;
        self
    }

    /// Gain limit for a given action
    ///
    /// Gain limit units are `output/input`. See also [`PidBuilder::gain()`].
    /// Multiple gains and limits may interact and lead to peaking.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let ki_limit = 1e3;
    /// let i = Biquad::pid()
    ///     .gain(PidAction::Ki, 8.0)
    ///     .limit(PidAction::Ki, ki_limit)
    ///     .build()
    ///     .unwrap();
    /// let mut xy = [0.0; 5];
    /// let x0 = 5.0;
    /// for _ in 0..1000 {
    ///     i.update(&mut xy, x0);
    /// }
    /// let y0 = i.update(&mut xy, x0);
    /// assert!((y0 / (x0 * ki_limit) - 1.0f32).abs() < 1e-3);
    /// ```
    ///
    /// # Arguments
    /// * `action`: Action to limit in gain
    /// * `limit`: Gain limit
    pub fn limit(mut self, action: PidAction, limit: T) -> Self {
        self.limits[action as usize] = limit;
        self
    }

    /// Perform checks, compute coefficients and return `Biquad`.
    ///
    /// No attempt is made to detect NaNs, non-finite gains, non-positive period,
    /// zero gain limits, or gain/limit sign mismatches.
    /// These will consequently result in NaNs/infinities, peaking, or notches in
    /// the Biquad coefficients.
    ///
    /// Gain limits for zero gain actions or for proportional action are ignored.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let i = Biquad::<f32>::pid()
    ///     .gain(PidAction::Kp, 3.0)
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(i, Biquad::proportional(3.0));
    /// ```
    pub fn build(self) -> Result<Biquad<T>, PidError> {
        const KP: usize = PidAction::Kp as usize;

        // Determine highest denominator (feedback, `a`) order
        let low = self
            .gains
            .iter()
            .take(KP)
            .position(|g| !g.is_zero())
            .unwrap_or(KP);

        if self.gains.iter().skip(low + 3).any(|g| !g.is_zero()) {
            return Err(PidError::OrderRange);
        }

        // Derivative/integration kernels
        let kernels = [
            [T::one(), T::zero(), T::zero()],
            [T::one(), -T::one(), T::zero()],
            [T::one(), -T::one() - T::one(), T::one()],
        ];

        // Coefficients
        let mut b = [T::zero(); 3];
        let mut a = [T::zero(); 3];

        let mut zi = self.period.powi(low as i32 - KP as i32);

        for ((i, (gi, li)), ki) in self
            .gains
            .iter()
            .zip(self.limits.iter())
            .enumerate()
            .skip(low)
            .zip(kernels.iter())
        {
            // Scale gains and compute limits in place
            let gi = *gi * zi;
            zi = zi * self.period;
            let li = if i == KP { T::one() } else { gi / *li };

            for (j, (bj, aj)) in b.iter_mut().zip(a.iter_mut()).enumerate() {
                *bj = *bj + gi * ki[j];
                *aj = *aj + li * ki[j];
            }
        }

        // Normalize
        let a0 = T::one() / a[0];
        for baj in b.iter_mut().chain(a.iter_mut().skip(1)) {
            *baj = *baj * a0;
        }

        Ok(Biquad {
            ba: [b[0], b[1], b[2], -a[1], -a[2]],
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn pid() {
        let b = Biquad::pid()
            .period(1.0f32)
            .gain(PidAction::Ki, 1e-3)
            .gain(PidAction::Kp, 1.0)
            .gain(PidAction::Kd, 1e2)
            .limit(PidAction::Ki, 1e3)
            .limit(PidAction::Kd, 1e1)
            .build()
            .unwrap();
        let want = [
            9.18190826,
            -18.27272561,
            9.09090826,
            1.90909074,
            -0.90909083,
        ];
        for (ba_have, ba_want) in b.ba.iter().zip(want.iter()) {
            assert!(
                (ba_have / ba_want - 1.0).abs() < 2.0 * f32::EPSILON,
                "have {:?} != want {want:?}",
                &b.ba,
            );
        }
    }

    #[test]
    fn units() {
        let ki = 5e-2;
        let tau = 3e-3;
        let b = Biquad::pid()
            .period(tau)
            .gain(PidAction::Ki, ki)
            .build()
            .unwrap();
        let mut xy = [0.0; 5];
        for i in 1..10 {
            let y_have = b.update(&mut xy, 1.0);
            let y_want = (i as f32) * (ki / tau);
            assert!(
                (y_have / y_want - 1.0).abs() < 3.0 * f32::EPSILON,
                "{i}: have {y_have} != {y_want}"
            );
        }
    }
}
