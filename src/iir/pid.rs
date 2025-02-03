use miniconf::{Leaf, Tree};
use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::{iir::Biquad, Coefficient};

/// PID controller builder
///
/// Builds `Biquad` from action gains, gain limits, input offset and output limits.
///
/// ```
/// # use idsp::iir::*;
/// let b: Biquad<f32> = PidBuilder::default()
///     .period(1e-3)
///     .gain(Action::Ki, 1e-3)
///     .gain(Action::Kp, 1.0)
///     .gain(Action::Kd, 1e2)
///     .limit(Action::Ki, 1e3)
///     .limit(Action::Kd, 1e1)
///     .build()
///     .unwrap()
///     .into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
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
pub enum PidBuilderError {
    /// The action gains cover more than three successive orders
    OrderRange,
}

/// PID action
///
/// This enumerates the five possible PID style actions of a [`crate::iir::Biquad`]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Action {
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
    pub fn period(&mut self, period: T) -> &mut Self {
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
    /// frequency response is warped.
    ///
    /// Note that limit signs and gain signs should match.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let tau = 1e-3;
    /// let ki = 1e-4;
    /// let i: Biquad<f32> = PidBuilder::default()
    ///     .period(tau)
    ///     .gain(Action::Ki, ki)
    ///     .build()
    ///     .unwrap()
    ///     .into();
    /// let x0 = 5.0;
    /// let y0 = i.update(&mut [0.0; 4], x0);
    /// assert!((y0 / (x0 * tau * ki) - 1.0).abs() < 2.0 * f32::EPSILON);
    /// ```
    ///
    /// # Arguments
    /// * `action`: Action to control
    /// * `gain`: Gain value
    pub fn gain(&mut self, action: Action, gain: T) -> &mut Self {
        self.gains[action as usize] = gain;
        self
    }

    /// Gain limit for a given action
    ///
    /// Gain limit units are `output/input`. See also [`PidBuilder::gain()`].
    /// Multiple gains and limits may interact and lead to peaking.
    ///
    /// Note that limit signs and gain signs should match and that the
    /// default limits are positive infinity.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let ki_limit = 1e3;
    /// let i: Biquad<f32> = PidBuilder::default()
    ///     .gain(Action::Ki, 8.0)
    ///     .limit(Action::Ki, ki_limit)
    ///     .build()
    ///     .unwrap()
    ///     .into();
    /// let mut xy = [0.0; 4];
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
    pub fn limit(&mut self, action: Action, limit: T) -> &mut Self {
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
    /// let i: Biquad<f32> = PidBuilder::default()
    ///     .gain(Action::Kp, 3.0)
    ///     .build()
    ///     .unwrap()
    ///     .into();
    /// assert_eq!(i, Biquad::proportional(3.0));
    /// ```
    ///
    /// # Panic
    /// Will panic in debug mode on fixed point coefficient overflow.
    pub fn build<C>(&self) -> Result<[C; 5], PidBuilderError>
    where
        C: Coefficient + AsPrimitive<T>,
        T: AsPrimitive<C>,
    {
        const KP: usize = Action::Kp as usize;

        // Determine highest denominator (feedback, `a`) order
        let low = self
            .gains
            .iter()
            .take(KP)
            .position(|g| !g.is_zero())
            .unwrap_or(KP);

        if self.gains.iter().skip(low + 3).any(|g| !g.is_zero()) {
            return Err(PidBuilderError::OrderRange);
        }

        // Scale gains, compute limits
        let mut zi = self.period.powi(KP as i32 - low as i32);
        let p = self.period.recip();
        let mut gl = [[T::zero(); 2]; 3];
        for (gli, (i, (ggi, lli))) in gl.iter_mut().zip(
            self.gains
                .iter()
                .zip(self.limits.iter())
                .enumerate()
                .skip(low),
        ) {
            gli[0] = *ggi * zi;
            gli[1] = if i == KP { T::one() } else { gli[0] / *lli };
            zi = zi * p;
        }
        let a0i = T::one() / (gl[0][1] + gl[1][1] + gl[2][1]);

        // Derivative/integration kernels
        let kernels = [
            [C::one(), C::zero(), C::zero()],
            [C::one(), C::zero() - C::one(), C::zero()],
            [C::one(), C::zero() - C::one() - C::one(), C::one()],
        ];

        // Coefficients
        let mut ba = [[C::ZERO; 2]; 3];
        for (gli, ki) in gl.iter().zip(kernels.iter()) {
            // Quantize the gains and not the coefficients
            let (g, l) = (C::quantize(gli[0] * a0i), C::quantize(gli[1] * a0i));
            for (j, baj) in ba.iter_mut().enumerate() {
                *baj = [baj[0] + ki[j] * g, baj[1] + ki[j] * l];
            }
        }

        Ok([ba[0][0], ba[1][0], ba[2][0], ba[1][1], ba[2][1]])
    }
}

/// PID Controller parameters
#[derive(Clone, Debug, Tree)]
pub struct Pid<T> {
    /// Integral gain
    ///
    /// Units: output/input per second
    pub ki: Leaf<T>,
    /// Proportional gain
    ///
    /// Note that this is the sign reference for all gains and limits
    ///
    /// Units: output/input
    pub kp: Leaf<T>,
    /// Derivative gain
    ///
    /// Units: output/input*second
    pub kd: Leaf<T>,
    /// Integral gain limit
    ///
    /// Units: output/input
    pub li: Leaf<T>,
    /// Derivative gain limit
    ///
    /// Units: output/input
    pub ld: Leaf<T>,
    /// Setpoint
    ///
    /// Units: input
    pub setpoint: Leaf<T>,
    /// Output lower limit
    ///
    /// Units: output
    pub min: Leaf<T>,
    /// Output upper limit
    ///
    /// Units: output
    pub max: Leaf<T>,
}

impl<T: Float> Default for Pid<T> {
    fn default() -> Self {
        Self {
            ki: Leaf(T::zero()),
            kp: Leaf(T::zero()), // Note positive default
            kd: Leaf(T::zero()),
            li: Leaf(T::infinity()),
            ld: Leaf(T::infinity()),
            setpoint: Leaf(T::zero()),
            min: Leaf(T::neg_infinity()),
            max: Leaf(T::infinity()),
        }
    }
}

impl<T: Float> Pid<T> {
    /// Return the `Biquad`
    pub fn build<C, I>(&self, period: T, out_scale: T) -> Biquad<C>
    where
        C: Coefficient + AsPrimitive<C> + AsPrimitive<I>,
        T: AsPrimitive<I> + AsPrimitive<C>,
        I: Float + 'static + AsPrimitive<C>,
    {
        let mut biquad: Biquad<C> = PidBuilder::<I>::default()
            .period(period.as_())
            .gain(Action::Ki, self.ki.copysign(*self.kp).as_())
            .gain(Action::Kp, self.kp.as_())
            .gain(Action::Kd, self.kd.copysign(*self.kp).as_())
            .limit(Action::Ki, self.li.copysign(*self.kp).as_())
            .limit(Action::Kd, self.ld.copysign(*self.kp).as_())
            .build()
            .unwrap()
            .into();
        biquad.set_input_offset((-*self.setpoint * out_scale).as_());
        biquad.set_min((*self.min * out_scale).as_());
        biquad.set_max((*self.max * out_scale).as_());
        biquad
    }
}

#[cfg(test)]
mod test {
    use crate::iir::*;

    #[test]
    fn pid() {
        let b: Biquad<f32> = PidBuilder::default()
            .period(1.0)
            .gain(Action::Ki, 1e-3)
            .gain(Action::Kp, 1.0)
            .gain(Action::Kd, 1e2)
            .limit(Action::Ki, 1e3)
            .limit(Action::Kd, 1e1)
            .build()
            .unwrap()
            .into();
        let want = [
            9.18190826,
            -18.27272561,
            9.09090826,
            -1.90909074,
            0.90909083,
        ];
        for (ba_have, ba_want) in b.ba().iter().zip(want.iter()) {
            assert!(
                (ba_have / ba_want - 1.0).abs() < 2.0 * f32::EPSILON,
                "have {:?} != want {want:?}",
                b.ba(),
            );
        }
    }

    #[test]
    fn pid_i32() {
        let b: Biquad<i32> = PidBuilder::default()
            .period(1.0)
            .gain(Action::Ki, 1e-5)
            .gain(Action::Kp, 1e-2)
            .gain(Action::Kd, 1e0)
            .limit(Action::Ki, 1e1)
            .limit(Action::Kd, 1e-1)
            .build()
            .unwrap()
            .into();
        println!("{b:?}");
    }

    #[test]
    fn units() {
        let ki = 5e-2;
        let tau = 3e-3;
        let b: Biquad<f32> = PidBuilder::default()
            .period(tau)
            .gain(Action::Ki, ki)
            .build()
            .unwrap()
            .into();
        let mut xy = [0.0; 4];
        for i in 1..10 {
            let y_have = b.update(&mut xy, 1.0);
            let y_want = (i as f32) * tau * ki;
            assert!(
                (y_have / y_want - 1.0).abs() < 3.0 * f32::EPSILON,
                "{i}: have {y_have} != {y_want}"
            );
        }
    }
}
