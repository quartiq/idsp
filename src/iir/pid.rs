use miniconf::{Leaf, Tree};
use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::{iir::Biquad, Coefficient};

/// Feedback term order
#[derive(Clone, Debug, Copy, Serialize, Deserialize, Default, PartialEq, PartialOrd)]
pub enum Order {
    /// Proportional
    P = 2,
    #[default]
    /// Integrator
    I = 1,
    /// Double integrator
    I2 = 0,
}

/// PID controller builder
///
/// Builds `Biquad` from action gains, gain limits, input offset and output limits.
///
/// ```
/// # use idsp::iir::*;
/// let b: Biquad<f32> = PidBuilder::default()
///     .period(1e-3)
///     .gain(Action::I, 1e-3)
///     .gain(Action::P, 1.0)
///     .gain(Action::D, 1e2)
///     .limit(Action::I, 1e3)
///     .limit(Action::D, 1e1)
///     .build()
///     .into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct PidBuilder<T> {
    period: T,
    order: Order,
    gain: [T; 5],
    limit: [T; 5],
}

impl<T: Float> Default for PidBuilder<T> {
    fn default() -> Self {
        Self {
            period: T::one(),
            order: Order::default(),
            gain: [T::zero(); 5],
            limit: [T::infinity(); 5],
        }
    }
}

/// PID action
///
/// This enumerates the five possible PID style actions of a [`crate::iir::Biquad`]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Action {
    /// Double integrating, -40 dB per decade
    I2 = 0,
    /// Integrating, -20 dB per decade
    I = 1,
    /// Proportional
    P = 2,
    /// Derivative=, 20 dB per decade
    D = 3,
    /// Double derivative, 40 dB per decade
    D2 = 4,
}

impl<T: Float> PidBuilder<T> {
    /// Feedback term order
    ///
    /// # Arguments
    /// * `order`: The maximum feedback term order.
    pub fn order(&mut self, order: Order) -> &mut Self {
        self.order = order;
        self
    }

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
    ///     .gain(Action::I, ki)
    ///     .build()
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
        self.gain[action as usize] = gain;
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
    ///     .gain(Action::I, 8.0)
    ///     .limit(Action::I, ki_limit)
    ///     .build()
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
        self.limit[action as usize] = limit;
        self
    }

    /// Compute coefficients and return `Biquad`.
    ///
    /// No attempt is made to detect NaNs, non-finite gains, non-positive period,
    /// zero gain limits, or gain/limit sign mismatches.
    /// These will consequently result in NaNs/infinities, peaking, or notches in
    /// the Biquad coefficients.
    ///
    /// Gain limits for unused gain actions or for proportional action are ignored.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let i: Biquad<f32> = PidBuilder::default()
    ///     .gain(Action::P, 3.0)
    ///     .order(Order::P)
    ///     .build()
    ///     .into();
    /// assert_eq!(i, Biquad::proportional(3.0));
    /// ```
    ///
    /// # Panic
    /// Will panic in debug mode on fixed point coefficient overflow.
    pub fn build<C>(&self) -> [C; 5]
    where
        C: Coefficient + AsPrimitive<T>,
        T: AsPrimitive<C>,
    {
        // Scale gains, compute limits
        let mut zi = self.period.powi(-(self.order as i32));
        let mut gl = [[T::zero(); 2]; 3];
        for (gli, (i, (ggi, lli))) in gl
            .iter_mut()
            .zip(
                self.gain
                    .iter()
                    .zip(self.limit.iter())
                    .enumerate()
                    .skip(self.order as usize),
            )
            .rev()
        {
            gli[0] = *ggi * zi;
            gli[1] = if i == Action::P as _ {
                T::one()
            } else {
                gli[0] / *lli
            };
            zi = zi * self.period;
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
            for (baj, &kij) in ba.iter_mut().zip(ki) {
                baj[0] = baj[0] + kij * g;
                baj[1] = baj[1] + kij * l;
            }
        }

        [ba[0][0], ba[1][0], ba[2][0], ba[1][1], ba[2][1]]
    }
}

/// Named gains
#[derive(Clone, Debug, Tree, Default)]
#[allow(unused)]
pub struct Gain<T> {
    /// Gain values
    ///
    /// See [`Action`] for indices.
    #[tree(skip)]
    pub value: [Leaf<T>; 5],
    #[tree(defer = "self.value[Action::I2 as usize]", typ = "Leaf<T>")]
    i2: (),
    #[tree(defer = "self.value[Action::I as usize]", typ = "Leaf<T>")]
    i: (),
    #[tree(defer = "self.value[Action::P as usize]", typ = "Leaf<T>")]
    p: (),
    #[tree(defer = "self.value[Action::D as usize]", typ = "Leaf<T>")]
    d: (),
    #[tree(defer = "self.value[Action::D2 as usize]", typ = "Leaf<T>")]
    d2: (),
}

impl<T: Float> Gain<T> {
    fn new(value: T) -> Self {
        Self {
            value: [Leaf(value); 5],
            i2: (),
            i: (),
            p: (),
            d: (),
            d2: (),
        }
    }
}

/// PID Controller parameters
#[derive(Clone, Debug, Tree)]
pub struct Pid<T: Float> {
    /// Feedback term order
    pub order: Leaf<Order>,
    /// Gain
    ///
    /// * Sequence: [I², I, P, D, D²]
    /// * Units: output/intput * second**order where Action::I2 has order=-2
    /// * Gains outside the range `order..=order + 3` are ignored
    /// * P gain sign determines sign of all gains
    pub gain: Gain<T>,
    /// Gain imit
    ///
    /// * Sequence: [I², I, P, D, D²]
    /// * Units: output/intput
    /// * P gain limit is ignored
    /// * Limits outside the range `order..order + 3` are ignored
    /// * P gain sign determines sign of all gain limits
    pub limit: Gain<T>,
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
            order: Leaf(Order::default()),
            gain: Gain::new(T::zero()),
            limit: Gain::new(T::infinity()),
            setpoint: Leaf(T::zero()),
            min: Leaf(T::neg_infinity()),
            max: Leaf(T::infinity()),
        }
    }
}

impl<T: Float> Pid<T> {
    /// Return the `Biquad`
    ///
    /// Builder intermediate type `I`, coefficient type C
    pub fn build<C, I>(&self, period: T, out_scale: T) -> Biquad<C>
    where
        C: Coefficient + AsPrimitive<C> + AsPrimitive<I>,
        T: AsPrimitive<I> + AsPrimitive<C>,
        I: Float + 'static + AsPrimitive<C>,
    {
        let mut biquad: Biquad<C> = PidBuilder::<I> {
            gain: self
                .gain
                .value
                .each_ref()
                .map(|g| g.copysign(*self.gain.value[Action::P as usize]).as_()),
            limit: self
                .limit
                .value
                .each_ref()
                .map(|g| g.copysign(*self.gain.value[Action::P as usize]).as_()),
            period: period.as_(),
            order: *self.order,
        }
        .build()
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
            .gain(Action::I, 1e-3)
            .gain(Action::P, 1.0)
            .gain(Action::D, 1e2)
            .limit(Action::I, 1e3)
            .limit(Action::D, 1e1)
            .build()
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
            .gain(Action::I, 1e-5)
            .gain(Action::P, 1e-2)
            .gain(Action::D, 1e0)
            .limit(Action::I, 1e1)
            .limit(Action::D, 1e-1)
            .build()
            .into();
        println!("{b:?}");
    }

    #[test]
    fn units() {
        let ki = 5e-2;
        let tau = 3e-3;
        let b: Biquad<f32> = PidBuilder::default()
            .period(tau)
            .gain(Action::I, ki)
            .build()
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
