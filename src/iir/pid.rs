use core::ops::{Add, AddAssign, Div, Mul, SubAssign};

use miniconf::Tree;
use num_traits::Float;
use serde::{Deserialize, Serialize};

use crate::iir::{Biquad, BiquadClamp};

/// Feedback term order
#[derive(Clone, Debug, Copy, Serialize, Deserialize, Default, PartialEq, PartialOrd)]
pub enum Order {
    /// Proportional
    P = 2,
    /// Integrator
    #[default]
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
/// This enumerates the five possible PID style actions of a Biquad/second-order-section.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Action {
    /// Double integrating, -40 dB per decade
    I2 = 0,
    /// Integrating, -20 dB per decade
    I = 1,
    /// Proportional
    P = 2,
    /// Derivative, 20 dB per decade
    D = 3,
    /// Double derivative, 40 dB per decade
    D2 = 4,
}

impl<T: Float> PidBuilder<T> {
    /// Feedback term order
    ///
    /// # Arguments
    /// * `order`: The maximum feedback term order.
    pub fn order(mut self, order: Order) -> Self {
        self.order = order;
        self
    }

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
    ///   (`-1` for integrating, `0` for proportional, etc.)
    ///
    /// Gains are accurate in the low frequency limit. Towards Nyquist, the
    /// frequency response is warped.
    ///
    /// Note that limit signs and gain signs should match.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// # use dsp_process::SplitProcess;
    /// let tau = 1e-3;
    /// let ki = 1e-4;
    /// let i: Biquad<f32> = PidBuilder::default().period(tau).gain(Action::I, ki).into();
    /// let x0 = 5.0;
    /// let y0 = i.process(&mut DirectForm1::default(), x0);
    /// assert!((y0 / (x0 * tau * ki) - 1.0).abs() < 2.0 * f32::EPSILON);
    /// ```
    ///
    /// # Arguments
    /// * `action`: Action to control
    /// * `gain`: Gain value
    pub fn gain(mut self, action: Action, gain: T) -> Self {
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
    /// # use dsp_process::SplitProcess;
    /// let ki_limit = 1e3;
    /// let i: Biquad<f32> = PidBuilder::default()
    ///     .gain(Action::I, 8.0)
    ///     .limit(Action::I, ki_limit)
    ///     .into();
    /// let mut xy = DirectForm1::default();
    /// let x0 = 5.0;
    /// for _ in 0..1000 {
    ///     i.process(&mut xy, x0);
    /// }
    /// let y0 = i.process(&mut xy, x0);
    /// assert!((y0 / (x0 * ki_limit) - 1.0f32).abs() < 1e-3);
    /// ```
    ///
    /// # Arguments
    /// * `action`: Action to limit in gain
    /// * `limit`: Gain limit
    pub fn limit(mut self, action: Action, limit: T) -> Self {
        self.limit[action as usize] = limit;
        self
    }
}

impl<C, T> From<PidBuilder<T>> for [C; 5]
where
    C: Copy + SubAssign + AddAssign + From<T>,
    T: Float,
{
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
    ///     .into();
    /// assert_eq!(i, Biquad::proportional(3.0f32));
    /// ```
    ///
    /// # Panic
    /// Will panic in debug mode on fixed point coefficient overflow.
    fn from(value: PidBuilder<T>) -> [C; 5] {
        // Choose relevant gains and limits and scale
        let mut z = value.period.powi(-(value.order as i32));
        let mut gl = [[T::zero(); 2]; 3];
        for (gl, (i, (gain, limit))) in gl
            .iter_mut()
            .zip(
                value
                    .gain
                    .iter()
                    .zip(value.limit.iter())
                    .enumerate()
                    .skip(value.order as usize),
            )
            .rev()
        {
            gl[0] = *gain * z;
            gl[1] = if i == Action::P as usize {
                T::one()
            } else {
                gl[0] / *limit
            };
            z = z * value.period;
        }

        // Normalization
        let a0i = T::one() / (gl[0][1] + gl[1][1] + gl[2][1]);

        // Derivative/integration kernels
        let kernels = [[1, 0, 0], [1, -1, 0], [1, -2, 1]];

        // Coefficients
        let mut ba = [[C::from(T::zero()); 2]; 3];
        for (gli, ki) in gl.into_iter().zip(kernels) {
            // Quantize the gains and not the coefficients
            let gli = gli.map(|c| C::from(c * a0i));
            for (baj, kij) in ba.iter_mut().zip(ki) {
                if kij > 0 {
                    for _ in 0..kij {
                        baj[0] += gli[0];
                        baj[1] -= gli[1];
                    }
                } else {
                    for _ in 0..-kij {
                        baj[0] -= gli[0];
                        baj[1] += gli[1];
                    }
                }
            }
        }

        [ba[0][0], ba[1][0], ba[2][0], ba[1][1], ba[2][1]]
    }
}

impl<C, T> From<PidBuilder<T>> for Biquad<C>
where
    PidBuilder<T>: Into<[C; 5]>,
    [C; 5]: Into<Biquad<C>>,
{
    fn from(value: PidBuilder<T>) -> Self {
        value.into().into()
    }
}

/// Named gains
#[derive(Clone, Debug, Tree, Default)]
#[allow(unused)]
pub struct Gains<T> {
    /// Gain values
    ///
    /// See [`Action`] for indices.
    #[tree(skip)]
    pub value: [T; 5],
    /// Double integral
    #[tree(defer = "self.value[Action::I2 as usize]", typ = "T")]
    i2: (),
    /// Integral
    #[tree(defer = "self.value[Action::I as usize]", typ = "T")]
    i: (),
    /// Proportional
    #[tree(defer = "self.value[Action::P as usize]", typ = "T")]
    p: (),
    /// Derivative
    #[tree(defer = "self.value[Action::D as usize]", typ = "T")]
    d: (),
    /// Double derivative
    #[tree(defer = "self.value[Action::D2 as usize]", typ = "T")]
    d2: (),
}

/// PID Controller parameters
#[derive(Clone, Debug, Tree)]
#[tree(meta(doc, typename))]
pub struct Pid<T> {
    /// Feedback term order
    #[tree(with=miniconf::leaf)]
    pub order: Order,
    /// Gain
    ///
    /// * Sequence: [I², I, P, D, D²]
    /// * Units: output/intput * second**order where Action::I2 has order=-2
    /// * Gains outside the range `order..=order + 3` are ignored
    /// * P gain sign determines sign of all gains
    pub gains: Gains<T>,
    /// Gain imit
    ///
    /// * Sequence: [I², I, P, D, D²]
    /// * Units: output/intput
    /// * P gain limit is ignored
    /// * Limits outside the range `order..order + 3` are ignored
    /// * P gain sign determines sign of all gain limits
    pub limits: Gains<T>,
    /// Setpoint
    ///
    /// Units: input
    pub setpoint: T,
    /// Output lower limit
    ///
    /// Units: output
    pub min: T,
    /// Output upper limit
    ///
    /// Units: output
    pub max: T,

    /// Sample period in SI units
    #[tree(skip)]
    pub period: T,

    /// Output scale in SI units
    #[tree(skip)]
    pub y_scale: T,

    /// Output/input scale in SI units
    #[tree(skip)]
    pub b_scale: T,
}

impl<T: Float + Default> Default for Pid<T> {
    fn default() -> Self {
        Self {
            order: Order::default(),
            gains: Gains {
                value: [T::zero(); 5],
                ..Default::default()
            },
            limits: Gains {
                value: [T::infinity(); 5],
                ..Default::default()
            },
            setpoint: T::zero(),
            min: T::neg_infinity(),
            max: T::infinity(),
            period: T::one(),
            y_scale: T::one(),
            b_scale: T::one(),
        }
    }
}

impl<T, C, Y> From<Pid<T>> for BiquadClamp<C, Y>
where
    PidBuilder<T>: Into<BiquadClamp<C, Y>>,
    Y: Copy + From<T> + Mul<C, Output = Y> + Div<C, Output = Y>,
    C: Add<Output = C> + Copy,
    T: Float,
{
    /// Return the `Biquad`
    ///
    /// Builder intermediate type `I`, coefficient type C
    fn from(value: Pid<T>) -> BiquadClamp<C, Y> {
        let p = value.gains.value[Action::P as usize];
        // FIXME: apply b_scale only to ba[..3]!
        let mut biquad: BiquadClamp<C, Y> = PidBuilder {
            gain: value.gains.value.map(|g| value.b_scale * g.copysign(p)),
            limit: value.limits.value.map(|l| {
                // infinite gain limit is meaningful but json can only do null/nan
                let l = if l.is_nan() { T::infinity() } else { l };
                value.b_scale * l.copysign(p)
            }),
            period: value.period,
            order: value.order,
        }
        .into();
        biquad.set_input_offset((-value.setpoint * value.y_scale).into());
        biquad.min = (value.min * value.y_scale).into();
        biquad.max = (value.max * value.y_scale).into();
        biquad
    }
}

#[cfg(test)]
mod test {
    use dsp_process::SplitProcess;

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
            .into();
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
                b.ba,
            );
        }
    }

    #[test]
    fn pid_i32() {
        use dsp_fixedpoint::Q32;
        let b: Biquad<Q32<29>> = PidBuilder::<f32>::default()
            .period(1.0)
            .gain(Action::I, 1e-5)
            .gain(Action::P, 1e-2)
            .gain(Action::D, 1e0)
            .limit(Action::I, 1e1)
            .limit(Action::D, 1e-1)
            .into();
        println!("{b:?}");
    }

    #[test]
    fn units() {
        let ki = 5e-2;
        let tau = 3e-3;
        let b: Biquad<f32> = PidBuilder::default().period(tau).gain(Action::I, ki).into();
        let mut xy = DirectForm1::default();
        for i in 1..10 {
            let y_have = b.process(&mut xy, 1.0);
            let y_want = (i as f32) * tau * ki;
            assert!(
                (y_have / y_want - 1.0).abs() < 3.0 * f32::EPSILON,
                "{i}: have {y_have} != {y_want}"
            );
        }
    }
}
