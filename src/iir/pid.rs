//! PID controller IIR filters

use core::ops::{Add, AddAssign, Div, Mul, SubAssign};

use miniconf::Tree;
use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::{
    Build,
    iir::{Biquad, BiquadClamp},
};

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
/// # use idsp::{iir::*, Build};
/// let b: Biquad<f32> = pid::Builder::default()
///     .gain(pid::Action::I, 1e-3)
///     .gain(pid::Action::P, 1.0)
///     .gain(pid::Action::D, 1e2)
///     .limit(pid::Action::I, 1e3)
///     .limit(pid::Action::D, 1e1)
///     .build(&1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Builder<T> {
    order: Order,
    gain: [T; 5],
    limit: [T; 5],
}

impl<T: Float> Default for Builder<T> {
    fn default() -> Self {
        Self {
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

impl<T: Float> Builder<T> {
    /// Feedback term order
    ///
    /// # Arguments
    /// * `order`: The maximum feedback term order.
    pub fn order(mut self, order: Order) -> Self {
        self.order = order;
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
    /// # use idsp::{iir::*, Build};
    /// # use dsp_process::SplitProcess;
    /// let tau = 1e-3;
    /// let ki = 1e-4;
    /// let i: Biquad<f32> = pid::Builder::default().gain(pid::Action::I, ki).build(&tau);
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
    /// Gain limit units are `output/input`. See also [`Builder::gain()`].
    /// Multiple gains and limits may interact and lead to peaking.
    ///
    /// Note that limit signs and gain signs should match and that the
    /// default limits are positive infinity.
    ///
    /// ```
    /// # use idsp::{iir::*, Build};
    /// # use dsp_process::SplitProcess;
    /// let ki_limit = 1e3;
    /// let i: Biquad<f32> = pid::Builder::default()
    ///     .gain(pid::Action::I, 8.0)
    ///     .limit(pid::Action::I, ki_limit)
    ///     .build(&1.0);
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

impl<T, C> Build<[C; 5]> for Builder<T>
where
    C: 'static + Copy + SubAssign + AddAssign,
    T: Float + AsPrimitive<C>,
{
    type Context = T;
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
    /// # use idsp::{iir::*, Build};
    /// let i: Biquad<f32> = pid::Builder::default()
    ///     .gain(pid::Action::P, 3.0)
    ///     .order(pid::Order::P)
    ///     .build(&1.0);
    /// assert_eq!(i, Biquad::proportional(3.0f32));
    /// ```
    ///
    /// # Arguments
    /// * `t_unit`: Sample period in some units, e.g. SI seconds
    ///
    /// # Panic
    /// Will panic in debug mode on fixed point coefficient overflow.
    fn build(&self, period: &T) -> [C; 5] {
        // Choose relevant gains and limits and scale
        let mut z = period.powi(-(self.order as i32));
        let mut gl = [[T::zero(); 2]; 3];
        for (gl, (i, (gain, limit))) in gl
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
            gl[0] = *gain * z;
            gl[1] = if i == Action::P as usize {
                T::one()
            } else {
                gl[0] / *limit
            };
            z = z * *period;
        }

        // Normalization
        let a0i = (gl[0][1] + gl[1][1] + gl[2][1]).recip();

        // Derivative/integration kernels
        let kernels = [[1, 0, 0], [1, -1, 0], [1, -2, 1]];

        // Coefficients
        let mut ba = [[T::zero().as_(); 2]; 3];
        for (gli, ki) in gl.into_iter().zip(kernels) {
            // Quantize the gains and not the coefficients
            let gli = gli.map(|c| (c * a0i).as_());
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

impl<C, T> Build<Biquad<C>> for Builder<T>
where
    Self: Build<[C; 5]>,
    Biquad<C>: From<[C; 5]>,
{
    type Context = <Self as Build<[C; 5]>>::Context;

    fn build(&self, ctx: &Self::Context) -> Biquad<C> {
        self.build(ctx).into()
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

/// Units for a biquad
///
/// In desired (e.g. SI) units per machine (e.g. full scale or LSB) unit
#[derive(Clone, Debug)]
pub struct Units<T> {
    /// Update period
    ///
    /// One update interval corresponds to this many physical units (e.g. seconds).
    pub t: T,
    /// Input unit
    ///
    /// Unit input in machine units corresponds to this many physical units.
    pub x: T,
    /// Output unit
    ///
    /// Unit output in machine units corresponds to this many physical units
    pub y: T,
}

impl<T: Float> Default for Units<T> {
    fn default() -> Self {
        Self {
            t: T::one(),
            x: T::one(),
            y: T::one(),
        }
    }
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
}

impl<T: Float + Default> Default for Pid<T> {
    fn default() -> Self {
        Self {
            order: Order::default(),
            gains: Gains::default(),
            limits: Gains {
                value: [T::infinity(); 5],
                ..Default::default()
            },
            setpoint: T::zero(),
            min: T::neg_infinity(),
            max: T::infinity(),
        }
    }
}

impl<T, Y, C> Build<BiquadClamp<C, Y>> for Pid<T>
where
    Y: 'static + Copy + Mul<C, Output = Y> + Div<C, Output = Y>,
    C: Add<Output = C> + Copy,
    T: AsPrimitive<Y> + Float,
    BiquadClamp<C, Y>: From<[C; 5]>,
    Builder<T>: Build<[C; 5], Context = T>,
{
    type Context = Units<T>;
    /// Return the `Biquad`
    ///
    /// Builder intermediate type `I`, coefficient type C
    fn build(&self, units: &Units<T>) -> BiquadClamp<C, Y> {
        let yu = units.y.recip();
        let yx = units.x * yu;
        let p = self.gains.value[Action::P as usize];
        let mut biquad: BiquadClamp<C, Y> = Builder {
            gain: self.gains.value.map(|g| yx * g.copysign(p)),
            limit: self.limits.value.map(|mut l| {
                // infinite gain limit is meaningful but json can only do null/nan
                if l.is_nan() {
                    l = T::infinity()
                }
                yx * l.copysign(p)
            }),
            order: self.order,
        }
        .build(&units.t)
        .into();
        biquad.set_input_offset((-self.setpoint * units.x.recip()).as_());
        biquad.min = (self.min * yu).as_();
        biquad.max = (self.max * yu).as_();
        biquad
    }
}

#[cfg(test)]
mod test {
    use dsp_process::SplitProcess;

    use crate::iir::{pid::Build, *};

    #[test]
    fn pid() {
        let b: Biquad<f32> = pid::Builder::default()
            .gain(pid::Action::I, 1e-3)
            .gain(pid::Action::P, 1.0)
            .gain(pid::Action::D, 1e2)
            .limit(pid::Action::I, 1e3)
            .limit(pid::Action::D, 1e1)
            .build(&1.0);
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
        let b: Biquad<Q32<29>> = pid::Builder::<f32>::default()
            .gain(pid::Action::I, 1e-5)
            .gain(pid::Action::P, 1e-2)
            .gain(pid::Action::D, 1e0)
            .limit(pid::Action::I, 1e1)
            .limit(pid::Action::D, 1e-1)
            .build(&1.0);
        println!("{b:?}");
    }

    #[test]
    fn units() {
        let ki = 5e-2;
        let tau = 3e-3;
        let b: Biquad<f32> = pid::Builder::default().gain(pid::Action::I, ki).build(&tau);
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
