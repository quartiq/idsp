//! Control-plane configuration types for `idsp`.

use core::any::Any;

use idsp::{
    Build,
    iir::{
        BiquadClamp,
        coefficients::{Filter, Shape, Type},
        pid::{Gains, Order, Pid, Units},
    },
};
use miniconf::Tree;
use num_traits::{AsPrimitive, Float, FloatConst};
use serde::{Serialize, de::DeserializeOwned};

/// Floating point BA coefficients before quantization.
#[derive(Debug, Clone, Tree)]
#[tree(meta(doc, typename))]
pub struct BaConfig<T> {
    /// Coefficient array: [[b0, b1, b2], [a0, a1, a2]]
    #[tree(with=miniconf::leaf, bounds(serialize="T: Serialize", deserialize="T: DeserializeOwned", any="T: Any"))]
    pub ba: [[T; 3]; 2],
    /// Summing junction offset.
    pub offset: T,
    /// Output lower limit.
    pub min: T,
    /// Output upper limit.
    pub max: T,
}

impl<T: Float> Default for BaConfig<T> {
    fn default() -> Self {
        Self {
            ba: [[T::zero(); 3], [T::one(), T::zero(), T::zero()]],
            offset: T::zero(),
            min: T::neg_infinity(),
            max: T::infinity(),
        }
    }
}

/// Standard biquad filter parameters.
#[derive(Clone, Debug, Tree)]
#[tree(meta(doc, typename))]
pub struct FilterConfig<T> {
    /// Filter style.
    #[tree(with=miniconf::leaf)]
    pub typ: Type,
    /// Relative critical frequency in units of the sample rate.
    pub frequency: T,
    /// Passband gain in dB.
    pub gain_db: T,
    /// Shelf gain in dB.
    ///
    /// Used for peaking, low shelf, and high shelf filters.
    pub shelf_db: T,
    /// Q, bandwidth, or slope.
    #[tree(with=miniconf::leaf, bounds(serialize="T: Serialize", deserialize="T: DeserializeOwned", any="T: Any"))]
    pub shape: Shape<T>,
    /// Summing junction offset.
    pub offset: T,
    /// Lower output limit.
    pub min: T,
    /// Upper output limit.
    pub max: T,
}

impl<T: Float + FloatConst> Default for FilterConfig<T> {
    fn default() -> Self {
        Self {
            typ: Type::default(),
            frequency: T::zero(),
            gain_db: T::zero(),
            shelf_db: T::zero(),
            shape: Shape::default(),
            offset: T::zero(),
            min: T::neg_infinity(),
            max: T::infinity(),
        }
    }
}

/// Named PID gains.
#[derive(Clone, Debug, Tree, Default)]
pub struct GainsConfig<T> {
    /// Double integral.
    pub i2: T,
    /// Integral.
    pub i: T,
    /// Proportional.
    pub p: T,
    /// Derivative.
    pub d: T,
    /// Double derivative.
    pub d2: T,
}

impl<T> GainsConfig<T> {
    /// Create named gain configuration.
    pub const fn new(i2: T, i: T, p: T, d: T, d2: T) -> Self {
        Self { i2, i, p, d, d2 }
    }
}

impl<T: Copy> GainsConfig<T> {
    /// Create gain configuration with the same value for all actions.
    pub const fn splat(value: T) -> Self {
        Self::new(value, value, value, value, value)
    }
}

impl<T: Copy> From<&GainsConfig<T>> for Gains<T> {
    fn from(config: &GainsConfig<T>) -> Self {
        Self::new([config.i2, config.i, config.p, config.d, config.d2])
    }
}

/// PID controller configuration.
#[derive(Clone, Debug, Tree)]
#[tree(meta(doc, typename))]
pub struct PidConfig<T> {
    /// Feedback term order.
    #[tree(with=miniconf::leaf)]
    pub order: Order,
    /// Gain.
    ///
    /// * Sequence: [I², I, P, D, D²]
    /// * Units: output/input * second**order where `Action::I2` has order=-2
    /// * Gains outside the range `order..=order + 3` are ignored
    /// * P gain sign determines sign of all gains
    pub gain: GainsConfig<T>,
    /// Gain limit.
    ///
    /// * Sequence: [I², I, P, D, D²]
    /// * Units: output/input
    /// * P gain limit is ignored
    /// * Limits outside the range `order..order + 3` are ignored
    /// * P gain sign determines sign of all gain limits
    pub limit: GainsConfig<T>,
    /// Setpoint.
    ///
    /// Units: input.
    pub setpoint: T,
    /// Output lower limit.
    ///
    /// Units: output.
    pub min: T,
    /// Output upper limit.
    ///
    /// Units: output.
    pub max: T,
}

impl<T: Float + Default> Default for PidConfig<T> {
    fn default() -> Self {
        Self {
            order: Order::default(),
            gain: GainsConfig::default(),
            limit: GainsConfig::splat(T::infinity()),
            setpoint: T::zero(),
            min: T::neg_infinity(),
            max: T::infinity(),
        }
    }
}

impl<T: Copy> From<&PidConfig<T>> for Pid<T> {
    fn from(config: &PidConfig<T>) -> Self {
        Self {
            order: config.order,
            gain: Gains::from(&config.gain),
            limit: Gains::from(&config.limit),
            setpoint: config.setpoint,
            min: config.min,
            max: config.max,
        }
    }
}

/// Configurable representation of a biquad.
///
/// `miniconf::Tree` can expose the selected variant as a string:
///
/// ```
/// use miniconf::{Tree, str_leaf};
///
/// #[derive(Tree)]
/// struct Foo {
///     #[tree(typ="&str", with=str_leaf, defer=self.config)]
///     typ: (),
///     config: idsp_config::BiquadConfig<f32>,
/// }
/// ```
#[derive(
    Debug,
    Clone,
    Tree,
    strum::AsRefStr,
    strum::EnumString,
    strum::EnumDiscriminants,
    strum::IntoStaticStr,
)]
#[strum_discriminants(derive(serde::Serialize, serde::Deserialize), allow(missing_docs))]
#[tree(meta(doc = "Configurable representation of a biquad", typename))]
pub enum BiquadConfig<T, C = T, Y = T>
where
    BaConfig<T>: Default,
    PidConfig<T>: Default,
    BiquadClamp<C, Y>: Default,
    FilterConfig<T>: Default,
{
    /// Normalized SI unit coefficients.
    Ba(BaConfig<T>),
    /// Raw, unscaled, possibly fixed-point machine-unit coefficients.
    Raw(
        #[tree(with=miniconf::leaf, bounds(
            serialize="C: Serialize, Y: Serialize",
            deserialize="C: DeserializeOwned, Y: DeserializeOwned",
            any="C: Any, Y: Any"))]
        BiquadClamp<C, Y>,
    ),
    /// PID controller parameters.
    Pid(PidConfig<T>),
    /// Standard biquad filters: notch, lowpass, highpass, shelf, and similar.
    Filter(FilterConfig<T>),
}

impl<T, C, Y> Default for BiquadConfig<T, C, Y>
where
    BaConfig<T>: Default,
    PidConfig<T>: Default,
    BiquadClamp<C, Y>: Default,
    FilterConfig<T>: Default,
{
    fn default() -> Self {
        Self::Ba(Default::default())
    }
}

impl<T, C, Y> BiquadConfig<T, C, Y>
where
    BiquadClamp<C, Y>: Default + Clone,
    Y: 'static + Copy,
    T: 'static + Float + FloatConst + Default + AsPrimitive<Y>,
    f32: AsPrimitive<T>,
    Pid<T>: Build<BiquadClamp<C, Y>, Context = Units<T>>,
    [[T; 3]; 2]: Into<BiquadClamp<C, Y>>,
{
    /// Build a biquad from this configuration.
    pub fn build(&self, units: &Units<T>) -> BiquadClamp<C, Y> {
        let yu = units.y.recip();
        let yx = units.x * yu;
        match self {
            Self::Ba(ba) => {
                let mut bba = ba.ba;
                bba[0] = bba[0].map(|b| b * yx);
                let mut b: BiquadClamp<C, Y> = bba.into();
                b.u = (ba.offset * yu).as_();
                b.min = (ba.min * yu).as_();
                b.max = (ba.max * yu).as_();
                b
            }
            Self::Raw(raw) => raw.clone(),
            Self::Pid(pid) => Pid::from(pid).build(units),
            Self::Filter(filter) => {
                let mut f = Filter::default();
                f.gain_db(filter.gain_db);
                f.critical_frequency(filter.frequency * units.t);
                f.shelf_db(filter.shelf_db);
                f.set_shape(filter.shape);
                let mut ba = f.build(filter.typ);
                ba[0] = ba[0].map(|b| b * yx);
                let mut b: BiquadClamp<C, Y> = ba.into();
                b.u = (filter.offset * yu).as_();
                b.min = (filter.min * yu).as_();
                b.max = (filter.max * yu).as_();
                b
            }
        }
    }
}
