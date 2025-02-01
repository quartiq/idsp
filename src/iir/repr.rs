use miniconf::{Leaf, Tree};
use num_traits::{AsPrimitive, Float, FloatConst};
use serde::{Deserialize, Serialize};

use crate::{
    iir::{Biquad, Pid},
    Coefficient,
};

/// Floating point BA coefficients before quantization
#[derive(Debug, Clone, Tree)]
pub struct Ba<T> {
    /// Coefficient array: [b0, b1, b2, a0, a1, a2]
    pub ba: Leaf<[T; 6]>,
    /// Summing junction offset
    pub u: Leaf<T>,
    /// Output lower limit
    pub min: Leaf<T>,
    /// Output upper limit
    pub max: Leaf<T>,
}

impl<T> Default for Ba<T>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            ba: Leaf([T::zero(); 6]),
            u: Leaf(T::zero()),
            min: Leaf(T::neg_infinity()),
            max: Leaf(T::infinity()),
        }
    }
}

/// Corner/transition shape parametrization
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default, PartialEq, PartialOrd)]
pub enum ShapeStyle {
    /// Quality factor
    #[default]
    Q,
    /// Relative bandwidth
    Bandwidth,
    /// Shelf slope. `1` is maximally sharp without overshoot.
    Slope,
}

/// Filter type
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default, PartialEq, PartialOrd)]
pub enum Type {
    /// A lowpass
    #[default]
    Lowpass,
    /// A highpass
    Highpass,
    /// A bandpass
    Bandpass,
    /// An allpass
    Allpass,
    /// A notch
    Notch,
    /// A peaking filter
    Peaking,
    /// A low shelf
    Lowshelf,
    /// A high shelf
    Highshelf,
    /// Integrator over harmonic oscillator
    IHo,
}

/// Standard biquad parametrizations
#[derive(Clone, Debug, Tree)]
pub struct FilterRepr<T> {
    /// Filter style
    r#type: Leaf<Type>,
    /// Angular critical frequency (in units of sampling frequency)
    /// Corner frequency, or 3dB cutoff frequency,
    frequency: Leaf<T>,
    /// Passband gain
    gain: Leaf<T>,
    /// Shelf gain (only for peaking, lowshelf, highshelf)
    /// Relative to passband gain
    shelf: Leaf<T>,
    /// Inverse Q/Bandwidth/Slope
    shape: Leaf<T>,
    /// Corner style
    shape_style: Leaf<ShapeStyle>,
    /// Summing junction offset
    offset: Leaf<T>,
    /// Lower output limit
    min: Leaf<T>,
    /// Upper output limit
    max: Leaf<T>,
}

impl<T: Float + FloatConst> Default for FilterRepr<T> {
    fn default() -> Self {
        Self {
            r#type: Leaf(Type::default()),
            frequency: Leaf(T::zero()),
            gain: Leaf(T::one()),
            shelf: Leaf(T::one()),
            shape_style: Leaf(ShapeStyle::default()),
            shape: Leaf(T::SQRT_2().recip()),
            offset: Leaf(T::zero()),
            min: Leaf(T::neg_infinity()),
            max: Leaf(T::infinity()),
        }
    }
}

/// Representation of Biquad
#[derive(Debug, Clone, Tree, strum::EnumString, strum::AsRefStr)]
pub enum BiquadRepr<T, C>
where
    C: Coefficient,
    T: Float + FloatConst,
{
    /// Normalized SI unit coefficients
    Ba(Ba<T>),
    /// Raw, unscaled, possibly fixed point machine unit coefficients
    Raw(Leaf<Biquad<C>>),
    /// A PID
    Pid(Pid<T>),
    /// Standard biquad filters: Notch, Lowpass, Highpass, Shelf etc
    Filter(FilterRepr<T>),
}

impl<T, C> Default for BiquadRepr<T, C>
where
    C: Coefficient,
    T: Float + FloatConst,
{
    fn default() -> Self {
        Self::Ba(Default::default())
    }
}

impl<T, C> BiquadRepr<T, C>
where
    C: Coefficient + AsPrimitive<C> + AsPrimitive<T>,
    T: AsPrimitive<C> + Float + FloatConst,
{
    /// Build a biquad
    pub fn build<I>(&self, period: T, scale: T) -> Biquad<C>
    where
        T: AsPrimitive<I>,
        I: Float + 'static + AsPrimitive<C>,
        C: AsPrimitive<I>,
        f32: AsPrimitive<T>,
    {
        match self {
            Self::Ba(ba) => {
                let mut b = Biquad::from(&*ba.ba);
                let s = scale.recip();
                b.set_u((*ba.u * s).as_());
                b.set_min((*ba.min * s).as_());
                b.set_max((*ba.max * s).as_());
                b
            }
            Self::Raw(Leaf(raw)) => raw.clone(),
            Self::Pid(pid) => pid.build::<_, I>(period, scale),
            Self::Filter(filter) => {
                let mut f = crate::iir::Filter::default();
                f.gain_db(*filter.gain);
                f.critical_frequency(*filter.frequency * period);
                f.shelf_db(*filter.shelf);
                match *filter.shape_style {
                    ShapeStyle::Q => f.q(*filter.shape),
                    ShapeStyle::Bandwidth => f.bandwidth(*filter.shape),
                    ShapeStyle::Slope => f.shelf_slope(*filter.shape),
                };
                let mut b: Biquad<C> = (&match *filter.r#type {
                    Type::Lowpass => f.lowpass(),
                    Type::Highpass => f.highpass(),
                    Type::Allpass => f.allpass(),
                    Type::Bandpass => f.bandpass(),
                    Type::Highshelf => f.highshelf(),
                    Type::Lowshelf => f.lowshelf(),
                    Type::IHo => f.iho(),
                    Type::Notch => f.notch(),
                    Type::Peaking => f.peaking(),
                })
                    .into();
                let s = scale.recip();
                b.set_u((*filter.offset * s).as_());
                b.set_min((*filter.min * s).as_());
                b.set_min((*filter.max * s).as_());
                b
            }
        }
    }
}
