//! Dynamic representation IIR filter builder

use core::any::Any;
use miniconf::Tree;
use num_traits::{AsPrimitive, Float, FloatConst};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    Build,
    iir::{
        BiquadClamp,
        coefficients::Shape,
        pid::{Pid, Units},
    },
};

/// Floating point BA coefficients before quantization
#[derive(Debug, Clone, Tree)]
#[tree(meta(doc, typename))]
pub struct Ba<T> {
    /// Coefficient array: [[b0, b1, b2], [a0, a1, a2]]
    #[tree(with=miniconf::leaf, bounds(serialize="T: Serialize", deserialize="T: DeserializeOwned", any="T: Any"))]
    pub ba: [[T; 3]; 2],
    /// Summing junction offset
    pub u: T,
    /// Output lower limit
    pub min: T,
    /// Output upper limit
    pub max: T,
}

impl<T: Float> Default for Ba<T> {
    fn default() -> Self {
        Self {
            ba: [[T::zero(); 3], [T::one(), T::zero(), T::zero()]],
            u: T::zero(),
            min: T::neg_infinity(),
            max: T::infinity(),
        }
    }
}

/// Filter type
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default, PartialEq, PartialOrd)]
pub enum Typ {
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
#[tree(meta(doc, typename))]
pub struct FilterRepr<T> {
    /// Filter style
    #[tree(with=miniconf::leaf)]
    pub typ: Typ,
    /// Angular critical frequency (in units of sampling frequency)
    /// Corner frequency, or 3dB cutoff frequency,
    pub frequency: T,
    /// Passband gain
    pub gain: T,
    /// Shelf gain (only for peaking, lowshelf, highshelf)
    /// Relative to passband gain
    pub shelf: T,
    /// Q/Bandwidth/Slope
    #[tree(with=miniconf::leaf, bounds(serialize="T: Serialize", deserialize="T: DeserializeOwned", any="T: Any"))]
    pub shape: Shape<T>,
    /// Summing junction offset
    pub offset: T,
    /// Lower output limit
    pub min: T,
    /// Upper output limit
    pub max: T,
}

impl<T: Float + FloatConst> Default for FilterRepr<T> {
    fn default() -> Self {
        Self {
            typ: Typ::default(),
            frequency: T::zero(),
            gain: T::one(),
            shelf: T::one(),
            shape: Shape::default(),
            offset: T::zero(),
            min: T::neg_infinity(),
            max: T::infinity(),
        }
    }
}

/// Representation of Biquad
///
/// `miniconf::Tree` can be used like this:
///
/// ```
/// use miniconf::{Tree, str_leaf};
/// #[derive(Tree)]
/// struct Foo {
///     #[tree(typ="&str", with=str_leaf, defer=self.repr)]
///     typ: (),
///     repr: idsp::iir::repr::BiquadRepr<f32>,
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
#[tree(meta(doc = "Representation of a biquad", typename))]
pub enum BiquadRepr<T, C = T, Y = T>
where
    Ba<T>: Default,
    Pid<T>: Default,
    BiquadClamp<C, Y>: Default,
    FilterRepr<T>: Default,
{
    /// Normalized SI unit coefficients
    Ba(Ba<T>),
    /// Raw, unscaled, possibly fixed point machine unit coefficients
    Raw(
        #[tree(with=miniconf::leaf, bounds(
            serialize="C: Serialize, Y: Serialize",
            deserialize="C: DeserializeOwned, Y: DeserializeOwned",
            any="C: Any, Y: Any"))]
        BiquadClamp<C, Y>,
    ),
    /// A PID
    Pid(Pid<T>),
    /// Standard biquad filters: Notch, Lowpass, Highpass, Shelf etc
    Filter(FilterRepr<T>),
}

impl<T, C, Y> Default for BiquadRepr<T, C, Y>
where
    Ba<T>: Default,
    Pid<T>: Default,
    BiquadClamp<C, Y>: Default,
    FilterRepr<T>: Default,
{
    fn default() -> Self {
        Self::Ba(Default::default())
    }
}

impl<T, C, Y> BiquadRepr<T, C, Y>
where
    BiquadClamp<C, Y>: Default + Clone,
    Y: 'static + Copy,
    T: 'static + Float + FloatConst + Default + AsPrimitive<Y>,
    f32: AsPrimitive<T>,
    Pid<T>: Build<BiquadClamp<C, Y>, Context = Units<T>>,
    [[T; 3]; 2]: Into<BiquadClamp<C, Y>>,
{
    /// Build a biquad
    pub fn build(&self, units: &Units<T>) -> BiquadClamp<C, Y> {
        let yu = units.y.recip();
        let yx = units.x * yu;
        match self {
            Self::Ba(ba) => {
                let mut bba = ba.ba;
                bba[0] = bba[0].map(|b| b * yx);
                let mut b: BiquadClamp<C, Y> = bba.into();
                b.u = (ba.u * yu).as_();
                b.min = (ba.min * yu).as_();
                b.max = (ba.max * yu).as_();
                b
            }
            Self::Raw(raw) => raw.clone(),
            Self::Pid(pid) => pid.build(units),
            Self::Filter(filter) => {
                let mut f = crate::iir::coefficients::Filter::default();
                f.gain_db(filter.gain);
                f.critical_frequency(filter.frequency * units.t);
                f.shelf_db(filter.shelf);
                f.set_shape(filter.shape);
                let mut ba = match filter.typ {
                    Typ::Lowpass => f.lowpass(),
                    Typ::Highpass => f.highpass(),
                    Typ::Allpass => f.allpass(),
                    Typ::Bandpass => f.bandpass(),
                    Typ::Highshelf => f.highshelf(),
                    Typ::Lowshelf => f.lowshelf(),
                    Typ::IHo => f.iho(),
                    Typ::Notch => f.notch(),
                    Typ::Peaking => f.peaking(),
                };
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
