use miniconf::{Leaf, Tree};
use num_traits::{AsPrimitive, Float, FloatConst};
use serde::{Deserialize, Serialize};

use crate::{
    Coefficient,
    iir::{Biquad, Pid, Shape},
};

/// Floating point BA coefficients before quantization
#[derive(Debug, Clone, Tree)]
pub struct Ba<T> {
    /// Coefficient array: [[b0, b1, b2], [a0, a1, a2]]
    pub ba: Leaf<[[T; 3]; 2]>,
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
            ba: Leaf([[T::zero(); 3], [T::one(), T::zero(), T::zero()]]),
            u: Leaf(T::zero()),
            min: Leaf(T::neg_infinity()),
            max: Leaf(T::infinity()),
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
pub struct FilterRepr<T> {
    /// Filter style
    typ: Leaf<Typ>,
    /// Angular critical frequency (in units of sampling frequency)
    /// Corner frequency, or 3dB cutoff frequency,
    frequency: Leaf<T>,
    /// Passband gain
    gain: Leaf<T>,
    /// Shelf gain (only for peaking, lowshelf, highshelf)
    /// Relative to passband gain
    shelf: Leaf<T>,
    /// Q/Bandwidth/Slope
    shape: Leaf<Shape<T>>,
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
            typ: Leaf(Typ::default()),
            frequency: Leaf(T::zero()),
            gain: Leaf(T::one()),
            shelf: Leaf(T::one()),
            shape: Leaf(Shape::default()),
            offset: Leaf(T::zero()),
            min: Leaf(T::neg_infinity()),
            max: Leaf(T::infinity()),
        }
    }
}

/// Representation of Biquad
#[derive(
    Debug,
    Clone,
    Tree,
    strum::EnumString,
    strum::AsRefStr,
    strum::FromRepr,
    strum::EnumDiscriminants,
)]
#[strum_discriminants(derive(serde::Serialize, serde::Deserialize))]
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

impl<T, C> BiquadRepr<T, C>
where
    C: Coefficient,
    T: Float + FloatConst,
{
    /// `TreeSerialize` for the discriminant
    ///
    /// Use this through a leaf node:
    ///
    /// ```ignore
    /// #[tree(typ="Leaf<iir::BiquadReprDiscriminants>", rename="typ",
    ///     with(serialize=self.repr.tag_serialize, deserialize=self.repr.tag_deserialize),
    ///     deny(ref_any="deny", mut_any="deny"))]
    /// _tag: (),
    /// repr: iir::BiquadRepr<f32, f32>,
    /// ```
    pub fn tag_serialize<K: miniconf::Keys, S: serde::Serializer>(
        &self,
        keys: K,
        ser: S,
    ) -> Result<S::Ok, miniconf::Error<S::Error>> {
        miniconf::TreeSerialize::serialize_by_key(
            &Leaf(BiquadReprDiscriminants::from(self)),
            keys,
            ser,
        )
    }

    /// `TreeDeserialize` for the discriminant
    pub fn tag_deserialize<'de, K: miniconf::Keys, D: serde::Deserializer<'de>>(
        &mut self,
        keys: K,
        de: D,
    ) -> Result<(), miniconf::Error<D::Error>> {
        let mut v = Leaf(BiquadReprDiscriminants::from(&*self));
        miniconf::TreeDeserialize::deserialize_by_key(&mut v, keys, de)?;
        *self = BiquadRepr::from_repr(*v as _).unwrap();
        Ok(())
    }
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
    ///
    /// # Args:
    /// * `period`: The sample period in desired units (e.g. SI seconds)
    /// * `b_scale`: The feed forward (`b` coefficient) conversion scale from
    ///   desired units to machine units.
    ///   An identity (`gain=1`) filter a `x` input in machine units
    ///   will lead to a `y=b_scale*x` filter output in machine units.
    /// * `y_scale`: The y output scale from desired units to machine units.
    ///   E.g. a `max` setting will lead to a `y=y_scale*max` upper limit
    ///   of the filter in machine units.
    pub fn build<I>(&self, period: T, b_scale: T, y_scale: T) -> Biquad<C>
    where
        T: AsPrimitive<I>,
        I: Float + 'static + AsPrimitive<C>,
        C: AsPrimitive<I>,
        f32: AsPrimitive<T>,
    {
        match self {
            Self::Ba(ba) => {
                let mut b = Biquad::from(&[ba.ba[0].map(|b| b * b_scale), ba.ba[1]]);
                b.set_u((*ba.u * y_scale).as_());
                b.set_min((*ba.min * y_scale).as_());
                b.set_max((*ba.max * y_scale).as_());
                b
            }
            Self::Raw(Leaf(raw)) => raw.clone(),
            Self::Pid(pid) => pid.build::<_, I>(period, b_scale, y_scale),
            Self::Filter(filter) => {
                let mut f = crate::iir::Filter::default();
                f.gain_db(*filter.gain);
                f.critical_frequency(*filter.frequency * period);
                f.shelf_db(*filter.shelf);
                f.set_shape(*filter.shape);
                let mut ba = match *filter.typ {
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
                ba[0] = ba[0].map(|b| b * b_scale);
                let mut b = Biquad::from(&ba);
                b.set_u((*filter.offset * y_scale).as_());
                b.set_min((*filter.min * y_scale).as_());
                b.set_max((*filter.max * y_scale).as_());
                b
            }
        }
    }
}
