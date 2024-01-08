use core::{
    iter::Sum,
    ops::{Add, Neg, Sub},
};
use num_traits::{AsPrimitive, Float, FloatConst};
use serde::{Deserialize, Serialize};

pub trait FilterNum:
    Copy
    + Sum<Self>
    + PartialEq
    + Neg<Output = Self>
    + Sub<Self, Output = Self>
    + Add<Self, Output = Self>
where
    Self: 'static,
{
    const ONE: Self;
    const NEG_ONE: Self;
    const ZERO: Self;
    const MIN: Self;
    const MAX: Self;
    fn macc(self, xa: impl Iterator<Item = (Self, Self)>) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn quantize<C>(value: C) -> Self
    where
        Self: AsPrimitive<C>,
        C: Float + AsPrimitive<Self>;
}

macro_rules! impl_float {
    ($T:ty) => {
        impl FilterNum for $T {
            const ONE: Self = 1.0;
            const NEG_ONE: Self = -Self::ONE;
            const ZERO: Self = 0.0;
            const MIN: Self = Self::NEG_INFINITY;
            const MAX: Self = Self::INFINITY;
            fn macc(self, xa: impl Iterator<Item = (Self, Self)>) -> Self {
                xa.fold(self, |y, (a, x)| a.mul_add(x, y))
                // xa.fold(self, |y, (a, x)| y + a * x)
            }
            fn clamp(self, min: Self, max: Self) -> Self {
                <$T>::clamp(self, min, max)
            }
            fn div(self, other: Self) -> Self {
                self / other
            }
            fn mul(self, other: Self) -> Self {
                self * other
            }
            fn quantize<C: Float + AsPrimitive<Self>>(value: C) -> Self {
                value.as_()
            }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

macro_rules! impl_int {
    ($T:ty, $A:ty, $Q:literal) => {
        impl FilterNum for $T {
            const ONE: Self = 1 << $Q;
            const NEG_ONE: Self = -Self::ONE;
            const ZERO: Self = 0;
            // Need to avoid `$T::MIN*$T::MIN` overflow.
            const MIN: Self = -Self::MAX;
            const MAX: Self = Self::MAX;
            fn macc(self, xa: impl Iterator<Item = (Self, Self)>) -> Self {
                self + (xa.fold(1 << ($Q - 1), |y, (a, x)| y + a as $A * x as $A) >> $Q) as Self
            }
            fn clamp(self, min: Self, max: Self) -> Self {
                Ord::clamp(self, min, max)
            }
            fn div(self, other: Self) -> Self {
                (((self as $A) << $Q) / other as $A) as Self
            }
            fn mul(self, other: Self) -> Self {
                (((1 << ($Q - 1)) + self as $A * other as $A) >> $Q) as Self
            }
            fn quantize<C>(value: C) -> Self
            where
                Self: AsPrimitive<C>,
                C: Float + AsPrimitive<Self>,
            {
                (value * Self::ONE.as_()).round().as_()
            }
        }
    };
}
// Q2.X chosen to be able to exactly and inclusively represent -2 as `-1 << X + 1`
impl_int!(i8, i16, 6);
impl_int!(i16, i32, 14);
impl_int!(i32, i64, 30);
impl_int!(i64, i128, 62);

/// Filter architecture
///
/// Direct Form 1 (DF1) and Direct Form 2 transposed (DF2T) are the only IIR filter
/// structures with an (effective bin the case of TDF2) single summing junction
/// this allows clamping of the output before feedback.
///
/// DF1 allows atomic coefficient change because only x/y are pipelined.
/// The summing junctuion pipelining of TDF2 would require incremental
/// coefficient changes and is thus less amenable to online tuning.
///
/// DF2T needs less state storage (2 instead of 4). This is in addition to the coefficient storage
/// (5 plus 2 limits plus 1 offset)
/// This implementation already saves storage by decoupling coefficients/limits and offset from state
/// and thus supports both (a) sharing a single filter between multiple states ("channels") and (b)
/// rapid switching of filters (tuning, transfer) for a given state without copying.
///
/// DF2T is less efficient and accurate for fixed-point architectures as quantization
/// happens at each intermediate summing junction in addition to the output quantization. This is
/// especially true for common `i64 + i32 * i32 -> i64` MACC architectures.
///
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
///
///
/// Offset and limiting disabled to suit lowpass applications.
/// Coefficient scaling fixed and optimized such that -2 is representable.
/// Tailored to low-passes, PID, II etc, where the integration rule is [1, -2, 1].
/// Since the relevant coefficients `a1` and `a2` are negated, we also negate the
/// stored `y1` and `y2` in the state.
/// Note that `xy` contains the negative `y1` and `y2`, such that `-a1`
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct Biquad<T> {
    ba: [T; 5],
    u: T,
    min: T,
    max: T,
}

impl<T: FilterNum> Default for Biquad<T> {
    fn default() -> Self {
        Self {
            ba: [T::ZERO; 5],
            u: T::ZERO,
            min: T::MIN,
            max: T::MAX,
        }
    }
}

impl<T: FilterNum> From<[T; 5]> for Biquad<T> {
    fn from(ba: [T; 5]) -> Self {
        Self {
            ba,
            ..Default::default()
        }
    }
}

impl<T, C> From<[C; 6]> for Biquad<T>
where
    T: FilterNum + AsPrimitive<C>,
    C: Float + AsPrimitive<T>,
{
    fn from(ba: [C; 6]) -> Self {
        let ia0 = C::one() / ba[3];
        Self::from([
            T::quantize(ba[0] * ia0),
            T::quantize(ba[1] * ia0),
            T::quantize(ba[2] * ia0),
            // b[3]: a0*ia0
            T::quantize(ba[4] * ia0),
            T::quantize(ba[5] * ia0),
        ])
    }
}

impl<T: FilterNum> Biquad<T> {
    /// A "hold" filter that ingests input and maintains output
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = core::array::from_fn(|i| i as _);
    /// let x0 = 7.0;
    /// let y0 = Biquad::HOLD.update(&mut xy, x0);
    /// assert_eq!(y0, -2.0);
    /// assert_eq!(xy, [x0, 0.0, -y0, -y0, 3.0]);
    /// ```
    pub const HOLD: Self = Self {
        ba: [T::ZERO, T::ZERO, T::ZERO, T::NEG_ONE, T::ZERO],
        u: T::ZERO,
        min: T::MIN,
        max: T::MAX,
    };

    /// A unity gain filter
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let x0 = 3.0;
    /// let y0 = Biquad::IDENTITY.update(&mut [0.0; 5], x0);
    /// assert_eq!(y0, x0);
    /// ```
    pub const IDENTITY: Self = Self::proportional(T::ONE);

    /// A filter with the given proportional gain at all frequencies
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let x0 = 2.0;
    /// let k = 5.0;
    /// let y0 = Biquad::proportional(k).update(&mut [0.0; 5], x0);
    /// assert_eq!(y0, x0 * k);
    /// ```
    pub const fn proportional(k: T) -> Self {
        Self {
            ba: [k, T::ZERO, T::ZERO, T::ZERO, T::ZERO],
            u: T::ZERO,
            min: T::MIN,
            max: T::MAX,
        }
    }

    /// Filter coefficients
    ///
    /// IIR filter tap gains (`ba`) are an array `[b0, b1, b2, a1, a2]` such that
    /// [`Biquad::update(&mut xy, x0)`] returns
    /// `y0 = clamp(b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 + u, min, max)`.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// assert_eq!(Biquad::<i32>::IDENTITY.ba()[0], i32::ONE);
    /// assert_eq!(Biquad::<i32>::HOLD.ba()[3], -i32::ONE);
    /// ```
    pub fn ba(&self) -> &[T; 5] {
        &self.ba
    }

    /// Mutable reference to the filter coefficients.
    ///
    /// See [`Biquad::ba()`].
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::default();
    /// i.ba_mut()[0] = i32::ONE;
    /// assert_eq!(i, Biquad::IDENTITY);
    /// ```
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
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::default();
    /// i.set_u(5);
    /// assert_eq!(i.update(&mut [0; 5], 0), 5);
    /// ```
    pub fn set_u(&mut self, u: T) {
        self.u = u;
    }

    /// Lower output limit
    ///
    /// Guaranteed minimum output value.
    /// The value is inclusive.
    /// The clamping also cleanly affects the feedback terms.
    ///
    /// Note: For fixed point filters `Biquad<T>`, `T::MIN` should not be passed
    /// to `min()` since the `y` samples stored in
    /// the filter state are negated. Instead use `-T::MAX` as the lowest
    /// possible limit.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// assert_eq!(Biquad::<i32>::default().min(), -i32::MAX);
    /// ```
    pub fn min(&self) -> T {
        self.min
    }

    /// Set the lower output limit
    ///
    /// See [`Biquad::min()`].
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::default();
    /// i.set_min(7);
    /// assert_eq!(i.update(&mut [0; 5], 0), 7);
    /// ```
    pub fn set_min(&mut self, min: T) {
        self.min = min;
    }

    /// Upper output limit
    ///
    /// Guaranteed maximum output value.
    /// The value is inclusive.
    /// The clamping also cleanly affects the feedback terms.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// assert_eq!(Biquad::<i32>::default().max(), i32::MAX);
    /// ```
    pub fn max(&self) -> T {
        self.max
    }

    /// Set the upper output limit
    ///
    /// See [`Biquad::max()`].
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::default();
    /// i.set_max(-7);
    /// assert_eq!(i.update(&mut [0; 5], 0), -7);
    /// ```
    pub fn set_max(&mut self, max: T) {
        self.max = max;
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
        self.ba.iter().take(3).copied().sum()
    }

    /// Compute input-referred (`x`) offset.
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::proportional(3);
    /// i.set_u(3);
    /// assert_eq!(i.input_offset(), i32::ONE);
    /// ```
    pub fn input_offset(&self) -> T {
        self.u.div(self.forward_gain())
    }

    /// Convert input (`x`) offset to equivalent summing junction offset (`u`) and apply.
    ///
    /// In the case of a "PID" controller the response behavior of the controller
    /// to the offset is "stabilizing", and not "tracking": its frequency response
    /// is exclusively according to the lowest non-zero [`Action`] gain.
    /// There is no high order ("faster") response as would be the case for a "tracking"
    /// controller.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::proportional(3.0);
    /// i.set_input_offset(2.0);
    /// let x0 = 0.5;
    /// let y0 = i.update(&mut [0.0; 5], x0);
    /// assert_eq!(y0, (x0 + i.input_offset()) * i.forward_gain());
    /// ```
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::proportional(-i32::ONE);
    /// i.set_input_offset(1);
    /// assert_eq!(i.u(), -1);
    /// ```
    ///
    /// # Arguments
    /// * `offset`: Input (`x`) offset.
    pub fn set_input_offset(&mut self, offset: T) {
        self.u = offset.mul(self.forward_gain());
    }

    /// Direct Form 1 Update
    ///
    /// Ingest a new input value into the filter, update the filter state, and
    /// return the new output. Only the state `xy` is modified.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = core::array::from_fn(|i| i as _);
    /// let x0 = 3.0;
    /// let y0 = Biquad::IDENTITY.update(&mut xy, x0);
    /// assert_eq!(y0, x0);
    /// assert_eq!(xy, [x0, 0.0, -y0, 2.0, 3.0]);
    /// ```
    ///
    /// # Arguments
    /// * `xy` - Current filter state.
    ///   On entry: `[x1, x2, -y1, -y2, -y3]`
    ///   On exit:  `[x0, x1, -y0, -y1, -y2]`
    /// * `x0` - New input.
    ///
    /// # Returns
    /// The new output `y0 = clamp(b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 + u, min, max)`
    ///
    /// # Panics
    /// Panics in debug mode if `!(self.min <= self.max)`.
    pub fn update(&self, xy: &mut [T; 5], x0: T) -> T {
        // `xy` contains    x0 x1 -y0 -y1 -y2
        // Increment time   x1 x2 -y1 -y2 -y3
        // Shift            x1 x1  x2 -y1 -y2
        xy.copy_within(0..4, 1);
        // Store x0         x0 x1  x2 -y1 -y2
        xy[0] = x0;
        // Compute y0
        let y0 = self
            .u
            .macc(xy.iter().copied().zip(self.ba.iter().copied()))
            .clamp(self.min, self.max);
        // Store -y0        x0 x1 -y0 -y1 -y2
        xy[2] = -y0;
        y0
    }

    /// Direct Form 1 Update
    ///
    /// Ingest a new input value into the filter, update the filter state, and
    /// return the new output. Only the state `xy` is modified.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = core::array::from_fn(|i| i as _);
    /// let x0 = 3.0;
    /// let y0 = Biquad::IDENTITY.update_df1(&mut xy, x0);
    /// assert_eq!(y0, x0);
    /// assert_eq!(xy, [x0, 0.0, -y0, 2.0]);
    /// ```
    ///
    /// # Arguments
    /// * `xy` - Current filter state.
    ///   On entry: `[x1, x2, -y1, -y2]`
    ///   On exit:  `[x0, x1, -y0, -y1]`
    /// * `x0` - New input.
    ///
    /// # Returns
    /// The new output `y0 = clamp(b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 + u, min, max)`
    ///
    /// # Panics
    /// Panics in debug mode if `!(self.min <= self.max)`.
    pub fn update_df1(&self, xy: &mut [T; 4], x0: T) -> T {
        // `xy` contains    x0 x1 -y0 -y1
        // Increment time   x1 x2 -y1 -y2
        // Compute y0
        let y0 = self
            .u
            .macc(
                core::iter::once(x0)
                    .chain(xy.iter().copied())
                    .zip(self.ba.iter().copied()),
            )
            .clamp(self.min, self.max);
        // Shift            x1 x1 -y1 -y2
        xy[1] = xy[0];
        // Store x0         x0 x1 -y1 -y2
        xy[0] = x0;
        // Shift            x0 x1 -y0 -y1
        xy[3] = xy[2];
        // Store -y0        x0 x1 -y0 -y1
        xy[2] = -y0;
        y0
    }

    /// Ingest new input and perform a Direct Form 2 Transposed update.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = core::array::from_fn(|i| i as _);
    /// let x0 = 3.0;
    /// let y0 = Biquad::IDENTITY.update_df2t(&mut xy, x0);
    /// assert_eq!(y0, x0);
    /// assert_eq!(xy, [1.0, 0.0]);
    /// ```
    ///
    /// # Arguments
    /// * `s` - Current filter state.
    ///   On entry: `[b1*x1 + b2*x2 - a1*y1 - a2*y2, b2*x1 - a2*y1]`
    ///   On exit:  `[b1*x0 + b2*x1 - a1*y0 - a2*y1, b2*x0 - a2*y0]`
    /// * `x0` - New input.
    ///
    /// # Returns
    /// The new output `y0 = clamp(b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 + u, min, max)`
    ///
    /// # Panics
    /// Panics in debug mode if `!(self.min <= self.max)`.
    pub fn update_df2t(&self, u: &mut [T; 2], x0: T) -> T {
        let y0 = (u[0] + self.ba[0].mul(x0)).clamp(self.min, self.max);
        u[0] = u[1] + self.ba[1].mul(x0) - self.ba[3].mul(y0);
        u[1] = self.u + self.ba[2].mul(x0) - self.ba[4].mul(y0);
        y0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
enum Shape<T> {
    InverseQ(T),
    Bandwidth(T),
    Slope(T),
}

impl<T: Float + FloatConst> Default for Shape<T> {
    fn default() -> Self {
        Self::InverseQ(T::SQRT_2())
    }
}

/// Standard audio biquad filter builder
///
/// <https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html>
#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Filter<T> {
    /// Angular critical frequency (in units of sampling frequency)
    /// Corner frequency, or 3dB cutoff frequency,
    w0: T,
    /// Passband gain
    gain: T,
    /// Shelf gain (only for peaking, lowshelf, highshelf)
    /// Relative to passband gain
    shelf: T,
    /// Inverse Q
    shape: Shape<T>,
}

impl<T: Float + FloatConst> Default for Filter<T> {
    fn default() -> Self {
        Self {
            w0: T::zero(),
            gain: T::one(),
            shape: Shape::default(),
            shelf: T::one(),
        }
    }
}

impl<T> Filter<T>
where
    T: 'static + Float + FloatConst,
    f32: AsPrimitive<T>,
{
    pub fn frequency(self, critical_frequency: T, sample_frequency: T) -> Self {
        self.critical_frequency(critical_frequency / sample_frequency)
    }

    pub fn critical_frequency(self, critical_frequency: T) -> Self {
        self.angular_critical_frequency(T::TAU() * critical_frequency)
    }

    pub fn angular_critical_frequency(mut self, w0: T) -> Self {
        self.w0 = w0;
        self
    }

    pub fn gain(mut self, k: T) -> Self {
        self.gain = k;
        self
    }

    pub fn gain_db(self, k_db: T) -> Self {
        self.gain(10.0.as_().powf(k_db / 20.0.as_()))
    }

    pub fn shelf(mut self, a: T) -> Self {
        self.shelf = a;
        self
    }

    pub fn shelf_db(self, k_db: T) -> Self {
        self.gain(10.0.as_().powf(k_db / 20.0.as_()))
    }

    pub fn inverse_q(mut self, qi: T) -> Self {
        self.shape = Shape::InverseQ(qi);
        self
    }

    pub fn q(self, q: T) -> Self {
        self.inverse_q(T::one() / q)
    }

    /// Set [`FilterBuilder::frequency()`] first.
    /// In octaves.
    pub fn bandwidth(mut self, bw: T) -> Self {
        self.shape = Shape::Bandwidth(bw);
        self
    }

    /// Set [`FilterBuilder::gain()`] first.
    pub fn shelf_slope(mut self, s: T) -> Self {
        self.shape = Shape::Slope(s);
        self
    }

    fn qi(&self) -> T {
        match self.shape {
            Shape::InverseQ(qi) => qi,
            Shape::Bandwidth(bw) => {
                2.0.as_() * (T::LN_2() / 2.0.as_() * bw * self.w0 / self.w0.sin()).sinh()
            }
            Shape::Slope(s) => {
                ((self.gain + T::one() / self.gain) * (T::one() / s - T::one()) + 2.0.as_()).sqrt()
            }
        }
    }

    fn alpha(&self) -> T {
        0.5.as_() * self.w0.sin() * self.qi()
    }

    /// Lowpass biquad filter.
    ///
    /// ```
    /// use idsp::iir::*;
    /// let ba = Filter::default().critical_frequency(0.1).lowpass();
    /// println!("{ba:?}");
    /// ```
    pub fn lowpass(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        let b = self.gain * 0.5.as_() * (T::one() - fcos);
        [
            b,
            b + b,
            b,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn highpass(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        let b = self.gain * 0.5.as_() * (T::one() + fcos);
        [
            b,
            -(b + b),
            b,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn bandpass(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        let b = self.gain * alpha;
        [
            b,
            T::zero(),
            b,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn notch(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        [
            self.gain,
            -(fcos + fcos) * self.gain,
            self.gain,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn allpass(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        [
            (T::one() - alpha) * self.gain,
            -(fcos + fcos) * self.gain,
            (T::one() + alpha) * self.gain,
            T::one() + alpha,
            -(fcos + fcos),
            T::one() - alpha,
        ]
    }

    pub fn peaking(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let alpha = self.alpha();
        [
            (T::one() + alpha * self.shelf) * self.gain,
            -(fcos + fcos) * self.gain,
            (T::one() - alpha * self.shelf) * self.gain,
            T::one() + alpha / self.shelf,
            -(fcos + fcos),
            T::one() - alpha / self.shelf,
        ]
    }

    pub fn lowshelf(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let sp1 = self.shelf + T::one();
        let sm1 = self.shelf - T::one();
        let tsa = 2.0.as_() * self.shelf.sqrt() * self.alpha();
        [
            self.shelf * self.gain * (sp1 - sm1 * fcos + tsa),
            2.0.as_() * self.shelf * self.gain * (sm1 - sp1 * fcos),
            self.shelf * self.gain * (sp1 - sm1 * fcos - tsa),
            sp1 + sm1 * fcos + tsa,
            (-2.0).as_() * (sm1 + sp1 * fcos),
            sp1 + sm1 * fcos - tsa,
        ]
    }

    pub fn highshelf(self) -> [T; 6] {
        let fcos = self.w0.cos();
        let sp1 = self.shelf + T::one();
        let sm1 = self.shelf - T::one();
        let tsa = 2.0.as_() * self.shelf.sqrt() * self.alpha();
        [
            self.shelf * self.gain * (sp1 + sm1 * fcos + tsa),
            (-2.0).as_() * self.shelf * self.gain * (sm1 + sp1 * fcos),
            self.shelf * self.gain * (sp1 + sm1 * fcos - tsa),
            sp1 - sm1 * fcos + tsa,
            2.0.as_() * (sm1 - sp1 * fcos),
            sp1 - sm1 * fcos - tsa,
        ]
    }

    // TODO
    // PI-notch
    //
    // SOS cascades:
    // butterworth
    // elliptic
    // chebychev1/2
    // bessel
}

/// PID controller builder
///
/// Builds `Biquad` from action gains, gain limits, input offset and output limits.
///
/// ```
/// # use idsp::iir::*;
/// let b: Biquad<f32> = Pid::default()
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
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Pid<T> {
    period: T,
    gains: [T; 5],
    limits: [T; 5],
}

impl<T: Float> Default for Pid<T> {
    fn default() -> Self {
        Self {
            period: T::one(),
            gains: [T::zero(); 5],
            limits: [T::infinity(); 5],
        }
    }
}

/// [`Pid::build()`] errors
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

impl<T: Float + Sum<T>> Pid<T> {
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
    /// frequency response is warped.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let tau = 1e-3;
    /// let ki = 1e-4;
    /// let i: Biquad<f32> = Pid::default()
    ///     .period(tau)
    ///     .gain(Action::Ki, ki)
    ///     .build()
    ///     .unwrap()
    ///     .into();
    /// let x0 = 5.0;
    /// let y0 = i.update(&mut [0.0; 5], x0);
    /// assert!((y0 / (x0 * ki / tau) - 1.0).abs() < 2.0 * f32::EPSILON);
    /// ```
    ///
    /// # Arguments
    /// * `action`: Action to control
    /// * `gain`: Gain value
    pub fn gain(mut self, action: Action, gain: T) -> Self {
        self.gains[action as usize] = gain;
        self
    }

    /// Gain limit for a given action
    ///
    /// Gain limit units are `output/input`. See also [`Pid::gain()`].
    /// Multiple gains and limits may interact and lead to peaking.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let ki_limit = 1e3;
    /// let i: Biquad<f32> = Pid::default()
    ///     .gain(Action::Ki, 8.0)
    ///     .limit(Action::Ki, ki_limit)
    ///     .build()
    ///     .unwrap()
    ///     .into();
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
    pub fn limit(mut self, action: Action, limit: T) -> Self {
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
    /// let i: Biquad<f32> = Pid::default().gain(Action::Kp, 3.0).build().unwrap().into();
    /// assert_eq!(i, Biquad::proportional(3.0));
    /// ```
    pub fn build<C: FilterNum + AsPrimitive<T>>(self) -> Result<[C; 5], PidError>
    where
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
            return Err(PidError::OrderRange);
        }

        // Derivative/integration kernels
        let kernels = [
            [C::ONE, C::ZERO, C::ZERO],
            [C::ONE, C::NEG_ONE, C::ZERO],
            [C::ONE, C::NEG_ONE + C::NEG_ONE, C::ONE],
        ];

        // Scale gains, compute limits, quantize
        let mut zi = self.period.powi(low as i32 - KP as i32);
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
            zi = zi * self.period;
        }
        let a0i = T::one() / gl.iter().map(|gli| gli[1]).sum();

        // Coefficients
        let mut ba = [[C::ZERO; 2]; 3];
        for (gli, ki) in gl.iter().zip(kernels.iter()) {
            let (g, l) = (C::quantize(gli[0] * a0i), C::quantize(gli[1] * a0i));
            for (j, baj) in ba.iter_mut().enumerate() {
                *baj = [baj[0] + ki[j].mul(g), baj[1] + ki[j].mul(l)];
            }
        }

        Ok([ba[0][0], ba[1][0], ba[2][0], ba[1][1], ba[2][1]])
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn pid() {
        let b: Biquad<f32> = Pid::default()
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
        for (ba_have, ba_want) in b.ba.iter().zip(want.iter()) {
            assert!(
                (ba_have / ba_want - 1.0).abs() < 2.0 * f32::EPSILON,
                "have {:?} != want {want:?}",
                &b.ba,
            );
        }
    }

    #[test]
    fn pid_i32() {
        let b: Biquad<i32> = Pid::default()
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
        let b: Biquad<f32> = Pid::default()
            .period(tau)
            .gain(Action::Ki, ki)
            .build()
            .unwrap()
            .into();
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

    #[test]
    fn lowpass_gen() {
        let ba = Biquad::<i32>::from(
            Filter::default()
                .critical_frequency(2e-9f64)
                .gain(2e7)
                .lowpass(),
        );
        println!("{:?}", ba);
    }
}
