use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::FilterNum;

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
/// `[T; 5]` is the coefficients type.
///
/// To represent the IIR state (input and output memory) during [`Biquad::update()`]
/// this contains the two previous inputs and output `[x1, x2, y1, y2]`
/// concatenated. Lower indices correspond to more recent samples.
/// To represent the IIR coefficients, this contains the feed-forward
/// coefficients `[b0, b1, b2]` followd by the negated feed-back coefficients
/// `[a1, a2]`, all five normalized such that `a0 = 1`.
/// Note that between filter [`Biquad::update()`] the `xy` state contains
/// `[x0, x1, y0, y1]`.
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
#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
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

impl<T, C> From<&[C; 6]> for Biquad<T>
where
    T: FilterNum + AsPrimitive<C>,
    C: Float + AsPrimitive<T>,
{
    fn from(ba: &[C; 6]) -> Self {
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

impl<T, C> From<&Biquad<T>> for [C; 6]
where
    T: FilterNum + AsPrimitive<C>,
    C: 'static + Copy,
{
    fn from(value: &Biquad<T>) -> Self {
        let ba = value.ba();
        [
            ba[0].as_(),
            ba[1].as_(),
            ba[2].as_(),
            T::ONE.as_(),
            ba[3].as_(),
            ba[4].as_(),
        ]
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
    /// assert_eq!(y0, 2.0);
    /// assert_eq!(xy, [x0, 0.0, y0, y0]);
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
    /// let y0 = Biquad::IDENTITY.update(&mut [0.0; 4], x0);
    /// assert_eq!(y0, x0);
    /// ```
    pub const IDENTITY: Self = Self::proportional(T::ONE);

    /// A filter with the given proportional gain at all frequencies
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let x0 = 2.0;
    /// let k = 5.0;
    /// let y0 = Biquad::proportional(k).update(&mut [0.0; 4], x0);
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
    /// # use idsp::FilterNum;
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
    /// # use idsp::FilterNum;
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
    /// assert_eq!(i.update(&mut [0; 4], 0), 5);
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
    /// assert_eq!(i.update(&mut [0; 4], 0), 7);
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
    /// assert_eq!(i.update(&mut [0; 4], 0), -7);
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
    ///
    /// ```
    /// # use idsp::FilterNum;
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
    /// is exclusively according to the lowest non-zero [`crate::iir::Action`] gain.
    /// There is no high order ("faster") response as would be the case for a "tracking"
    /// controller.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = Biquad::proportional(3.0);
    /// i.set_input_offset(2.0);
    /// let x0 = 0.5;
    /// let y0 = i.update(&mut [0.0; 4], x0);
    /// assert_eq!(y0, (x0 + i.input_offset()) * i.forward_gain());
    /// ```
    ///
    /// ```
    /// # use idsp::FilterNum;
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
    /// assert_eq!(xy, [x0, 0.0, y0, 2.0]);
    /// ```
    ///
    /// # Arguments
    /// * `xy` - Current filter state.
    ///   On entry: `[x1, x2, y1, y2]`
    ///   On exit:  `[x0, x1, y0, y1]`
    /// * `x0` - New input.
    ///
    /// # Returns
    /// The new output `y0 = clamp(b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 + u, min, max)`    
    pub fn update_df1(&self, xy: &mut [T; 4], x0: T) -> T {
        let y0 = self
            .u
            .macc([x0, xy[0], xy[1], -xy[2], -xy[3]].into_iter().zip(self.ba))
            .clamp(self.min, self.max);
        xy[1] = xy[0];
        xy[0] = x0;
        xy[3] = xy[2];
        xy[2] = y0;
        y0
    }

    /// Direct Form 1 update
    ///
    /// See [`Biquad::update_df1()`].
    #[inline]
    pub fn update(&self, xy: &mut [T; 4], x0: T) -> T {
        self.update_df1(xy, x0)
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
    pub fn update_df2t(&self, u: &mut [T; 2], x0: T) -> T {
        let y0 = (u[0] + self.ba[0].mul(x0)).clamp(self.min, self.max);
        u[0] = u[1] + self.ba[1].mul(x0) - self.ba[3].mul(y0);
        u[1] = self.u + self.ba[2].mul(x0) - self.ba[4].mul(y0);
        y0
    }
}
