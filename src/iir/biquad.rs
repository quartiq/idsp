use num_traits::{AsPrimitive, Float};
use serde::{Deserialize, Serialize};

use crate::Coefficient;

/// Biquad IIR filter
///
/// A biquadratic IIR filter supports up to two zeros and two poles in the transfer function.
/// It can be used to implement a wide range of responses to input signals.
///
/// The Biquad performs the following operation to compute a new output sample `y0` from a new
/// input sample `x0` given its configuration and previous samples:
///
/// `y0 = clamp(b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 + u, min, max)`
///
/// This implementation here saves storage and improves caching opportunities by decoupling
/// filter configuration (coefficients, limits and offset) from filter state
/// and thus supports both (a) sharing a single filter between multiple states ("channels") and (b)
/// rapid switching of filters (tuning, transfer) for a given state without copying either
/// state of configuration.
///
/// # Filter architecture
///
/// Direct Form 1 (DF1) and Direct Form 2 transposed (DF2T) are the only IIR filter
/// structures with an (effective bin the case of TDF2) single summing junction
/// this allows clamping of the output before feedback.
///
/// DF1 allows atomic coefficient change because only inputs and outputs are pipelined.
/// The summing junction pipelining of TDF2 would require incremental
/// coefficient changes and is thus less amenable to online tuning.
///
/// DF2T needs less state storage (2 instead of 4). This is in addition to the coefficient
/// storage (5 plus 2 limits plus 1 offset)
///
/// DF2T is less efficient and accurate for fixed-point architectures as quantization
/// happens at each intermediate summing junction in addition to the output quantization. This is
/// especially true for common `i64 + i32 * i32 -> i64` MACC architectures.
/// One could use wide state storage for fixed point DF2T but that would negate the storage
/// and processing advantages.
///
/// # Coefficients
///
/// `ba: [T; 5] = [b0, b1, b2, a1, a2]` is the coefficients type.
/// To represent the IIR coefficients, this contains the feed-forward
/// coefficients `b0, b1, b2` followed by the feed-back coefficients
/// `a1, a2`, all five normalized such that `a0 = 1`.
///
/// The summing junction of the filter also receives an offset `u`.
///
/// The filter applies clamping such that `min <= y <= max`.
///
/// See [`crate::iir::Filter`] and [`crate::iir::PidBuilder`] for ways to generate coefficients.
///
/// # Fixed point
///
/// Coefficient scaling (see [`Coefficient`]) is fixed and optimized such that -2 is exactly
/// representable. This is tailored to low-passes, PID, II etc, where the integration rule is
/// [1, -2, 1].
///
/// There are two guard bits in the accumulator before clamping/limiting.
/// While this isn't enough to cover the worst case accumulator, it does catch many real world
/// overflow cases.
///
/// # State
///
/// To represent the IIR state (input and output memory) during [`Biquad::update()`]
/// the DF1 state contains the two previous inputs and output `[x1, x2, y1, y2]`
/// concatenated. Lower indices correspond to more recent samples.
///
/// In the DF2T case the state contains `[b1*x1 + b2*x2 - a1*y1 - a2*y2, b2*x1 - a2*y1]`
///
/// In the DF1 case with first order noise shaping, the state contains `[x1, x2, y1, y2, e1]`
/// where `e0` is the accumulated quantization error.
///
/// # PID controller
///
/// The IIR coefficients can be mapped to other transfer function
/// representations, for example PID controllers as described in
/// <https://hackmd.io/IACbwcOTSt6Adj3_F9bKuw> and
/// <https://arxiv.org/abs/1508.06319>.
///
/// Using a Biquad as a template for a PID controller achieves several important properties:
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
///   implementation of transfer functions beyond biquadratic terms.
#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct Biquad<T> {
    ba: [T; 5],
    u: T,
    min: T,
    max: T,
}

impl<T: Coefficient> Default for Biquad<T> {
    fn default() -> Self {
        Self {
            ba: [T::ZERO; 5],
            u: T::ZERO,
            min: T::MIN,
            max: T::MAX,
        }
    }
}

impl<T: Coefficient> From<[T; 5]> for Biquad<T> {
    fn from(ba: [T; 5]) -> Self {
        Self {
            ba,
            ..Default::default()
        }
    }
}

impl<T, C> From<&[C; 6]> for Biquad<T>
where
    T: Coefficient + AsPrimitive<C>,
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
    T: Coefficient + AsPrimitive<C>,
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

impl<T: Coefficient> Biquad<T> {
    /// A "hold" filter that ingests input and maintains output
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = [0.0, 1.0, 2.0, 3.0];
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
    /// [`Biquad::update()`] returns
    /// `y0 = clamp(b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 + u, min, max)`.
    ///
    /// ```
    /// # use idsp::Coefficient;
    /// # use idsp::iir::*;
    /// assert_eq!(Biquad::<i32>::IDENTITY.ba()[0], <i32 as Coefficient>::ONE);
    /// assert_eq!(Biquad::<i32>::HOLD.ba()[3], -<i32 as Coefficient>::ONE);
    /// ```
    pub fn ba(&self) -> &[T; 5] {
        &self.ba
    }

    /// Mutable reference to the filter coefficients.
    ///
    /// See [`Biquad::ba()`].
    ///
    /// ```
    /// # use idsp::Coefficient;
    /// # use idsp::iir::*;
    /// let mut i = Biquad::default();
    /// i.ba_mut()[0] = <i32 as Coefficient>::ONE;
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
    /// For fixed point types, during the comparison,
    /// the lowest two bits of value and limit are truncated.
    ///
    /// ```
    /// # use idsp::iir::*;
    /// assert_eq!(Biquad::<i32>::default().min(), i32::MIN);
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
    /// i.set_min(4);
    /// assert_eq!(i.update(&mut [0; 4], 0), 4);
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
    /// For fixed point types, during the comparison,
    /// the lowest two bits of value and limit are truncated.
    /// The behavior is as if those two bits were 0 in the case
    /// of `min` and one in the case of `max`.
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
    /// i.set_max(-5);
    /// assert_eq!(i.update(&mut [0; 4], 0), -5);
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
        self.ba[0] + self.ba[1] + self.ba[2]
    }

    /// Compute input-referred (`x`) offset.
    ///
    /// ```
    /// # use idsp::Coefficient;
    /// # use idsp::iir::*;
    /// let mut i = Biquad::proportional(3);
    /// i.set_u(3);
    /// assert_eq!(i.input_offset(), <i32 as Coefficient>::ONE);
    /// ```
    pub fn input_offset(&self) -> T {
        self.u.div_scaled(self.forward_gain())
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
    /// # use idsp::Coefficient;
    /// # use idsp::iir::*;
    /// let mut i = Biquad::proportional(-<i32 as Coefficient>::ONE);
    /// i.set_input_offset(1);
    /// assert_eq!(i.u(), -1);
    /// ```
    ///
    /// # Arguments
    /// * `offset`: Input (`x`) offset.
    pub fn set_input_offset(&mut self, offset: T) {
        self.u = offset.mul_scaled(self.forward_gain());
    }

    /// Direct Form 1 Update
    ///
    /// Ingest a new input value into the filter, update the filter state, and
    /// return the new output. Only the state `xy` is modified.
    ///
    /// ## `N=4` Direct Form 1
    ///
    /// `xy` contains:
    /// * On entry: `[x1, x2, y1, y2]`
    /// * On exit:  `[x0, x1, y0, y1]`
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = [0.0, 1.0, 2.0, 3.0];
    /// let x0 = 4.0;
    /// let y0 = Biquad::IDENTITY.update(&mut xy, x0);
    /// assert_eq!(y0, x0);
    /// assert_eq!(xy, [x0, 0.0, y0, 2.0]);
    /// ```
    ///
    /// ## `N=5` Direct Form 1 with first order noise shaping
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = [1, 2, 3, 4, 5];
    /// let x0 = 6;
    /// let y0 = Biquad::IDENTITY.update(&mut xy, x0);
    /// assert_eq!(y0, x0);
    /// assert_eq!(xy, [x0, 1, y0, 3, 5]);
    /// ```
    ///
    /// `xy` contains:
    /// * On entry: `[x1, x2, y1, y2, e1]`
    /// * On exit:  `[x0, x1, y0, y1, e0]`
    ///
    /// Note: This is only useful for fixed point filters.
    ///
    /// ## `N=2` Direct Form 2 transposed
    ///
    /// Note: This is only useful for floating point filters.
    /// Don't use this for fixed point: Quantization happens at each state store operation.
    /// Ideally the state would be `[T::ACCU; 2]` but then for fixed point it would use equal amount
    /// of storage compared to DF1 for no gain in performance and loss in functionality.
    /// There are also no guard bits here.
    ///
    /// `xy` contains:
    /// * On entry: `[b1*x1 + b2*x2 - a1*y1 - a2*y2, b2*x1 - a2*y1]`
    /// * On exit:  `[b1*x0 + b2*x1 - a1*y0 - a2*y1, b2*x0 - a2*y0]`
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut xy = [0.0, 1.0];
    /// let x0 = 3.0;
    /// let y0 = Biquad::IDENTITY.update(&mut xy, x0);
    /// assert_eq!(y0, x0);
    /// assert_eq!(xy, [1.0, 0.0]);
    /// ```
    ///
    /// # Arguments
    /// * `xy` - Current filter state.
    /// * `x0` - New input.
    ///
    /// # Returns
    /// The new output `y0 = clamp(b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2 + u, min, max)`
    pub fn update<const N: usize>(&self, xy: &mut [T; N], x0: T) -> T {
        match N {
            // DF1
            4 => {
                let s = self.ba[0].as_() * x0.as_()
                    + self.ba[1].as_() * xy[0].as_()
                    + self.ba[2].as_() * xy[1].as_()
                    - self.ba[3].as_() * xy[2].as_()
                    - self.ba[4].as_() * xy[3].as_();
                let (y0, _) = self.u.macc(s, self.min, self.max, T::ZERO);
                xy[1] = xy[0];
                xy[0] = x0;
                xy[3] = xy[2];
                xy[2] = y0;
                y0
            }
            // DF1 with noise shaping for fixed point
            5 => {
                let s = self.ba[0].as_() * x0.as_()
                    + self.ba[1].as_() * xy[0].as_()
                    + self.ba[2].as_() * xy[1].as_()
                    - self.ba[3].as_() * xy[2].as_()
                    - self.ba[4].as_() * xy[3].as_();
                let (y0, e0) = self.u.macc(s, self.min, self.max, xy[4]);
                xy[4] = e0;
                xy[1] = xy[0];
                xy[0] = x0;
                xy[3] = xy[2];
                xy[2] = y0;
                y0
            }
            // DF2T for floating point
            2 => {
                let y0 = (xy[0] + self.ba[0].mul_scaled(x0)).clip(self.min, self.max);
                xy[0] = xy[1] + self.ba[1].mul_scaled(x0) - self.ba[3].mul_scaled(y0);
                xy[1] = self.u + self.ba[2].mul_scaled(x0) - self.ba[4].mul_scaled(y0);
                y0
            }
            _ => unimplemented!(),
        }
    }
}
