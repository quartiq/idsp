use core::ops::{Add, Div, Mul};
use dsp_fixedpoint::{Const, Q};
use dsp_process::{SplitInplace, SplitProcess};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::FloatCore as _;
use num_traits::{AsPrimitive, clamp};

/// Biquad IIR (second order section)
///
/// A biquadratic IIR filter supports up to two zeros and two poles in the transfer function.
///
/// The Biquad performs the following operation to compute a new output sample `y0` from a new
/// input sample `x0` given its configuration and previous samples:
///
/// `y0 = b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2`
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
/// structures with an (effective in the case of TDF2) single summing junction
/// this allows clamping of the output before feedback.
///
/// DF1 allows atomic coefficient change because only inputs and outputs are stored.
/// The summing junction pipelining of TDF2 would require incremental
/// coefficient changes and is thus less amenable to online tuning.
///
/// DF2T needs less state storage (2 instead of 4). This is in addition to the coefficient
/// storage (5 plus 2 limits plus 1 offset)
///
/// DF2T is less efficient and less accurate for fixed-point architectures as quantization
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
/// The summing junction of the [`BiquadClamp`] filter also receives an offset `u`
/// and applies clamping such that `min <= y <= max`.
///
/// See [`crate::iir::Filter`] and [`crate::iir::PidBuilder`] for ways to generate coefficients.
///
/// # Fixed point
///
/// Coefficient scaling for fixed point (i.e. integer) processing relies on [`dsp_fixedpoint::Q`].
///
/// Choose the number of fractional bits to meet coefficient range (e.g. potentially `a1 = 2`
/// for a double integrator) and guard bits.
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
#[derive(Clone, Debug, Default, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Biquad<C> {
    /// Coefficients
    ///
    /// `[b0, b1, b2, a1, a2]`
    ///
    /// Such that
    /// `y0 = (b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2)/(1 << F)`
    /// where `x0, x1, x2` are current, delayed, and doubly delayed inputs and
    /// `y0, y1, y2` are current, delayed, and doubly delayed outputs.
    ///
    /// Note the a1, a2 sign. The transfer function is:
    /// `H(z) = (b0 + b1*z^-1 + b2*z^-2)/((1 << F) - a2*z^-1 - a2*z^-2)`
    pub ba: [C; 5],
}

/// Second-order-section with offset and clamp
#[derive(Clone, Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct BiquadClamp<C, T = C> {
    /// Coefficients
    pub coeff: Biquad<C>,

    /// Summing junction offset
    ///
    /// ```
    /// # use idsp::iir::*;
    /// # use dsp_process::SplitProcess;
    /// let mut i = BiquadClamp::<f32>::default();
    /// i.u = 5.0;
    /// assert_eq!(i.process(&mut DirectForm1::default(), 0.0), 5.0);
    /// ```
    pub u: T,

    /// Summing junction lower limit
    ///
    /// ```
    /// # use idsp::iir::*;
    /// # use dsp_process::SplitProcess;
    /// let mut i = BiquadClamp::<f32>::default();
    /// i.min = 5.0;
    /// assert_eq!(i.process(&mut DirectForm1::default(), 0.0), 5.0);
    /// ```
    pub min: T,

    /// Summing junction upper limit
    ///
    /// ```
    /// # use idsp::iir::*;
    /// # use dsp_process::SplitProcess;
    /// let mut i = BiquadClamp::<f32>::default();
    /// i.max = -5.0;
    /// assert_eq!(i.process(&mut DirectForm1::default(), 0.0), -5.0);
    /// ```
    pub max: T,
}

impl<C, T: Const> Default for BiquadClamp<C, T>
where
    Biquad<C>: Default,
{
    fn default() -> Self {
        Self {
            coeff: Biquad::default(),
            u: T::ZERO,
            min: T::MIN,
            max: T::MAX,
        }
    }
}

impl<C: Const + Copy> Biquad<C> {
    /// A unity gain filter
    ///
    /// ```
    /// # use idsp::iir::*;
    /// # use dsp_process::SplitProcess;
    /// let x0 = 3.0;
    /// let y0 = Biquad::<f32>::IDENTITY.process(&mut DirectForm1::default(), x0);
    /// assert_eq!(y0, x0);
    /// ```
    pub const IDENTITY: Self = Self::proportional(C::ONE);

    /// A filter with the given proportional gain at all frequencies
    ///
    /// ```
    /// # use idsp::iir::*;
    /// # use dsp_process::SplitProcess;
    /// let x0 = 3.0;
    /// let y0 = Biquad::<f32>::proportional(2.0).process(&mut DirectForm1::default(), x0);
    /// assert_eq!(y0, 2.0 * x0);
    /// ```
    pub const fn proportional(k: C) -> Self {
        Self {
            ba: [k, C::ZERO, C::ZERO, C::ZERO, C::ZERO],
        }
    }
    /// A "hold" filter that ingests input and maintains output
    ///
    /// ```
    /// # use idsp::iir::*;
    /// # use dsp_process::SplitProcess;
    /// let mut state = DirectForm1::<f32>::default();
    /// state.xy[2] = 2.0;
    /// let x0 = 7.0;
    /// let y0 = Biquad::<f32>::HOLD.process(&mut state, x0);
    /// assert_eq!(y0, 2.0);
    /// assert_eq!(state.xy, [x0, 0.0, y0, y0]);
    /// ```
    pub const HOLD: Self = Self {
        ba: [C::ZERO, C::ZERO, C::ZERO, C::ONE, C::ZERO],
    };
}

impl<C: Copy + Add<Output = C>> Biquad<C> {
    /// DC forward gain fro input to summing junction
    ///
    /// ```
    /// # use idsp::iir::*;
    /// assert_eq!(Biquad::proportional(3.0).forward_gain(), 3.0);
    /// ```
    pub fn forward_gain(&self) -> C {
        self.ba[0] + self.ba[1] + self.ba[2]
    }
}

impl<C: Copy + Add<Output = C>, T: Copy + Div<C, Output = T> + Mul<C, Output = T>>
    BiquadClamp<C, T>
{
    /// Summing junction offset referred to input
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = BiquadClamp::from(Biquad::proportional(3.0));
    /// i.u = 6.0;
    /// assert_eq!(i.input_offset(), 2.0);
    /// ```
    pub fn input_offset(&self) -> T {
        self.u / self.coeff.forward_gain()
    }

    /// Summing junction offset referred to input
    ///
    /// ```
    /// # use idsp::iir::*;
    /// let mut i = BiquadClamp::from(Biquad::proportional(3.0));
    /// i.set_input_offset(2.0);
    /// assert_eq!(i.u, 6.0);
    /// ```
    pub fn set_input_offset(&mut self, i: T) {
        self.u = i * self.coeff.forward_gain();
    }
}

/// SOS state
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct DirectForm1<T> {
    /// X,Y state
    ///
    /// Contents before `process()`: `[x1, x2, y1, y2]`
    pub xy: [T; 4],
}

/// ```
/// # use dsp_process::SplitProcess;
/// # use idsp::iir::*;
/// let mut state = DirectForm1 {
///     xy: [0.0, 1.0, 2.0, 3.0],
/// };
/// let x0 = 4.0;
/// let y0 = Biquad::<f32>::IDENTITY.process(&mut state, x0);
/// assert_eq!(y0, x0);
/// assert_eq!(state.xy, [x0, 0.0, y0, 2.0]);
/// ```
impl<T: 'static + Copy, C: Copy + Mul<T, Output = A>, A: Add<Output = A> + AsPrimitive<T>>
    SplitProcess<T, T, DirectForm1<T>> for Biquad<C>
{
    fn process(&self, state: &mut DirectForm1<T>, x0: T) -> T {
        let xy = &mut state.xy;
        let ba = &self.ba;
        let y0 = (ba[0] * x0 + ba[1] * xy[0] + ba[2] * xy[1] + ba[3] * xy[2] + ba[4] * xy[3]).as_();
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

/// ```
/// use dsp_process::SplitProcess;
/// use idsp::iir::*;
/// let biquad = BiquadClamp::<f32, f32>::from(Biquad::IDENTITY);
/// let mut state = DirectForm2Transposed::default();
/// let x = 3.0f32;
/// let y = biquad.process(&mut state, x);
/// assert_eq!(x, y);
/// ```
impl<T: Copy + Add<Output = T> + PartialOrd, C> SplitProcess<T, T, DirectForm1<T>>
    for BiquadClamp<C, T>
where
    Biquad<C>: SplitProcess<T, T, DirectForm1<T>>,
{
    fn process(&self, state: &mut DirectForm1<T>, x0: T) -> T {
        let y0 = clamp(self.coeff.process(state, x0) + self.u, self.min, self.max);
        state.xy[2] = y0; // overwrite
        y0
    }
}

/// Direct form 2 transposed SOS state
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct DirectForm2Transposed<T> {
    /// internal state
    ///
    /// `[u1, u2]`
    pub u: [T; 2],
}

/// ```
/// use dsp_process::SplitProcess;
/// use idsp::iir::*;
/// let biquad = Biquad::<f32>::IDENTITY;
/// let mut state = DirectForm2Transposed::default();
/// let x = 3.0f32;
/// let y = biquad.process(&mut state, x);
/// assert_eq!(x, y);
/// ```
impl<T: Copy + Mul<Output = T> + Add<Output = T>> SplitProcess<T, T, DirectForm2Transposed<T>>
    for Biquad<T>
{
    fn process(&self, state: &mut DirectForm2Transposed<T>, x0: T) -> T {
        let u = &mut state.u;
        let ba = &self.ba;
        let y0 = u[0] + ba[0] * x0;
        u[0] = u[1] + ba[1] * x0 + ba[3] * y0;
        u[1] = ba[2] * x0 + ba[4] * y0;
        y0
    }
}

impl<T: Copy + Add<Output = T> + Mul<Output = T> + PartialOrd>
    SplitProcess<T, T, DirectForm2Transposed<T>> for BiquadClamp<T, T>
{
    fn process(&self, state: &mut DirectForm2Transposed<T>, x0: T) -> T {
        let u = &mut state.u;
        let ba = &self.coeff.ba;
        let y0 = clamp(u[0] + ba[0] * x0 + self.u, self.min, self.max);
        u[0] = u[1] + ba[1] * x0 + ba[3] * y0;
        u[1] = ba[2] * x0 + ba[4] * y0;
        y0
    }
}

/// SOS state with wide Y
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct DirectForm1Wide {
    /// X state
    ///
    /// `[x1, x2]`
    pub x: [i32; 2],
    /// Y state
    ///
    /// `[y1, y2]`
    pub y: [i64; 2],
}

impl<const F: i8> SplitProcess<i32, i32, DirectForm1Wide> for Biquad<Q<i32, i64, F>> {
    fn process(&self, state: &mut DirectForm1Wide, x0: i32) -> i32 {
        let x = &mut state.x;
        let y = &mut state.y;
        let ba = &self.ba;
        let mut acc = (ba[0] * x0 + ba[1] * x[0] + ba[2] * x[1]).inner;
        *x = [x0, x[0]];
        acc += (y[0] as u32 as i64 * ba[3].inner as i64) >> 32;
        acc += (y[0] >> 32) as i32 as i64 * ba[3].inner as i64;
        acc += (y[1] as u32 as i64 * ba[4].inner as i64) >> 32;
        acc += (y[1] >> 32) as i32 as i64 * ba[4].inner as i64;
        acc <<= 32 - F;
        *y = [acc, y[0]];
        (acc >> 32) as _
    }
}

impl<const F: i8> SplitProcess<i32, i32, DirectForm1Wide> for BiquadClamp<Q<i32, i64, F>, i32> {
    fn process(&self, state: &mut DirectForm1Wide, x0: i32) -> i32 {
        let x = &mut state.x;
        let y = &mut state.y;
        let ba = &self.coeff.ba;
        let mut acc = (ba[0] * x0 + ba[1] * x[0] + ba[2] * x[1]).inner;
        *x = [x0, x[0]];
        acc += (y[0] as u32 as i64 * ba[3].inner as i64) >> 32;
        acc += (y[0] >> 32) as i32 as i64 * ba[3].inner as i64;
        acc += (y[1] as u32 as i64 * ba[4].inner as i64) >> 32;
        acc += (y[1] >> 32) as i32 as i64 * ba[4].inner as i64;
        acc <<= 32 - F;
        let y0 = Ord::clamp((acc >> 32) as i32 + self.u, self.min, self.max);
        *y = [((y0 as i64) << 32) | acc as u32 as i64, y[0]];
        y0
    }
}

/// SOS state with first order error feedback
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct DirectForm1Dither {
    /// X,Y state
    ///
    /// `[x1, x2, y1, y2]`
    pub xy: [i32; 4],
    /// Error feedback
    pub e: u32,
}

/// ```
/// # use dsp_process::SplitProcess;
/// # use dsp_fixedpoint::Q32;
/// # use idsp::iir::*;
/// let mut state = DirectForm1Dither {
///     xy: [1, 2, 3, 4],
///     e: 5,
/// };
/// let x0 = 6;
/// let y0 = Biquad::<Q32<30>>::IDENTITY.process(&mut state, x0);
/// assert_eq!(y0, x0);
/// assert_eq!(state.xy, [x0, 1, y0, 3]);
/// assert_eq!(state.e, 20);
/// ```
impl<const F: i8> SplitProcess<i32, i32, DirectForm1Dither> for Biquad<Q<i32, i64, F>> {
    fn process(&self, state: &mut DirectForm1Dither, x0: i32) -> i32 {
        let xy = &mut state.xy;
        let e = &mut state.e;
        let ba = &self.ba;
        let mut acc = *e as i64
            + (ba[0] * x0 + ba[1] * xy[0] + ba[2] * xy[1] + ba[3] * xy[2] + ba[4] * xy[3]).inner;
        acc <<= 32 - F;
        *e = acc as _;
        let y0 = (acc >> 32) as _;
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<const F: i8> SplitProcess<i32, i32, DirectForm1Dither> for BiquadClamp<Q<i32, i64, F>, i32> {
    fn process(&self, state: &mut DirectForm1Dither, x0: i32) -> i32 {
        let xy = &mut state.xy;
        let e = &mut state.e;
        let ba = &self.coeff.ba;
        let mut acc = *e as i64
            + (ba[0] * x0 + ba[1] * xy[0] + ba[2] * xy[1] + ba[3] * xy[2] + ba[4] * xy[3]).inner;
        acc <<= 32 - F;
        *e = acc as _;
        let y0 = Ord::clamp((acc >> 32) as i32 + self.u, self.min, self.max);
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<C, T: Copy, S> SplitInplace<T, S> for Biquad<C> where Self: SplitProcess<T, T, S> {}
impl<C, T: Copy, S> SplitInplace<T, S> for BiquadClamp<C, T> where Self: SplitProcess<T, T, S> {}

/// `[[b0, b1, b2], [a0, a1, a2]]` coefficients with the literature sign of a1/a2
macro_rules! impl_from_float {
    ($ty:ident) => {
        impl<C> From<[[$ty; 3]; 2]> for Biquad<C>
        where
            [$ty; 5]: Into<Biquad<C>>,
        {
            fn from(ba: [[$ty; 3]; 2]) -> Self {
                let a0 = 1.0 / ba[1][0];
                [
                    ba[0][0] * a0,
                    ba[0][1] * a0,
                    ba[0][2] * a0,
                    -ba[1][1] * a0,
                    -ba[1][2] * a0,
                ]
                .into()
            }
        }
    };
}
impl_from_float!(f32);
impl_from_float!(f64);

/// Normalized and sign-flipped coefficients
/// `[b0, b1, b2, a1, a2]`
impl<C: Copy + 'static, T> From<[T; 5]> for Biquad<C>
where
    T: AsPrimitive<C>,
{
    fn from(ba: [T; 5]) -> Self {
        Self {
            ba: ba.map(AsPrimitive::as_),
        }
    }
}

impl<C, T, F> From<F> for BiquadClamp<C, T>
where
    F: Into<Biquad<C>>,
    Self: Default,
{
    fn from(coeff: F) -> Self {
        Self {
            coeff: coeff.into(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod test {
    #![allow(dead_code)]
    use super::*;
    use dsp_fixedpoint::Q32;
    use dsp_process::SplitInplace;
    // No manual tuning needed here.
    // Compiler knows best how and when:
    //   unroll loops
    //   cache on stack
    //   handle alignment
    //   register allocate variables
    //   manage pipeline and insn issue

    // cargo asm idsp::iir::biquad::test::pnm -p idsp --rust --target thumbv7em-none-eabihf --lib --target-cpu cortex-m7 --color --mca -M=-iterations=1 -M=-timeline -M=-skip-unsupported-instructions=lack-sched | less -R

    pub fn pnm(
        config: &[Biquad<Q32<29>>; 4],
        state: &mut [DirectForm1<i32>; 4],
        xy0: &mut [i32; 1 << 3],
    ) {
        config.inplace(state, xy0);
    }

    // ~20 cycles/sample/sos on skylake, >200 MS/s
    #[test]
    #[ignore]
    fn sos_insn() {
        let cfg = [
            [[1., 3., 5.], [19., -9., 9.]],
            [[3., 3., 5.], [21., -11., 11.]],
            [[1., 3., 5.], [55., -17., 17.]],
            [[1., 8., 5.], [77., -7., 7.]],
        ]
        .map(|c| Biquad::from(c));
        let mut state = Default::default();
        let mut x = [977371917; 1 << 7];
        for _ in 0..1 << 20 {
            x[9] = x[63];
            let (x, []) = x.as_chunks_mut() else {
                unreachable!()
            };
            for x in x {
                pnm(&cfg, &mut state, x);
            }
        }
    }
}
