use core::{
    iter::Sum,
    ops::{Add, Div, Mul},
};
use dsp_fixedpoint::{Const, Q};
use dsp_process::{SplitInplace, SplitProcess};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::float::FloatCore as _;

/// Second-order-section
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
    pub u: T,
    /// Summing junction min clamp
    pub min: T,
    /// Summing junction min clamp
    pub max: T,
}

macro_rules! impl_default_float {
    ($ty:ident) => {
        impl Default for BiquadClamp<$ty, $ty> {
            fn default() -> Self {
                Self {
                    coeff: Default::default(),
                    u: 0.0,
                    min: <$ty>::MIN,
                    max: <$ty>::MAX,
                }
            }
        }
    };
}
impl_default_float!(f32);
impl_default_float!(f64);

impl<T, A, const F: i8> Default for BiquadClamp<Q<T, A, F>, T>
where
    Biquad<Q<T, A, F>>: Default,
    T: Const,
{
    fn default() -> Self {
        Self {
            coeff: Default::default(),
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

impl<C: Copy + Sum> Biquad<C> {
    /// DC forward gain fro input to summing junction
    pub fn k(&self) -> C {
        self.ba[..3].iter().copied().sum()
    }
}

impl<C: Copy + Sum, T: Copy + Div<C, Output = T> + Mul<C, Output = T>> BiquadClamp<C, T> {
    /// DC forward gain fro input to summing junction
    pub fn k(&self) -> C {
        self.coeff.k()
    }

    /// Summing junction offset referred to input
    pub fn input_offset(&self) -> T {
        self.u / self.k()
    }

    /// Summing junction offset referred to input
    pub fn set_input_offset(&mut self, i: T) {
        self.u = i * self.k();
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

/// Direct form 2 transposed SOS state
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct DirectForm2Transposed<T> {
    /// internal state
    ///
    /// `[u1, u2]`
    pub u: [T; 2],
}

impl<T: Copy, C: Copy + Mul<T, Output = A>, A: Add<Output = A> + Into<T>>
    SplitProcess<T, T, DirectForm1<T>> for Biquad<C>
{
    fn process(&self, state: &mut DirectForm1<T>, x0: T) -> T {
        let xy = &mut state.xy;
        let ba = &self.ba;
        let y0 =
            (ba[0] * x0 + ba[1] * xy[0] + ba[2] * xy[1] + ba[3] * xy[2] + ba[4] * xy[3]).into();
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

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

impl<T: Copy + Add<Output = T> + Ord, C> SplitProcess<T, T, DirectForm1<T>> for BiquadClamp<C, T>
where
    Biquad<C>: SplitProcess<T, T, DirectForm1<T>>,
{
    fn process(&self, state: &mut DirectForm1<T>, x0: T) -> T {
        let y0 = (self.coeff.process(state, x0) + self.u).clamp(self.min, self.max);
        state.xy[2] = y0; // overwrite
        y0
    }
}

impl<T: Copy + Add<Output = T> + Mul<Output = T> + Ord> SplitProcess<T, T, DirectForm2Transposed<T>>
    for BiquadClamp<T, T>
{
    fn process(&self, state: &mut DirectForm2Transposed<T>, x0: T) -> T {
        let u = &mut state.u;
        let ba = &self.coeff.ba;
        let y0 = (u[0] + ba[0] * x0 + self.u).clamp(self.min, self.max);
        u[0] = u[1] + ba[1] * x0 + ba[3] * y0;
        u[1] = ba[2] * x0 + ba[4] * y0;
        y0
    }
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
        let y0 = ((acc >> 32) as i32 + self.u).clamp(self.min, self.max);
        *y = [((y0 as i64) << 32) | acc as u32 as i64, y[0]];
        y0
    }
}

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
        let y0 = ((acc >> 32) as i32 + self.u).clamp(self.min, self.max);
        *xy = [x0, xy[0], y0, xy[2]];
        y0
    }
}

impl<C, T: Copy, S> SplitInplace<T, S> for Biquad<C> where Self: SplitProcess<T, T, S> {}
impl<C, T: Copy, S> SplitInplace<T, S> for BiquadClamp<C, T> where Self: SplitProcess<T, T, S> {}

macro_rules! impl_from_float {
    ($ty:ident) => {
        impl<C: From<$ty>> From<[[$ty; 3]; 2]> for Biquad<C>
        where
            Biquad<C>: From<[$ty; 5]>,
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

impl<C, T, F> From<F> for BiquadClamp<C, T>
where
    Biquad<C>: From<F>,
    Self: Default,
{
    fn from(ba: F) -> Self {
        Self {
            coeff: ba.into(),
            ..Default::default()
        }
    }
}

impl<C: From<T>, T> From<[T; 5]> for Biquad<C> {
    fn from(ba: [T; 5]) -> Self {
        Self {
            ba: ba.map(Into::into),
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

    // cargo asm idsp::iir::sos::pnm --rust --target thumbv7em-none-eabihf --lib --target-cpu cortex-m7 --color --mca -M=-iterations=1 -M=-timeline -M=-skip-unsupported-instructions=lack-sched | less -R

    pub fn pnm(
        config: &[Biquad<Q32<29>>; 4],
        state: &mut [DirectForm1<i32>; 4],
        xy0: &mut [i32; 1 << 5],
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
