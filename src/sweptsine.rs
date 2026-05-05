use core::iter::FusedIterator;
use core::num::Wrapping;

use crate::Complex;
use dsp_process::{Integrator, Process, Split, SplitProcess};
use num_traits::{FloatConst, real::Real};

const Q: f32 = (1i64 << 32) as f32;

/// Exponential sweep
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Sweep {
    /// Rate of exponential increase
    pub rate: i32,
    /// Current state
    ///
    /// Includes first order delta sigma modulator
    pub state: i64,
}

/// Post-increment
impl Iterator for Sweep {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        const BIAS: i64 = 1 << 31;
        let s = self.state;
        self.state = s.checked_add(self.rate as i64 * ((s + BIAS) >> 32))?;
        Some(s)
    }
}

impl Sweep {
    /// Create a new exponential sweep
    #[inline]
    pub const fn new(rate: i32, state: i64) -> Self {
        Self { rate, state }
    }

    /// Continuous time exponential sweep rate
    #[inline]
    pub fn rate(&self) -> f64 {
        Real::ln_1p(self.rate as f64 / Q as f64)
    }

    /// Harmonic delay/length
    #[inline]
    pub fn delay(&self, harmonic: f64) -> f64 {
        Real::ln(harmonic) / self.rate()
    }

    /// Samples per octave
    #[inline]
    pub fn octave(&self) -> f64 {
        f64::LN_2() / self.rate()
    }

    /// Samples per decade
    #[inline]
    pub fn decade(&self) -> f64 {
        f64::LN_10() / self.rate()
    }

    /// Current continuous time state
    #[inline]
    pub fn state(&self) -> f64 {
        self.cycles() * self.rate()
    }

    /// Number of cycles per harmonic
    #[inline]
    pub fn cycles(&self) -> f64 {
        self.state as f64 / (Q as f64 * self.rate as f64)
    }

    /// Evaluate integrated sweep at a given time
    #[inline]
    pub fn continuous(&self, t: f64) -> f64 {
        self.cycles() * Real::exp(self.rate() * t)
    }

    /// Inverse filter
    ///
    /// * Stimulus `x(t)` (`AccuOsc::new(Sweep::fit(...)).collect()` plus windowing)
    /// * Response `y(t)`
    /// * Response FFT `Y(f)`
    /// * Stimulus inverse filter `X'(f)`
    /// * Transfer function `H(f) = X'(f) Y(f)'
    /// * Impulse response `h(t)`
    /// * Windowing each response using `delay()`
    /// * Order responses `H_n(f)`
    pub fn inverse_filter(&self, mut f: f32) -> Complex<f32> {
        let rate = Real::ln_1p(self.rate as f32 / Q);
        f /= rate;
        let amp = 2.0 * rate * f.sqrt();
        let inv_cycles = Q * self.rate as f32 / self.state as f32;
        let turns = 0.125 - f * (1.0 - Real::ln(f * inv_cycles));
        let (im, re) = Real::sin_cos(f32::TAU() * turns);
        Complex::new(amp * re, amp * im)
    }

    /// Create new sweep
    ///
    /// * stop: maximum stop frequency in units of sample rate (e.g. Nyquist, 0.5)
    /// * harmonics: number of harmonics to sweep
    /// * cycles: number of cycles (phase wraps) per harmonic (>= 1)
    pub fn fit(stop: f32, harmonics: f32, cycles: f32) -> Result<Self, SweepError> {
        if !(0.0..=0.5).contains(&stop) {
            return Err(SweepError::Stop);
        }
        let rate = Real::round(Q * Real::exp_m1(stop / (cycles * harmonics))) as i32;
        let state = (rate as i64 * cycles as i64) << 32;
        if state <= 0 {
            return Err(SweepError::Start);
        }
        Ok(Self::new(rate, state))
    }
}

/// Sweep parameter error
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, thiserror::Error)]
pub enum SweepError {
    /// Start out of bounds
    #[error("Start out of bounds")]
    Start,
    /// Stop out of bounds
    #[error("Stop out of bounds")]
    Stop,
}

/// Phase to IQ conversion
#[derive(Clone, Debug, PartialEq, PartialOrd, Default)]
pub struct Osc;

macro_rules! impl_osc {
    ($x:ty, $y:ty) => {
        impl SplitProcess<$x, Complex<$y>> for Osc {
            fn process(&self, _: &mut (), x: $x) -> Complex<$y> {
                Complex::<$y>::from_angle(x)
            }
        }
    };
}
impl_osc!(Wrapping<i32>, i32);
impl_osc!(f32, f32);
impl_osc!(f64, f64);

/// Exponentially swept sine
#[derive(Clone, Debug)]
pub struct AccuOsc<T> {
    sweep: T,
    accu: Integrator<Wrapping<i64>>,
}

impl<T> AccuOsc<T> {
    /// Create a new ExpSweptSine
    pub fn new(sweep: T) -> Self {
        Self {
            sweep,
            accu: Default::default(),
        }
    }

    /// Get the current phase
    pub fn state(&self) -> Wrapping<i64> {
        self.accu.0
    }
}

impl<T: Iterator<Item = i64>> Iterator for AccuOsc<T> {
    type Item = Complex<i32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.sweep
            .next()
            .map(|p| Split::stateless(Osc).process(Wrapping((self.accu.process(p).0 >> 32) as _)))
    }
}

impl<T: FusedIterator + Iterator<Item = i64>> FusedIterator for AccuOsc<T> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::testing::*;

    #[test]
    fn test() {
        let stop = 0.3;
        let harmonics = 3000.0;
        let cycles = 3.0;
        let sweep = Sweep::fit(stop, harmonics, cycles).unwrap();
        assert_eq!(sweep.rate, 0x22f40);
        let len = sweep.delay(harmonics as _);
        assert!(isclose(len, 240190.96, 0.0, 1e-2));
        // Check API
        assert!(isclose(sweep.cycles(), cycles as f64, 0.0, 1e-2));
        assert_eq!(sweep.state(), sweep.continuous(0.0) * sweep.rate());
        // Start/stop as desired
        assert!((stop * 0.99..=1.01 * stop).contains(&(sweep.state() as f32 * harmonics)));
        assert!(
            (stop * 0.99..=1.01 * stop)
                .contains(&((sweep.continuous(len as _) * sweep.rate()) as f32))
        );
        // Zero crossings and wraps
        // Note inclusion of 0
        for h in 0..harmonics as i32 {
            let p = sweep.continuous(sweep.delay(h as _) as _);
            assert!(isclose(p, h as f64 * cycles as f64, 0.0, 1e-10));
        }
        let sweep0 = sweep.clone();
        for (t, p) in sweep
            .scan(0i64, |p, f| {
                let p0 = *p;
                *p = p0.wrapping_add(f);
                Some(p0 as f64 / Q.powi(2) as f64)
            })
            .take(len as _)
            .enumerate()
        {
            // Analytic continuous time
            let err = p - sweep0.continuous(t as _);
            assert!(isclose(err - err.round(), 0.0, 0.0, 5e-5));
        }
    }
}
