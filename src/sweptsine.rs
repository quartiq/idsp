use core::iter::FusedIterator;

use crate::{cossin::cossin, Complex};
#[allow(unused)]
use num_traits::real::Real as _;
use num_traits::FloatConst as _;

const Q: f64 = (1i64 << 32) as f64;

/// Exponential sweep
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Sweep {
    /// Rate of exponential increase
    pub rate: i32,
    /// Current state
    pub state: i64,
}

impl Iterator for Sweep {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let s = self.state;
        self.state += (self.rate as i64) * ((s + (1 << 31)) >> 32);
        Some(s)
    }
}

impl Sweep {
    /// Create a new exponential sweep
    #[inline]
    pub fn new(rate: i32, state: i64) -> Self {
        Self { rate, state }
    }

    /// Continuous time exponential sweep rate
    #[inline]
    pub fn rate(&self) -> f64 {
        (self.rate as f64 / Q).ln_1p()
    }

    /// Samples per octave
    #[inline]
    pub fn octave_len(&self) -> f64 {
        f64::LN_2() / self.rate()
    }

    /// Current continuous time state
    #[inline]
    pub fn state(&self) -> f64 {
        self.cycles() * self.rate()
    }

    /// Response order delay (order >= 1, )
    #[inline]
    pub fn order_delay(&self, order: u32) -> f64 {
        -(order as f64).ln() / self.rate()
    }

    /// Number of cycles in the current octave
    #[inline]
    pub fn cycles(&self) -> f64 {
        self.state as f64 / (Q * self.rate as f64)
    }

    /// Evaluate sweep at a given time
    #[inline]
    pub fn continuous(&self, t: f64) -> f64 {
        let rate = self.rate();
        self.cycles() * rate * (rate * t).exp()
    }

    /// Inverse filter
    ///
    /// * Stimulus `x(t)` (`AccuOsc::new(Sweep::fit(...)).collect()` plus windowing)
    /// * Response `y(t)`
    /// * Response FFT `Y(f)`
    /// * Stimulus inverse filter `X'(f)`
    /// * Transfer function `H(f) = X'(f) Y(f)'
    /// * Impulse response `h(t)`
    /// * Windowing each response using `order_delay()`
    /// * Order responses `H_n(f)`
    pub fn inverse_filter(&self, f: f64) -> Complex<f64> {
        let rt = self.rate();
        let fp = f / rt;
        let r = 2.0 * rt * fp.sqrt();
        let phi = f64::TAU() * (0.125 - fp * (1.0 + self.cycles().ln() - fp.ln()));
        let (s, c) = phi.sin_cos();
        Complex::new(r * c, r * s)
    }

    /// Create new sweep
    ///
    /// * f_end: maximum final frequency in units of sample rate (e.g. 0.5)
    /// * octaves: number of octaves to sweep
    /// * cycles: number of cycles in the first octave (>= 1)
    pub fn fit(f_end: f64, octaves: u32, cycles: u32) -> Result<(Self, usize), SweepError> {
        if !(0.0..=0.5).contains(&f_end) {
            return Err(SweepError::End);
        }
        let u0 = (f64::LN_2() * (cycles << octaves) as f64 / f_end).ceil() as i32;
        // A 20% search range on u, one sided, typically yields < 1e-5 error,
        // and a few e-5 max phase error over the entire sweep.
        // One sided to larger u as this leads one sided lower f_end
        // (do not wrap around Nyquist)
        // Alternatively one could search until tolerance is reached.
        let (u, rate, _err) = (u0..(u0 + u0 / 5))
            .map(|u| {
                let rate = (Q * (f64::LN_2() / u as f64).exp_m1()).round();
                let err = u as f64 - f64::LN_2() / (rate / Q).ln_1p();
                (u, rate as i32, err.abs())
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(core::cmp::Ordering::Equal))
            .ok_or(SweepError::Length)?;
        let state = ((rate * cycles as i32) as i64) << 32;
        if state == 0 {
            return Err(SweepError::Start);
        }
        Ok((Self::new(rate, state), octaves as usize * u as usize))
    }
}

/// Sweep parameter error
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, thiserror::Error)]
pub enum SweepError {
    /// Start out of bounds
    #[error("Start out of bounds")]
    Start,
    /// End out of bounds
    #[error("End out of bounds")]
    End,
    /// Invalid length
    #[error("Length invalid")]
    Length,
}

/// Accumulating oscillator
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct AccuOsc<T> {
    sweep: T,
    /// Current phase
    pub state: i64,
}

impl<T> AccuOsc<T> {
    /// Create a new accumulating oscillator
    #[inline]
    pub fn new(sweep: T) -> Self {
        Self { sweep, state: 0 }
    }
}

impl<T: Iterator<Item = i64>> Iterator for AccuOsc<T> {
    type Item = Complex<i32>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.sweep.next().map(|f| {
            let s = self.state;
            self.state = s.wrapping_add(f);
            let (re, im) = cossin((s >> 32) as _);
            Complex::new(re, im)
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.sweep.size_hint()
    }
}

impl<T: FusedIterator + Iterator<Item = i64>> FusedIterator for AccuOsc<T> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::testing::*;

    #[test]
    fn example() {
        let f_end = 0.3;
        let octaves = 13;
        let cycles = 3;
        let (sweep, len) = Sweep::fit(f_end, octaves, cycles).unwrap();
        // Expected fit result
        let u = 0xed0f;
        assert_eq!(len, u * octaves as usize);
        // Check API
        assert_eq!(sweep.octave_len().round() as usize, u);
        assert_eq!(sweep.cycles().round() as u32, cycles);
        assert_eq!(sweep.state(), sweep.continuous(0.0));
        let f_start = f_end / (1 << octaves) as f64;
        // End in fit range
        assert!((f_start * 0.8..=f_start).contains(&sweep.state()));
        let sweep0 = sweep.clone();
        let x: Vec<_> = AccuOsc::new(sweep)
            .map(|c| Complex::new(c.re as f64 / (0.5 * Q), c.im as f64 / (0.5 * Q)))
            .take(len)
            .collect();
        // Unit circle
        assert!(x.iter().all(|c| isclose(c.norm(), 1.0, 0.0, 1e-3)));
        let phase: Vec<_> = x.iter().map(|c| c.arg() / f64::TAU()).collect();
        // Zero crossings
        for p in phase.iter().step_by(u as _) {
            assert!(isclose(*p, 0.0, 0.0, 2e-5));
        }
        // Analytic continuous time
        for (t, p) in phase.iter().enumerate() {
            let err = p - sweep0.continuous(t as _) / sweep0.rate();
            assert!(isclose(err - err.round(), 0.0, 0.0, 1e-4));
        }
    }
}
