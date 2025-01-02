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
    /// Length
    pub count: usize,
}

impl Iterator for Sweep {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.count = self.count.checked_sub(1)?;
        let s = self.state;
        self.state += (self.rate as i64) * ((s + (1 << 31)) >> 32);
        Some(s)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl FusedIterator for Sweep {}

impl Sweep {
    /// Create a new exponential sweep
    #[inline]
    pub fn new(rate: i32, state: i64, count: usize) -> Self {
        Self { rate, state, count }
    }

    /// Continuous time exponential sweep rate
    pub fn rate(&self) -> f64 {
        (self.rate as f64 / Q).ln_1p()
    }

    /// Samples per octave
    pub fn octave_len(&self) -> f64 {
        f64::LN_2() / self.rate()
    }

    /// Current normalized state
    pub fn state(&self) -> f64 {
        self.state as f64 / Q.powi(2)
    }

    /// Create new sweep
    ///
    /// * f_end: maximum final frequency in units of sample rate (e.g. 0.5)
    /// * octaves: number of octaves to sweep
    /// * cycles: number of cycles in the first octave (>= 1)
    pub fn optimize(f_end: f64, octaves: u32, cycles: u32) -> Result<Self, SweepError> {
        if !(0.0..=0.5).contains(&f_end) {
            return Err(SweepError::End);
        }
        let u0 = (f64::LN_2() * (cycles << octaves) as f64 / f_end).ceil() as i32;
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
        Ok(Self::new(rate, state, octaves as usize * u as usize))
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
        let sweep = Sweep::optimize(f_end, octaves, cycles).unwrap();
        let f_start = f_end / (1 << octaves) as f64;
        assert_eq!(sweep.state, 0x23ee0 << 32);
        assert!((f_start * 0.8..=f_start).contains(&sweep.state()));
        assert_eq!(sweep.rate, 0xbfa0);
        let u = 0xed0f;
        assert_eq!(sweep.count, u * octaves as usize);
        let it: Vec<_> = AccuOsc::new(sweep)
            .map(|c| Complex::new(c.re as f64 / (0.5 * Q), c.im as f64 / (0.5 * Q)))
            .collect();
        assert_eq!(it.len(), u * octaves as usize);
        assert!(it.iter().all(|c| isclose(c.norm(), 1.0, 0.0, 1e-3)));
        let it: Vec<_> = it.iter().map(|c| c.arg() / f64::TAU()).collect();
        for v in it.iter().step_by(u as _) {
            assert!(isclose(*v, 0.0, 0.0, 2e-5));
        }
    }
}
