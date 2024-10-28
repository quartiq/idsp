use crate::{cossin::cossin, Complex};
use num_traits::Float;

/// Exponential sweep
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Sweep {
    /// Rate of increase
    pub a: i32,
    /// Current
    pub f: i64,
}

impl Iterator for Sweep {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.f = self
            .f
            .wrapping_add((self.a as i64).wrapping_mul(self.f >> 32));
        Some(self.f)
    }
}

impl Sweep {
    /// Create a new exponential sweep
    #[inline]
    pub fn new(a: i32, f: i64) -> Self {
        Self { a, f }
    }
}

/// Sync Sweep parameter error
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Error {
    /// Start out of bounds
    Start,
    /// End out of bounds
    End,
    /// Length invalid
    Length,
}

/// Exponential synchronized sweep
pub struct SyncExpSweep {
    sweep: Sweep,
    f_end: i32,
}

impl SyncExpSweep {
    /// Create new sweep
    pub fn new(f_start: i32, octaves: u32, samples_octave: u32) -> Result<Self, Error> {
        if f_start.checked_ilog2().ok_or(Error::Start)? + octaves >= 31 {
            return Err(Error::End);
        }
        if samples_octave == 0 {
            return Err(Error::Length);
        }
        let a = Float::round(
            (Float::exp2((samples_octave as f64).recip()) - 1.0) * (1i64 << 32) as f64,
        ) as i32;
        Ok(Self {
            sweep: Sweep::new(a, (f_start as i64) << 32),
            f_end: f_start << octaves,
        })
    }
}

impl Iterator for SyncExpSweep {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.sweep
            .next()
            .filter(|f| ((f >> 32) as i32) < self.f_end)
    }
}

/// Accumulating oscillator
pub struct AccuOsc<T> {
    f: T,
    p: i64,
}

impl<T> From<T> for AccuOsc<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self { f: value, p: 0 }
    }
}

impl<T: Iterator<Item = i64>> Iterator for AccuOsc<T> {
    type Item = Complex<i32>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.f.next().map(|f| {
            self.p = self.p.wrapping_add(f);
            let (re, im) = cossin((self.p >> 32) as _);
            Complex { re, im }
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ComplexExt as _;

    #[test]
    fn len() {
        let f_start = 0x7fff;
        let octaves = 16;
        let samples_octave = 1 << 16;
        let it: Vec<_> =
            AccuOsc::from(SyncExpSweep::new(f_start, octaves, samples_octave).unwrap()).collect();
        let f1 =
            it[it.len() - 1].arg().wrapping_sub(it[it.len() - 2].arg()) as f32 / f_start as f32;
        println!("octaves {}", f1 / (1 << octaves) as f32 - 1.0);
        println!(
            "length {}",
            it.len() as f32 / (octaves * samples_octave) as f32 - 1.0
        );
    }
}
