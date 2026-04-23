use core::num::NonZero;

/// A tiny PRNG
///
/// Marsaglia's 32-bit xorshift.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct XorShift32(NonZero<u32>);

impl Default for XorShift32 {
    fn default() -> Self {
        Self::new(NonZero::<u32>::MIN)
    }
}

impl XorShift32 {
    /// Create a new generator.
    ///
    /// Zero is remapped to one to avoid the absorbing all-zero state.
    pub const fn new(seed: NonZero<u32>) -> Self {
        Self(seed)
    }

    /// Generate the next sample.
    pub fn sample(&mut self) -> NonZero<u32> {
        let mut x = self.0.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = NonZero::new(x).unwrap();
        self.0
    }
}

impl Iterator for XorShift32 {
    type Item = NonZero<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sample())
    }
}

/// Uniform bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Uniform {
    prng: XorShift32,
    cache: u32,
    idx: usize,
}

impl Uniform {
    /// Generate the next sample.
    pub fn sample(&mut self) -> u8 {
        if let Some(idx) = self.idx.checked_sub(1) {
            self.idx = idx;
            self.cache >>= 8;
        } else {
            self.idx = 3;
            self.cache = self.prng.sample().get();
        }
        (self.cache & 0xff) as u8
    }
}

impl Iterator for Uniform {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sample())
    }
}

/// Triangular `[-(1<<8), (1<<8) - 1[`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Triangular {
    uniform: Uniform,
}

impl Triangular {
    /// Generate one sample
    pub fn sample(&mut self) -> i16 {
        self.uniform.sample() as i8 as i16 - self.uniform.sample() as i8 as i16
    }
}

impl Iterator for Triangular {
    type Item = i16;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sample())
    }
}
