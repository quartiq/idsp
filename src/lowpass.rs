use num_traits::{Float, Pow};

/// Arbitrary order, high dynamic range, wide coefficient range,
/// lowpass filter implementation. DC gain is 1.
///
/// Type argument N is the filter order.
#[derive(Copy, Clone)]
pub struct Lowpass<const N: usize> {
    // IIR state storage
    y: [i32; N],
}

impl<const N: usize> Default for Lowpass<N> {
    fn default() -> Self {
        Lowpass { y: [0i32; N] }
    }
}

impl<const N: usize> Lowpass<N> {
    /// Update the filter with a new sample.
    ///
    /// # Args
    /// * `x`: Input data. Needs 1 bit headroom but will saturate cleanly beyond that.
    /// * `k`: Log2 time constant, 1..=31.
    ///
    /// # Return
    /// Filtered output y.
    pub fn update(&mut self, x: i32, k: u32) -> i32 {
        debug_assert!(k & 31 == k);
        // This is an unrolled and optimized first-order IIR loop
        // that works for all possible time constants.
        // Note T-DF-I and the zeros at Nyquist.
        let mut x = x;
        for y in self.y.iter_mut() {
            let dy = x.saturating_sub(*y) >> k;
            *y += dy;
            x = *y - (dy >> 1);
        }
        x.saturating_add((N as i32) << (k - 1).max(0))
    }

    /// Return the current filter output
    pub fn output(&self) -> i32 {
        self.y[N - 1]
    }
}

#[derive(Copy, Clone, Default)]
pub struct Lowpass2 {
    pub(crate) y: i64,
    pub(crate) dy: i64,
}

impl Lowpass2 {
    pub fn update(&mut self, x: i32, k: &[i32; 2]) -> i64 {
        self.dy -= (self.dy >> 32) * k[0] as i64;
        self.dy += (x - (self.y >> 32) as i32) as i64 * k[1] as i64;
        self.y += self.dy;
        self.y - (self.dy >> 1)
    }

    pub fn gain(k: i32, g: Option<i32>) -> [i32; 2] {
        let g = g.unwrap_or((2f32.sqrt() * 2f32.pow(32)) as i32);
        [
            ((k as i64 * g as i64) >> 32) as _,
            ((k as i64 * k as i64) >> 32) as _,
        ]
    }
}
