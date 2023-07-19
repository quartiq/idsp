pub trait Filter {
    type Config;
    /// Update the filter with a new sample.
    ///
    /// # Args
    /// * `x`: Input data.
    /// * `k`: Filter configuration.
    ///
    /// # Return
    /// Filtered output y.
    fn update(&mut self, x: i32, k: &Self::Config) -> i32;
    /// Return the current filter output
    fn get(&self) -> i32;
    /// Update the filter so that it outputs the provided value.
    /// This does not completely define the state of the filter.
    fn set(&mut self, x: i32);
}

#[derive(Copy, Clone, Default)]
pub struct Nyquist(pub(crate) i32);
impl Filter for Nyquist {
    type Config = ();
    fn update(&mut self, x: i32, _k: &Self::Config) -> i32 {
        let x = x >> 1; // x/2 for less bias but more distortion
        let y = x.wrapping_add(self.0);
        self.0 = x;
        y
    }
    fn get(&self) -> i32 {
        self.0
    }
    fn set(&mut self, x: i32) {
        self.0 = x;
    }
}

#[derive(Copy, Clone)]
pub struct Chain<const N: usize, T>(pub(crate) [T; N]);
impl<const N: usize, T: Filter> Filter for Chain<N, T> {
    type Config = T::Config;
    fn update(&mut self, x: i32, k: &Self::Config) -> i32 {
        self.0.iter_mut().fold(x, |x, stage| stage.update(x, k))
    }
    fn get(&self) -> i32 {
        self.0[N - 1].get()
    }
    fn set(&mut self, x: i32) {
        self.0.iter_mut().for_each(|stage| stage.set(x));
    }
}
impl<const N: usize, T: Default + Copy> Default for Chain<N, T> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

/// Arbitrary order, high dynamic range, wide coefficient range,
/// lowpass filter implementation. DC gain is 1.
///
/// Type argument N is the filter order.
#[derive(Copy, Clone)]
pub struct Lowpass<const N: usize>(pub(crate) [i64; N]);
impl<const N: usize> Filter for Lowpass<N> {
    type Config = [i32; N];
    fn update(&mut self, x: i32, k: &Self::Config) -> i32 {
        let mut dy;
        if N >= 2 {
            dy = self.0[1];
            dy -= (self.0[1] >> 32) * k[1] as i64;
            dy += x.wrapping_sub((self.0[0] >> 32) as i32) as i64 * k[0] as i64;
            self.0[1] = dy;
        } else {
            dy = x.wrapping_sub((self.0[0] >> 32) as i32) as i64 * k[0] as i64;
        }
        self.0[0] += dy;
        let y0 = self.get();
        self.0[0] += dy;
        y0
    }
    fn get(&self) -> i32 {
        (self.0[0] >> 32) as i32
    }
    fn set(&mut self, x: i32) {
        self.0[0] = (x as i64) << 32;
    }
}

impl<const N: usize> Default for Lowpass<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

pub type Lowpass1 = Lowpass<1>;
pub type Lowpass2 = Lowpass<2>;
