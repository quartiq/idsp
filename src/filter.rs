/// Single inpout single output i32 filter
pub trait Filter {
    /// Filter configuration type.
    ///
    /// While the filter struct owns the state,
    /// the configuration is decoupled to allow sharing.
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

/// Nyquist zero
///
/// Filter with a flat transfer function and a transfer function zero at Nyquist.
#[derive(Copy, Clone, Default)]
pub struct Nyquist(i32);
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

/// Repeat another filter
#[derive(Copy, Clone)]
pub struct Repeat<const N: usize, T>([T; N]);
impl<const N: usize, T: Filter> Filter for Repeat<N, T> {
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
impl<const N: usize, T: Default + Copy> Default for Repeat<N, T> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

/// Combine two different filters in cascade
#[derive(Copy, Clone, Default)]
pub struct Cascade<T, U>(T, U);
impl<T: Filter, U: Filter> Filter for Cascade<T, U> {
    type Config = (T::Config, U::Config);
    fn update(&mut self, x: i32, k: &Self::Config) -> i32 {
        self.1.update(self.0.update(x, &k.0), &k.1)
    }
    fn get(&self) -> i32 {
        self.1.get()
    }
    fn set(&mut self, x: i32) {
        self.1.set(x)
    }
}
