use num_traits::ops::wrapping::WrappingAdd;

/// Wrapping Accumulator
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
pub struct Accu<T> {
    /// Current accumulator state
    pub state: T,
    /// Step
    pub step: T,
}

impl<T> Accu<T> {
    /// Create a new accumulator with given initial state and step.
    pub fn new(state: T, step: T) -> Self {
        Self { state, step }
    }
}

impl<T: WrappingAdd + Copy> Iterator for Accu<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        let s = self.state;
        self.state = s.wrapping_add(&self.step);
        Some(s)
    }
}
