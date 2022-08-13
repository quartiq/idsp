use num_traits::ops::wrapping::WrappingAdd;

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct Accu<T> {
    state: T,
    step: T,
}

impl<T> Accu<T> {
    pub fn new(state: T, step: T) -> Self {
        Self { state, step }
    }
}

impl<T> Iterator for Accu<T>
where
    T: WrappingAdd + Copy,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        let s = self.state;
        self.state = s.wrapping_add(&self.step);
        Some(s)
    }
}
