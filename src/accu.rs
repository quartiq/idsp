use core::ops::AddAssign;

/// Accumulator
///
/// Use [`core::num::Wrapping`].
///
/// ```
/// use core::num::Wrapping;
/// use idsp::Accu;
/// let mut a = Accu::new(Wrapping(0i8), Wrapping(127));
/// assert_eq!(a.next(), Some(Wrapping(127)));
/// assert_eq!(a.next(), Some(Wrapping(-2)));
/// ```
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
pub struct Accu<T> {
    /// Current accumulator state
    pub state: T,
    /// Step
    pub step: T,
}

impl<T> Accu<T> {
    /// Create a new accumulator with given initial state and step.
    pub const fn new(state: T, step: T) -> Self {
        Self { state, step }
    }
}

impl<T: Copy> Iterator for Accu<T>
where
    T: AddAssign,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.state += self.step;
        Some(self.state)
    }
}
