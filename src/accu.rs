use core::ops::{Add, AddAssign, Mul, Sub};

use dsp_fixedpoint::Q;

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

impl<T: Copy, A, const F: i8> Mul<T> for Accu<Q<T, A, F>>
where
    Q<T, A, F>: Mul<T, Output = Q<T, A, F>>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.state * rhs, self.step * rhs)
    }
}

impl<T: Copy, A, const F: i8> Mul<Q<T, A, F>> for Accu<Q<T, A, F>>
where
    Q<T, A, F>: Mul<Output = Q<T, A, F>>,
{
    type Output = Self;

    fn mul(self, rhs: Q<T, A, F>) -> Self::Output {
        Self::new(self.state * rhs, self.step * rhs)
    }
}

impl<T: Copy + Add<Output = T>> Add for Accu<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.state + rhs.state, self.step + rhs.step)
    }
}

impl<T: Copy + Sub<Output = T>> Sub for Accu<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.state - rhs.state, self.step - rhs.step)
    }
}
