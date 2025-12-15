//! State variable filter

use num_traits::{Float, FloatConst};
use serde::{Deserialize, Serialize};

use crate::process::{Process, Split};

/// Second order state variable filter state
pub struct State<T> {
    /// Lowpass output
    pub lp: T,
    /// Highpass output
    pub hp: T,
    /// Bandpass output
    pub bp: T,
}

impl<T: Float> State<T> {
    /// Bandreject (notch) output
    pub fn br(&self) -> T {
        self.hp + self.lp
    }
}

/// State variable filter
///
/// <https://www.earlevel.com/main/2003/03/02/the-digital-state-variable-filter/>
#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct Svf<T> {
    f: T,
    q: T,
}

impl<T: Float + FloatConst> Svf<T> {
    /// Set the critical frequency
    ///
    /// In units of the sample frequency.
    pub fn set_frequency(&mut self, f0: T) {
        self.f = (T::one() + T::one()) * (T::PI() * f0).sin();
    }

    /// Set the Q parameter
    pub fn set_q(&mut self, q: T) {
        self.q = T::one() / q;
    }
}

impl<T> Process<T, ()> for Split<&Svf<T>, &mut State<T>>
where
    T: Float + FloatConst + Copy,
{
    /// Update the filter
    ///
    /// Ingest an input sample and update state correspondingly.
    /// Selected output(s) are available from [`State`].
    fn process(&mut self, x: T) {
        self.state.lp = self.state.bp * self.config.f + self.state.lp;
        self.state.hp = x - self.state.lp - self.state.bp * self.config.q;
        self.state.bp = self.state.hp * self.config.f + self.state.bp;
    }
}
