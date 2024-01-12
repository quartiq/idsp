//! State variable filter

use num_traits::{Float, FloatConst};
use serde::{Deserialize, Serialize};

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

    /// Update the filter
    ///
    /// Ingest an input sample and update state correspondingly.
    /// Selected output(s) are available from [`State`].
    pub fn update(&self, s: &mut State<T>, x0: T) {
        s.lp = s.bp * self.f + s.lp;
        s.hp = x0 - s.lp - s.bp * self.q;
        s.bp = s.hp * self.f + s.bp;
    }
}
