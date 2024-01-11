use num_traits::{Float, FloatConst};
use serde::{Deserialize, Serialize};
pub struct State<T> {
    pub lp: T,
    pub hp: T,
    pub bp: T,
}

impl<T: Float> State<T> {
    pub fn br(&self) -> T {
        self.hp + self.lp
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct Svf<T> {
    f: T,
    q: T,
}

impl<T: Float + FloatConst> Svf<T> {
    pub fn set_frequency(&mut self, f0: T) {
        self.f = (T::one() + T::one()) * (T::PI() * f0).sin();
    }

    pub fn set_q(&mut self, q: T) {
        self.q = T::one()/q;
    }

    pub fn update(&self, s: &mut State<T>, x0: T) {
        s.lp = s.bp * self.f + s.lp;
        s.hp = x0 - s.lp - s.bp * self.q;
        s.bp = s.hp * self.f + s.bp;
    }
}
