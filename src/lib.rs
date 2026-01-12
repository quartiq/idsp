#![cfg_attr(not(any(test, doctest, feature = "std")), no_std)]
#![doc = include_str!("../README.md")]

mod atan2;
pub use atan2::*;
mod accu;
pub use accu::*;
mod complex;
pub use complex::*;
mod cossin;
pub use cossin::*;
mod lockin;
pub use lockin::*;
mod lowpass;
pub use lowpass::*;
mod pll;
pub use pll::*;
mod rpll;
pub use rpll::*;
mod unwrap;
pub use unwrap::*;
mod dsm;
pub use dsm::*;
pub mod hbf;
pub mod iir;
mod sweptsine;
pub use sweptsine::*;
mod cic;
pub use cic::*;
mod cordic;
pub use cordic::*;
mod num;
pub use num::*;

#[cfg(feature = "py")]
mod py;

#[cfg(test)]
pub mod testing;
