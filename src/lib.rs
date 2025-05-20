#![cfg_attr(not(any(test, doctest, feature = "std")), no_std)]
#![doc = include_str!("../README.md")]
#![deny(rust_2018_compatibility)]
#![deny(rust_2018_idioms)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod atan2;
pub use atan2::*;
mod accu;
pub use accu::*;
mod filter;
pub use filter::*;
mod complex;
pub use complex::*;
mod cossin;
pub use cossin::*;
pub mod iir;
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
pub mod hbf;
mod num;
pub use num::*;
mod dsm;
pub mod svf;
pub use dsm::*;
mod sweptsine;
pub use sweptsine::*;
mod cic;
pub use cic::*;
mod cordic;
pub use cordic::*;

#[cfg(test)]
pub mod testing;
