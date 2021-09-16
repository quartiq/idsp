# Embedded DSP algorithms

[![GitHub release](https://img.shields.io/github/v/release/quartiq/idsp?include_prereleases)](https://github.com/quartiq/idsp/releases)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://docs.rs/idsp)
[![QUARTIQ Matrix Chat](https://img.shields.io/matrix/quartiq:matrix.org)](https://matrix.to/#/#quartiq:matrix.org)
[![Continuous Integration](https://github.com/quartiq/idsp/actions/workflows/ci.yml/badge.svg)](https://github.com/quartiq/idsp/actions/workflows/ci.yml)

This crate contains some tuned DSP algorithms for general and especially embedded use.
Many of the algorithms are implemented on integer datatypes for several reasons that become important in certain cases:

* Speed: even with a hard FP unit integer operations are faster.
* Accuracy: single precision FP has a 24 bit mantissa, `i32` has full 32 bit.
* No rounding errors.
* Natural wrap around (modulo) at the integer overflow: critical for phase/frequency applications.
* Natural definition of "full scale".

One comprehensive user for these algorithms is [Stabilizer](https://github.com/quartiq/stabilizer).

# Cosine/Sine

This uses a small (128 element or 512 byte LUT), smart octant (un)mapping, linear interpolation and comprehensive analysis of corner cases to achieve a very clean signal (4e-6 RMS error, 9e-6 max error, 108 dB SNR), low spurs, and no bias with about 40 cortex-m instruction per call. It computes both cosine and sine (i.e. the complex signal) at once given a phase input.

# atan2

This returns a phase given a complex signal (a pair of in-phase/`x`/cosine and quadrature/`y`/sine), The RMS phase error is less than 3e-3, max error is 5e-3, relative phase error decreases further near the octant cuts.

# ComplexExt

An extension trait for the `num::Complex` type featuring especially a `std`-like API to the two functions above.

# PLL, RPLL

High accuracy, zero-assumption, fully robust, forward and reciprocal PLLs with dynamically adjustable time constant and arbitrary capture range.

# Unwrapper, Accu, saturating_scale

Tools to handle, track, and unwrap phase signals or generate them.

# iir_int, iir

`i32` and `f32` biquad IIR filters with properly implemented clipping and offset (anti-windup, no derivative kick, dynamically adjustable gains).
