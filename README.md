# Embedded DSP algorithms

[![GitHub release](https://img.shields.io/github/v/release/quartiq/idsp?include_prereleases)](https://github.com/quartiq/idsp/releases)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://docs.rs/idsp)
[![QUARTIQ Matrix Chat](https://img.shields.io/matrix/quartiq:matrix.org)](https://matrix.to/#/#quartiq:matrix.org)
[![Continuous Integration](https://github.com/quartiq/idsp/actions/workflows/ci.yml/badge.svg)](https://github.com/quartiq/idsp/actions/workflows/ci.yml)

This crate contains some tuned DSP algorithms for general and especially embedded use.
Many of the algorithms are implemented on integer datatypes for reasons that become important in certain cases:

* Speed: even with a hard floating point unit integer operations are faster.
* Accuracy: single precision FP has a 24 bit mantissa, `i32` has full 32 bit.
* No rounding errors.
* Natural wrap around (modulo) at the integer overflow: critical for phase/frequency applications.
* Natural definition of "full scale".

One comprehensive user for these algorithms is [Stabilizer](https://github.com/quartiq/stabilizer).

## Cosine/Sine `cossin`

This uses a small (128 element or 512 byte) LUT, smart octant (un)mapping, linear interpolation and comprehensive analysis of corner cases to achieve a very clean signal (4e-6 RMS error, 9e-6 max error, 108 dB SNR typ), low spurs, and no bias with about 40 cortex-m instruction per call. It computes both cosine and sine (i.e. the complex signal) at once given a phase input.

## Two-argument arcus-tangens `atan2`

This returns a phase given a complex signal (a pair of in-phase/`x`/cosine and quadrature/`y`/sine). The RMS phase error is less than 5e-6 rad, max error is less than 1.2e-5 rad, i.e. 20.5 bit RMS, 19.1 bit max accuracy. The bias is minimal.

## ComplexExt

An extension trait for the `num::Complex` type featuring especially a `std`-like API to the two functions above.

## PLL, RPLL

High accuracy, zero-assumption, fully robust, forward and reciprocal PLLs with dynamically adjustable time constant and arbitrary (in the Nyquist sampling sense) capture range, and noise shaping.

## Unwrapper, Accu, saturating_scale

Tools to handle, track, and unwrap phase signals or generate them.

## iir

Fixed point (`i8`, `i16`, `i32`, `i64`) and floating point (`f32`, `f64`) biquad IIR filters.
Robust and clean clipping and offset (anti-windup, no derivative kick, dynamically adjustable gains) suitable for PID controller applications.
Direct Form 1 and Direct Form 2 Transposed supported.
Coefficient sharing for multiple channels.

## Lowpass, Lockin

Fast, infinitely cascadable, first- and second-order lowpass and the corresponding integration into a lockin amplifier algorithm.

## FIR filters

Fast `f32` symmetric FIR filters, optimized half-band filters, half-band filter decimators and integators and cascades.

## SVF filter

Simple IIR state variable filter simultaneously providing highpass, lowpass,
bandpass, and notch filtering of a signal.
