# Embedded DSP algorithms

[![GitHub release](https://img.shields.io/github/v/release/quartiq/idsp?include_prereleases)](https://github.com/quartiq/idsp/releases)
[![crates.io](https://img.shields.io/crates/v/idsp.svg)](https://crates.io/crates/idsp)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://docs.rs/idsp)
[![QUARTIQ Matrix Chat](https://img.shields.io/matrix/quartiq:matrix.org)](https://matrix.to/#/#quartiq:matrix.org)
[![Continuous Integration](https://github.com/quartiq/idsp/actions/workflows/ci.yml/badge.svg)](https://github.com/quartiq/idsp/actions/workflows/ci.yml)

This crate contains some tuned DSP algorithms for general and especially embedded use.
Many of the algorithms are implemented on integer (fixed point) datatypes.

One comprehensive user for these algorithms is [Stabilizer](https://github.com/quartiq/stabilizer).

## Fixed point

### Cosine/Sine

[`cossin()`] uses a small (128 element or 512 byte) LUT, smart octant (un)mapping, linear interpolation and comprehensive analysis of corner cases to achieve a very clean signal (4e-6 RMS error, 9e-6 max error, 108 dB SNR typ), low spurs, and no bias with about 40 cortex-m instruction per call. It computes both cosine and sine (i.e. the complex signal) at once given a phase input.

### Two-argument arcus-tangens

[`atan2()`] returns a phase given a complex signal (a pair of in-phase/`x`/cosine and quadrature/`y`/sine). The RMS phase error is less than 5e-6 rad, max error is less than 1.2e-5 rad, i.e. 20.5 bit RMS, 19.1 bit max accuracy. The bias is minimal.

## PLL, RPLL

[`PLL`], [`RPLL`]: High accuracy, zero-assumption, fully robust, forward and reciprocal PLLs with dynamically adjustable time constant and arbitrary (in the Nyquist sampling sense) capture range, and noise shaping.

## `Unwrapper`, `Accu`, `saturating_scale()`

[`Unwrapper`], [`Accu`], [`saturating_scale()`]:
Tools to handle, track, and unwrap phase signals or generate them.

## Float and Fixed point

## IIR/Biquad

[`iir::Sos`] are fixed point (`i8`, `i16`, `i32`, `i64`) and floating point (`f32`, `f64`) biquad IIR filters.
Robust and clean clipping and offset (anti-windup, no derivative kick, dynamically adjustable gains and gain limits) suitable for PID controller applications.
Three kinds of filter actions: Direct Form 1, Direct Form 2 Transposed, Direct Form 1 with noise shaping, and Direct Form 1 with wide output storage are supported.
Coefficient sharing for multiple channels is supported through [`dsp_process::SplitProcess`], [`dsp_process::Channels`].

### Comparison

This is a rough feature comparison of several available `biquad` crates, with no claim for completeness, accuracy, or even fairness.
TL;DR: `idsp` is slower but offers more features.

| Feature\Crate | [`biquad-rs`](https://crates.io/crates/biquad) | [`fixed-filters`](https://crates.io/crates/fixed-filters) | `idsp::iir` |
|---|---|---|---|
| Floating point `f32`/`f64` | ✅ | ❌ | ✅ |
| Fixed point `i32` | ❌ | ✅ | ✅ |
| Parametric fixed point `i32` | ❌ | ✅ | ❌ |
| Fixed point `i8`/`i16`/`i64`/`i128` | ❌ | ❌ | ✅ |
| DF2T | ✅ | ❌ | ✅ |
| Limiting/Clamping | ❌ | ✅ | ✅ |
| Fixed point accumulator guard bits | ❌ | ❌ | ✅ |
| Summing junction offset | ❌ | ❌ | ✅ |
| Fixed point noise shaping | ❌ | ❌ | ✅ |
| Configuration/state decoupling/multi-channel | ❌ | ❌ | ✅ |
| `f32` parameter audio filter builder | ✅ | ✅ | ✅ |
| `f64` parameter audio filter builder | ✅ | ❌ | ✅ |
| Additional filters (I/HO) | ❌ | ❌ | ✅ |
| `f32` PI builder | ❌ | ✅ | ✅ |
| `f32/f64` PI²D² builder | ❌ | ❌ | ✅ |
| PI²D² builder limits | ❌ | ❌ | ✅ |
| Support for fixed point `a1=-2` | ❌ | ❌ | ✅ |

Three crates have been compared when processing 4x1M samples (4 channels) with a biquad lowpass.
Hardware was `thumbv7em-none-eabihf`, `cortex-m7`, code in ITCM, data in DTCM, caches enabled.

| Crate | Type, features | Cycles per sample |
|---|---|---|
| [`biquad-rs`](https://crates.io/crates/biquad) | `f32` | 11.4 |
| `idsp::iir` | `f32`, limits, offset | 15.5 |
| [`fixed-filters`](https://crates.io/crates/fixed-filters) | `i32`, limits | 20.3 |
| `idsp::iir` | `i32`, limits, offset | 23.5 |
| `idsp::iir` | `i32`, limits, offset, noise shaping | 30.0 |

## State variable filter

[`svf`] is a simple IIR state variable filter simultaneously providing highpass, lowpass,
bandpass, and notch filtering of a signal.

## `Lowpass`, `Lockin`

[`Lowpass`], [`Lockin`] are fast, infinitely cascadable, first- and second-order lowpass and the corresponding integration into a lockin amplifier algorithm.

## FIR filters

[`hbf::HbfDec`], [`hbf::HbfInt`], [`hbf::HbfDec32`], [`hbf::HbfInt32`]:
Fast `f32` symmetric FIR filters, optimized half-band filters, half-band filter decimators and integators and cascades.
These are used in [`stabilizer-stream`](https://github.com/quartiq/stabilizer-stream) for online PSD calculation for
arbitrarily low offset frequencies.

## Delta Sigma modulator/noise shaper

[`Dsm`] is a delta sigma modulator/noise shaper in MASH-(1)^K architecture.
