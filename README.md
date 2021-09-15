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
* Natural wrap around at the integer overflow: critical for phase/frequency applications (`cossin`, `Unwrapper`, `atan2`, `PLL`, `RPLL` etc.)

The prime user for these algorithms is [Stabilizer](https://github.com/quartiq/stabilizer).