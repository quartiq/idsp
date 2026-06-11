<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED](https://github.com/quartiq/idsp/compare/dsp-fixedpoint-v0.1.1...HEAD) - DATE

### Added

* Optional `bytemuck` support for transparent fixed-point wrappers.
* Optional `defmt::Format` support.
* Fixed-point binary, octal, and hexadecimal dot formatting.
* `num-traits` support for bounds, primitive conversion, signed values, and parsing.
* `mul_wide()` and `apply()` to spell widened multiplication and quantized gain application explicitly.
* Serde adapters `serde::as_f32` and `serde::as_f64` for lossy scaled values.

### Changed

* `serde` is no longer enabled by default.
* Default builds no longer pull `libm`.
* `Q::one()` and `Q::ONE` are only available when `1` is exactly representable.
* Make `Q` raw storage private; use `from_bits()` and `into_bits()` for raw representation access.

## [0.1.1](https://github.com/quartiq/idsp/compare/dsp-fixedpoint-v0.1.0...dsp-fixedpoint-v0.1.1) - 2026-04-06

## [0.1.0](https://github.com/quartiq/idsp/compare/v0.19.0...dsp-fixedpoint-v0.1.0) - 2026-01-13
