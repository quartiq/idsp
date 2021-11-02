# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
### Changed
### Removed

## [0.3.0] - 2021-11-02

### Removed
* Removed `nightly` feature as it was broken in 0.2.0 and hard to fix with
  generics. Instead use `num_cast::clamp`.

## [0.2.0] - 2021-11-01

### Changed
* IIR is now generic over the float type (f32 and f64)

## [0.1.0] - 2021-09-15

Library initially released on crates.io

[Unreleased]: https://github.com/quartiq/idsp/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/quartiq/idsp/releases/tag/v0.3.0
[0.2.0]: https://github.com/quartiq/idsp/releases/tag/v0.2.0
[0.1.0]: https://github.com/quartiq/idsp/releases/tag/v0.1.0
