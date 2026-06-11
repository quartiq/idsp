<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [UNRELEASED](https://github.com/quartiq/idsp/compare/dsp-process-v0.3.0...HEAD) - DATE

## [0.3.0](https://github.com/quartiq/idsp/compare/dsp-process-v0.2.0...dsp-process-v0.3.0) - 2026-06-11

### Added

* Typed `View`/`ViewMut` processing with explicit `FrameMajor` and `LaneMajor` layouts.
* `Lanes`, `ByLane`, and `PerFrame` adapters for layout-sensitive block processing.
* `Downsample`, `Hold`, `TryDecimator`, `DecimatorError`, and `ChunkOutPod`.
* Convenience adapters on `Split`: `map`, `chunk`, `interpolate`, `decimate`,
  `try_decimate`, `per_frame`, `lanes`, and `by_lane`.
* Optional `bytemuck` feature for POD chunk-output flattening.

### Changed

* Rename channel-oriented composition to lane-oriented APIs.
* Harden chunk regrouping adapters with explicit ratio checks.
* The optional `bytemuck` feature no longer enables bytemuck defaults.

## [0.2.0](https://github.com/quartiq/idsp/compare/dsp-process-v0.1.0...dsp-process-v0.2.0) - 2026-02-11

## [0.1.0](https://github.com/quartiq/idsp/compare/v0.19.0...dsp-process-v0.2.0) - 2026-01-13
