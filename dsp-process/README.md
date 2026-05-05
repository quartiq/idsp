# `dsp-process`

`dsp-process` provides small `no_std` processing traits and composition adapters
for DSP code that needs to stay explicit about state, memory layout, and hot-path
costs.

It was extracted from [`idsp`](https://github.com/quartiq/idsp), where the same
abstractions are used to build fixed-point and floating-point filters for embedded
and real-time signal-processing pipelines. The crate is intended for code that
cares about:

* predictable code generation
* `no_std` and no allocation
* separating immutable configuration from mutable runtime state
* composing filters without hiding the data layout
* sharing one configuration across many lanes or states

The root `idsp` repository also carries composite examples that show how these
primitives fit together in real DSP graphs.

This README is the crate-level documentation via `#![doc = include_str!(...)]`.

## Mission

The core idea is simple: treat a DSP stage as a tiny object with a `process()`
method, and make composition cheap enough that it still works in embedded hot
paths.

The crate is deliberately narrower than general stream-processing frameworks. It
does not try to model async execution, dynamic graphs, allocation, buffering
ownership, or runtime scheduling. It focuses on synchronous sample/slice
processing and on layouts that map cleanly to loops the compiler can optimize.

## Core Traits

The four main traits are:

* [`Process`]: stateful single-input processing, `&mut self`
* [`Inplace`]: stateful in-place processing
* [`SplitProcess`]: immutable configuration plus separate mutable state
* [`SplitInplace`]: in-place variant of `SplitProcess`

`SplitProcess` is the distinctive part of the crate. It lets one configuration be
reused across many independent states, which is a good fit for multi-lane DSP,
I/Q processing, polyphase banks, or coefficient sharing in embedded systems.

## Basic Example

```rust
use dsp_process::{Process, Split, Offset, Gain};

let mut offset = Split::stateless(Offset(3));
assert_eq!(offset.process(5), 8);

let mut gain = Split::stateless(Gain(4));
assert_eq!(gain.process(5), 20);
```

## Composition Example

Serial composition uses tuples or arrays. Here two stateless stages are combined
into one processor:

```rust
use dsp_process::{Process, Split, Offset, Gain};

let mut pipeline = (Split::stateless(Offset(3)) * Split::stateless(Gain(4))).minor();
assert_eq!(pipeline.process(5), 32);
```

The tuple/array implementations cover common static compositions without dynamic
dispatch or heap allocation.

## Split Configuration and Shared Coefficients

A single filter configuration can be applied to multiple states:

```rust
use dsp_process::{Process, Split, Offset};

let mut lanes = Split::stateless(Offset(3)).lanes::<2>();
assert_eq!(lanes.process([1, 2]), [4, 5]);
assert_eq!(lanes.process([10, 20]), [13, 23]);
```

This is one of the main reasons the crate exists: the split form makes sharing
configuration explicit instead of forcing each lane to own a full copy.

## Closures and Adapters

`FnProcess` and `FnSplitProcess` let closures participate when that is the
clearest representation:

```rust
use dsp_process::{Process, FnProcess};

let mut abs = FnProcess(|x: i32| x.abs());
assert_eq!(abs.process(-7), 7);
```

Adapters such as [`Chunk`], [`ChunkIn`], [`ChunkOut`], [`ChunkInOut`],
[`Interpolator`], [`Decimator`], and [`Map`] help lift sample processors to chunk
processors and back without changing the underlying stage.

## Layout and Composition Modes

The crate supports several composition styles:

* plain tuples and arrays for straightforward serial composition
* [`Minor`] for processor-minor/data-major composition
* [`Major`] for slice processing with explicit intermediate scratch
* [`Parallel`] for parallel branches
* [`Lanes`] for many states with shared configuration
* [`ByLane`] for explicit lane-major view processing

These are not interchangeable in performance terms. The point is to make the
choice explicit.

`Minor` tends to fit processors with small state and configuration. `Major` is
for cases where slice processing and explicit intermediate storage improve cache
behavior or register pressure. `Lanes` and `ByLane` pair with typed views so
multi-lane locality and vectorization can be expressed explicitly.

## Context in `idsp`

Inside `idsp`, these traits are used to express IIR sections, half-band filters,
decimators, interpolators, low-pass stages, and lock-in style pipelines, often on
fixed-point integer data and often with strong embedded constraints. The crate is
therefore biased toward:

* static composition
* small reusable building blocks
* explicit scratch buffers
* low ceremony in `no_std`

If your problem is “I have a hot DSP inner loop and want composition without
giving up control”, this crate is aimed at that.

## Benefits and Unique Selling Points

Compared with hand-written monolithic loops, `dsp-process` gives:

* reusable composition primitives instead of copy-pasted loop nests
* explicit split config/state for coefficient sharing
* static dispatch and fixed layouts instead of runtime graph machinery
* adapters for common DSP reshaping tasks such as chunking, interpolation, and
  decimation

Compared with more general iterator or stream-style APIs, it keeps:

* `no_std` support
* no allocation requirement
* layout control as part of the API
* a closer mapping between the public abstraction and the generated loop nest

## Costs and Limitations

This crate is intentionally not beginner-friendly in every corner.

* The API is low level. Callers are expected to understand sample, slice, and
  view layout.
* The traits are designed for hot paths, so some contracts are preconditions
  rather than dynamically checked ergonomic errors.
* Static composition means type signatures can become large.
* Some adapters rely on const-generic shape relations that are correct but not
  always obvious at the callsite.
* `Lanes` and `ByLane` use explicit views for layout-sensitive view
  processing, which is more precise but also a more advanced API than ordinary
  ordinary slice use.

In short: this crate optimizes for control and performance first, convenience
second.

## Alternatives

Concisely:

* Hand-written loops: maximal control, minimal reuse. Often best for a single
  kernel, worse once many variants or compositions need to stay consistent.
* Iterator-heavy style: concise for non-hot code, usually a poor fit when state,
  aliasing, and exact loop shape matter.
* Dynamic flowgraph runtimes such as GNU Radio or FutureSDR: better when the
  graph, scheduling policy, runtime reconfiguration, or heterogeneous execution
  are part of the problem. Those frameworks operate at a higher level around
  blocks, buffers, message passing, and schedulers; `dsp-process` is much lower
  level and closer to the inner kernels inside such blocks.
* Dynamic in-process graph libraries such as `dasp_graph`: better when nodes and
  edges must be edited at runtime. `dsp-process` instead assumes fixed topology
  and avoids graph ownership and scheduler concerns entirely.
* Static signal or graph composition libraries such as `dasp_signal` or FunDSP:
  those also support static composition, but they focus on frame/signal or
  audio-node abstractions. `dsp-process` is narrower: split config/state,
  explicit view layout, `no_std`, and predictable loop shape are the center of
  the design.
* Plain `Process`-only designs: simpler surface, but weaker support for
  coefficient sharing across many states.

`dsp-process` is most useful in the middle ground: fixed topology, static
composition, explicit state, and performance-sensitive DSP.

## Notes on `Lanes` and `ByLane`

`Lanes` and `ByLane` are layout tools, not just semantic conveniences. Their
layout-sensitive view behavior is explicit through
[`View<_, _, LaneMajor, _>`] and [`ViewMut<_, _, LaneMajor, _>`], so the
lane-major interpretation is visible at the type level instead of being
silently inferred from `[[X; N]]`.

```rust
use dsp_process::{LaneMajor, Offset, Split, View, ViewMut, ViewProcess};

let mut p = Split::stateless(Offset(3)).lanes::<2>();
let x = View::<_, LaneMajor, 2>::from_flat(&[1, 2, 3, 10, 20, 30], 3);
let mut y = [0; 6];
let yb = ViewMut::<_, LaneMajor, 2>::from_flat(&mut y, 3);
ViewProcess::process_view(&mut p, x, yb);
assert_eq!(y, [4, 5, 6, 13, 23, 33]);
```

Chunk adapters remain useful alongside typed views. Use [`Split::per_frame()`]
and then call `process_frames()` or `inplace_frames()` on the resulting
processor to apply a chunk processor frame by frame to a frame-major view.

## Guidance for Implementors

As a rule of thumb:

* implement [`SplitProcess<X, Y, ()>`] for stateless/config-only processors
* implement [`Process`] for processors that carry all state internally
* implement [`SplitInplace`] or [`Inplace`] when a true in-place specialization
  exists
* override `block()` only when it meaningfully improves the loop shape or memory
  traffic

## Small Reference Example

```rust
use dsp_process::{Buffer, Inplace, Process};

let mut dly = Buffer::<[i32; 2]>::default();
let y: i32 = dly.process(10);
assert_eq!(y, 0);
let y: i32 = dly.process(20);
assert_eq!(y, 0);
let y: i32 = dly.process(30);
assert_eq!(y, 10);

let mut block = [1, 2, 3];
dly.inplace(&mut block);
assert_eq!(block, [20, 30, 1]);
```
