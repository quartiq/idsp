# `dsp-process`

Small `no_std` traits for static DSP composition. State, layout, and loop shape
remain explicit; no allocation or dynamic dispatch is required.

## Processing shapes

| Shape | Representation | Use |
| --- | --- | --- |
| one config, one state | `Split<C, S>` | ordinary stateful filter |
| one config, many states | `Split<C, S>::lanes()` | shared coefficients |
| many configs, one state | direct `SplitProcess` calls | prediction/correction phases |
| many config/state pairs | tuples, arrays, `Minor`, `Major` | static pipelines |

`SplitProcess<X, Y, S>` is the primitive: `&self` is immutable configuration;
`&mut S` is caller-owned state. Different configurations may therefore operate
on the same state:

```rust
use dsp_process::SplitProcess;

struct State(i32);
struct Predict(i32);
struct Correct;

impl SplitProcess<(), (), State> for Predict {
    fn process(&self, state: &mut State, (): ()) {
        state.0 += self.0;
    }
}

impl SplitProcess<i32, i32, State> for Correct {
    fn process(&self, state: &mut State, measurement: i32) -> i32 {
        state.0 = (state.0 + measurement) / 2;
        state.0
    }
}

let mut state = State(2);
Predict(3).process(&mut state, ());
assert_eq!(Correct.process(&mut state, 9), 7);
```

This is useful when processing has distinct phases but one state, such as a
Kalman transition and observation. Tuple composition instead gives each stage
its own state.

## Owned processors

`Split<C, S>` binds one configuration to one state and implements `Process`:

```rust
use dsp_process::{Offset, Process, Split};

let mut offset = Split::stateless(Offset(3));
assert_eq!(offset.process(5), 8);
```

One configuration can drive several independent states:

```rust
use dsp_process::{Offset, Process, Split};

let mut lanes = Split::stateless(Offset(3)).lanes::<2>();
assert_eq!(lanes.process([1, 10]), [4, 13]);
```

## Composition

Tuples and arrays form serial pipelines. `Parallel` forms branches. `Minor` and
`Major` select loop nesting and scratch placement.

```rust
use dsp_process::{Gain, Offset, Process, Split};

let mut pipeline = (Split::stateless(Offset(3)) * Split::stateless(Gain(4))).minor();
assert_eq!(pipeline.process(5), 32);
```

Adapters (`Chunk`, `ChunkIn`, `ChunkOut`, `Interpolator`, `Decimator`, `Map`)
change rate or call shape without hiding state.

## Layout

Layout-sensitive processing uses typed views. `FrameMajor` and `LaneMajor`
make the physical interpretation part of the type; sample newtypes do not.

```rust
use dsp_process::{LaneMajor, Offset, Split, View, ViewMut, ViewProcess};

let mut lanes = Split::stateless(Offset(3)).lanes::<2>();
let input = View::<_, LaneMajor, 2>::from_flat(&[1, 2, 3, 10, 20, 30], 3);
let mut output = [0; 6];
let output = ViewMut::<_, LaneMajor, 2>::from_flat(&mut output, 3);
lanes.process_view(input, output);
```

Use `Split::per_frame()` to apply a chunk processor to each frame of a
frame-major view.

## Implementing a stage

- Implement `SplitProcess` when configuration and state are distinct.
- Implement `Process` when one value naturally owns both.
- Implement `SplitInplace` or `Inplace` for a real in-place specialization.
- Override `block()` only to improve the loop or memory traffic.

Runtime fields hold values; const generics encode shape; wrappers encode
composition and layout.
