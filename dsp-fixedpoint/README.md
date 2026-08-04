# `dsp-fixedpoint`

`no_std` fixed-point arithmetic with explicit storage, accumulator, and
quantization boundaries.

## Type shape

`Q<T, A, F>` stores raw `T` bits scaled by `2^-F`; products accumulate in `A`.

- `T`: stored and signal value
- `A`: wide accumulator
- `F`: fractional bits

Aliases include `Q32<F> = Q<i32, i64, F>` and wrapping/unsigned variants.

## Wide MAC, late quantization

Put the coefficient on the left and the raw signal on the right:

```rust
use dsp_fixedpoint::Q32;

type C = Q32<28>;
let a = [C::from_f32(0.25), C::from_f32(-0.5)];
let x = [100_i32, 20];

let mut acc = a[0] * x[0]; // Q<i64, i32, 28>
acc += a[1] * x[1];        // still wide
let y = acc.quantize();     // one shift and narrowing conversion

assert_eq!(y, 15);
```

The shape is `Q<T, A, F> * T -> Q<A, T, F>`. Accumulate products of that type,
then call `quantize()` once. Reversing the operands, `T * Q -> T`, quantizes each
product immediately.

## Construction

`from_int` and `from_f32`/`from_f64` construct scaled values; `from_bits`
preserves raw bits. `FromRatio` constructs a coefficient from two raw values:

```rust
use dsp_fixedpoint::{FromRatio, Q32};

let gain = Q32::<28>::from_ratio(1, 3);
assert!((gain.as_f64() - 1.0 / 3.0).abs() < Q32::<28>::DELTA as f64);
```

For positive `F`, `FromRatio` widens the numerator before scaling and division.
The denominator must be nonzero and the result must fit `T`.

## Operators

| Operation | Result | Quantization |
| --- | --- | --- |
| `Q * T` | `Q<A, T, F>` | none; wide product |
| `T * Q` | `T` | immediate |
| `Q * Q`, `Q / Q` | `Q<T, A, F>` | immediate |
| `Q + Q`, `Q - Q` | `Q<T, A, F>` | none; equal `F` required |
| `T / Q` | `T` | immediate |

`mul_wide()` and `apply()` spell `Q * T` and `T * Q` explicitly.

## Scale and conversion

Use `.scale::<F1>()` when changing fractional bits. Numeric traits convert the
represented value; `from_bits`/`into_bits` access the representation.

`Q::<_, _, -128>::DELTA` is invalid. `One` is available only when `1` is exactly
representable by `T` and `F`.

## Optional integrations

- `serde`: transparent raw representation; `serde::as_f32`/`as_f64` for scaled
  values.
- `defmt`: compact decimal logging through `f32`.
- `bytemuck`: `Pod`, `Zeroable`, and `TransparentWrapper` when the component
  types permit them.

`Display` is decimal. Binary, octal, and hexadecimal formatting place the radix
point according to `F`.
