# Fixed point primitives

`dsp-fixedpoint` provides small `no_std` fixed-point primitives with explicit
integer storage and widening accumulators.

## Model

`Q<T, A, F>` stores a raw integer `T` and interprets it as scaled by `2^-F`.

- `T` is the storage type.
- `A` is the widened accumulator type used for intermediate results.
- `F` is the number of fractional bits.

Type aliases cover the common signed, unsigned, and wrapping pairs:
`Q8/Q16/Q32/Q64`, `P8/P16/P32/P64`, `W8/W16/W32/W64`, and `V8/V16/V32/V64`.

## Construction

There are two construction modes:

- `from_int`, `from_f32`, and `from_f64` scale into the fixed-point domain.
- `new` and `from_bits` are raw-representation constructors.

```rust
use dsp_fixedpoint::Q8;

let scaled = Q8::<4>::from_int(3);
let raw = Q8::<4>::from_bits(3 << 4);

assert_eq!(scaled, raw);
assert_eq!(raw.into_inner(), 48);
```

## Operator semantics

The crate keeps addition-like operators conservative and multiplication-like
operators efficient:

- `Q<F> + Q<F>` and similar operators require the same `F`.
- Use `.scale::<F1>()` explicitly before `Add`, `Sub`, or `Rem` when scales differ.
- `Q * Q -> Q` and `Q / Q -> Q` preserve the left-hand scale.
- `Q * T -> Q<A, T, F>` widens into the accumulator domain.
- `T * Q -> T` quantizes back into the base integer domain.
- `Q / T -> Q`, while `T / Q -> T`.

That asymmetry is intentional: operand order chooses whether the result remains
wide or is quantized immediately.

## Scale restrictions

`F` is an `i8`, but not every `i8` value is meaningful everywhere.

- `Q::<_, _, -128>::DELTA` is rejected at compile time.
- `Q::<_, _, F>::one()` and `Q::<_, _, F>::ONE` are rejected at compile time
  when `F < 0`, because the mathematical value `1` is not exactly representable
  for those types.

```compile_fail
use dsp_fixedpoint::Q8;
use num_traits::One;

let _ = Q8::<-1>::one();
```

## Serialization

With the default `serde` feature, `Q` serializes transparently as its raw
representation.
