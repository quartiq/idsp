# Declarative processing primitives

## Traits

The major traits are [`Process`]/[`SplitProcess`]/[`Inplace`]/[`SplitInplace`].

## Composition

Impls can be cascaded in (homogeneous) `[C; N]` arrays/`[C]` slices, and heterogeneous
`(C0, C1)` tuples. They can be used as configuration-major or
configuration-minor (through [`Minor`]) or in [`Add`]s on complementary allpasses and polyphase banks.
Tuples, arrays, and Pairs, and Minor can be mixed and nested ad lib.
Stateless filters should implement `SplitProcess<X, Y, ()> for Self` for composability.
Configuration-less filters or filters that include their configuration should implement
`Process for Self` and can be used in split configurations through [`Split<(), Unsplit<Self>>`].
