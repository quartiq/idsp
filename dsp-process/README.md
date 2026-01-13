# Declarative processing primitives

## Traits

The major traits are [`Process`]/[`SplitProcess`]/[`Inplace`]/[`SplitInplace`].

## Composition

Impls can be cascaded in (homogeneous) `[C; N]` arrays/`[C]` slices, and heterogeneous
`(C0, C1)` tuples. They can be used as configuration-major or
configuration-minor (through [`Minor`]) or in [`Add`]s on complementary allpasses and polyphase banks.
Tuples, arrays, and Pairs, and Minor can be mixed and nested ad lib.
For a given filter configuration `C` and state `S` pair the trait is usually implemented
through [`Split<&'a C, &mut S>`] (created ad-hoc from by borrowing configuration and state)
or [`Split<C, S>`] (owned configuration and state).
Stateless filters should implement `Process for &Self` for composability through
[`Split<Unsplit<&Self>, ()>`].
Configuration-less filters or filters that include their configuration should implement
`Process for Self` and can be used in split configurations through [`Split<(), Unsplit<Self>>`].
