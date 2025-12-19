//! Sample processing, filtering, combination of filters.

/// Processing block
///
/// Single input, single output
///
/// Process impls can be cascaded in (homogeneous) `[C; N]` arrays/`[C]` slices, and heterogeneous
/// `(C0, C1)` tuples. They can be used as configuration-major or
/// configuration-minor (through [`Minor`]) or in [`Add`]s on complementary allpasses and polyphase banks.
/// Tuples, arrays, and Pairs, and Minor can be mixed and nested ad lib.
///
/// For a given filter configuration `C` and state `S` pair the trait is usually implemented
/// through [`Split<&'a C, &mut S>`] (created ad-hoc from by borrowing configuration and state)
/// or [`Split<C, S>`] (owned configuration and state).
/// Stateless filters should implement `Process for &Self` for composability through
/// [`Split<Stateless<Self>, ()>`].
/// Configuration-less filters or filters that include their configuration should implement
/// `Process for Self` and can be used in split configurations through [`Split<(), Stateful<Self>>`].
pub trait Process<X: Copy, Y = X> {
    /// Update the state with a new input and obtain an output
    fn process(&mut self, x: X) -> Y;

    /// Process a block of inputs into a block of outputs
    ///
    /// Input and output must be of the same size.
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(y) {
            *y = self.process(*x);
        }
    }
}

/// Process a block in place.
pub trait Inplace<X: Copy>: Process<X> {
    /// Process an input block into the same data as output
    fn inplace(&mut self, xy: &mut [X]) {
        for xy in xy.iter_mut() {
            *xy = self.process(*xy);
        }
    }
}

//////////// BLANKET ////////////

impl<X: Copy, Y, T: Process<X, Y>> Process<X, Y> for &mut T {
    fn process(&mut self, x: X) -> Y {
        (*self).process(x)
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        (*self).block(x, y)
    }
}

impl<X: Copy, T: Inplace<X>> Inplace<X> for &mut T {
    fn inplace(&mut self, xy: &mut [X]) {
        (*self).inplace(xy)
    }
}
