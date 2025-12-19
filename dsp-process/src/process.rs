//! Sample processing, filtering, combination of filters.

/// Processing block
///
/// Single input, single output
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

/// Inplace processing
pub trait Inplace<X: Copy>: Process<X> {
    /// Process an input block into the same data as output
    fn inplace(&mut self, xy: &mut [X]) {
        for xy in xy.iter_mut() {
            *xy = self.process(*xy);
        }
    }
}

/// Processing with split state
///
/// Splitting configuration (the part of the filter that is unaffected
/// by processing inputs, e.g. "coefficients"), from state (the part
/// that is modified by processing) allows:
///
/// * Separating mutable from immutable state guarantees consistency
///   (configuration can not change state and processing can
///   not change configuration)
/// * Reduces memory traffic when swapping configuration
/// * Allows the same filter to be applied to multiple states
///   (e.g. IQ data, multiple channels) guaranteeing consistency,
///   reducing memory usage, and improving caching.
pub trait SplitProcess<X: Copy, Y = X, S = ()>
where
    S: ?Sized,
{
    /// Process an input into an output
    ///
    /// See also [`Process::process`]
    fn process(&self, state: &mut S, x: X) -> Y;

    /// Process a block of inputs
    ///
    /// See also [`Process::block`]
    fn block(&self, state: &mut S, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(y) {
            *y = self.process(state, *x);
        }
    }
}

/// Inplace processing with a split state
pub trait SplitInplace<X: Copy, S = ()>: SplitProcess<X, X, S>
where
    S: ?Sized,
{
    /// See also [`Inplace::inplace`]
    fn inplace(&self, state: &mut S, xy: &mut [X]) {
        for xy in xy.iter_mut() {
            *xy = self.process(state, *xy);
        }
    }
}

//////////// BLANKET ////////////

impl<X: Copy, Y, T: Process<X, Y>> Process<X, Y> for &mut T {
    fn process(&mut self, x: X) -> Y {
        T::process(self, x)
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        T::block(self, x, y)
    }
}

impl<X: Copy, T: Inplace<X>> Inplace<X> for &mut T {
    fn inplace(&mut self, xy: &mut [X]) {
        T::inplace(self, xy)
    }
}

impl<X: Copy, Y, S, T: SplitProcess<X, Y, S>> SplitProcess<X, Y, S> for &T {
    fn process(&self, state: &mut S, x: X) -> Y {
        T::process(self, state, x)
    }

    fn block(&self, state: &mut S, x: &[X], y: &mut [Y]) {
        T::block(self, state, x, y)
    }
}

impl<X: Copy, S, T: SplitInplace<X, S>> SplitInplace<X, S> for &T {
    fn inplace(&self, state: &mut S, xy: &mut [X]) {
        T::inplace(self, state, xy)
    }
}

impl<X: Copy, Y, S, T: SplitProcess<X, Y, S>> SplitProcess<X, Y, S> for &mut T {
    fn process(&self, state: &mut S, x: X) -> Y {
        T::process(self, state, x)
    }

    fn block(&self, state: &mut S, x: &[X], y: &mut [Y]) {
        T::block(self, state, x, y)
    }
}

impl<X: Copy, S, T: SplitInplace<X, S>> SplitInplace<X, S> for &mut T {
    fn inplace(&self, state: &mut S, xy: &mut [X]) {
        T::inplace(self, state, xy)
    }
}
