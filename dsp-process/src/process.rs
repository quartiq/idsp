//! Core traits for synchronous sample and block processing.

/// Processing block
///
/// Single-input processing with state held in `self`.
///
/// This is the simplest trait in the crate: one new sample goes in, one output
/// value comes out. Override [`block()`](Self::block) when a specialized loop can
/// reuse scratch storage, reduce bounds checks, or better match the desired data
/// layout.
///
/// [`SplitProcess`] is the corresponding trait when immutable configuration and
/// mutable runtime state should be separated.
///
/// # Examples
///
/// ```rust
/// use dsp_process::Process;
///
/// #[derive(Default)]
/// struct Acc(i32);
///
/// impl Process<i32> for Acc {
///     fn process(&mut self, x: i32) -> i32 {
///         self.0 += x;
///         self.0
///     }
/// }
///
/// let mut acc = Acc::default();
/// assert_eq!(acc.process(2), 2);
/// assert_eq!(acc.process(3), 5);
/// ```
pub trait Process<X: Copy, Y = X> {
    /// Update the state with a new input and obtain an output
    fn process(&mut self, x: X) -> Y;

    /// Process a block of inputs into a block of outputs
    ///
    /// Input and output must be of the same size.
    ///
    /// For hot-path use this is treated as a caller precondition; the default
    /// implementation only checks it in debug builds.
    fn block(&mut self, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(y) {
            *y = self.process(*x);
        }
    }
}

/// Inplace processing
///
/// This is a convenience trait for processors where input and output element
/// types are identical and the computation can be expressed as overwriting a
/// mutable slice.
///
/// See also [`SplitInplace`] for the split configuration/state form.
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
///   (e.g. IQ data, multiple lanes) guaranteeing consistency,
///   reducing memory usage, and improving caching.
///
/// This is the central abstraction used throughout `dsp-process`. A typical DSP
/// filter coefficient set becomes `Self`, while delay lines, accumulators, and
/// history buffers become the separate `state` argument.
///
/// Use this when one configuration should drive many runtime states, or when it
/// is beneficial to keep mutable state small and move immutable data out of hot
/// loops.
///
/// [`Process`] is often easier when state and configuration naturally live
/// together, while [`crate::Split`] turns a `SplitProcess` back into a stateful
/// [`Process`] value.
///
/// # Examples
///
/// ```rust
/// use dsp_process::SplitProcess;
///
/// #[derive(Copy, Clone)]
/// struct Gain(i32);
///
/// impl SplitProcess<i32> for Gain {
///     fn process(&self, _: &mut (), x: i32) -> i32 {
///         self.0 * x
///     }
/// }
///
/// let mut state = ();
/// assert_eq!(Gain(4).process(&mut state, 3), 12);
/// ```
pub trait SplitProcess<X: Copy, Y = X, S: ?Sized = ()> {
    /// Process an input into an output
    ///
    /// See also [`Process::process`]
    fn process(&self, state: &mut S, x: X) -> Y;

    /// Process a block of inputs
    ///
    /// See also [`Process::block`]
    ///
    /// Length matching is a caller precondition in release builds.
    fn block(&self, state: &mut S, x: &[X], y: &mut [Y]) {
        debug_assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(y) {
            *y = self.process(state, *x);
        }
    }
}

/// Inplace processing with a split state
///
/// This is the split-state companion to [`Inplace`]. Implement it when a
/// `SplitProcess<X, X, S>` can update a buffer in place more efficiently than
/// routing through a separate output slice.
pub trait SplitInplace<X: Copy, S: ?Sized = ()>: SplitProcess<X, X, S> {
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

impl<X: Copy, Y, S: ?Sized, T: SplitProcess<X, Y, S>> SplitProcess<X, Y, S> for &T {
    fn process(&self, state: &mut S, x: X) -> Y {
        T::process(self, state, x)
    }

    fn block(&self, state: &mut S, x: &[X], y: &mut [Y]) {
        T::block(self, state, x, y)
    }
}

impl<X: Copy, S: ?Sized, T: SplitInplace<X, S>> SplitInplace<X, S> for &T {
    fn inplace(&self, state: &mut S, xy: &mut [X]) {
        T::inplace(self, state, xy)
    }
}

impl<X: Copy, Y, S: ?Sized, T: SplitProcess<X, Y, S>> SplitProcess<X, Y, S> for &mut T {
    fn process(&self, state: &mut S, x: X) -> Y {
        T::process(self, state, x)
    }

    fn block(&self, state: &mut S, x: &[X], y: &mut [Y]) {
        T::block(self, state, x, y)
    }
}

impl<X: Copy, S: ?Sized, T: SplitInplace<X, S>> SplitInplace<X, S> for &mut T {
    fn inplace(&self, state: &mut S, xy: &mut [X]) {
        T::inplace(self, state, xy)
    }
}

/// Wrap a `FnMut` into a `Process`/`Inplace`
///
/// This is useful for quick experiments, benchmarks, or adapters at the edge of
/// a pipeline. For reusable DSP stages, prefer a named type once the closure
/// starts carrying real semantics.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{FnProcess, Process};
///
/// let mut square = FnProcess(|x: i32| x * x);
/// assert_eq!(square.process(7), 49);
/// ```
pub struct FnProcess<F>(pub F);

impl<F: FnMut(X) -> Y, X: Copy, Y> Process<X, Y> for FnProcess<F> {
    fn process(&mut self, x: X) -> Y {
        (self.0)(x)
    }
}

impl<F, X: Copy> Inplace<X> for FnProcess<F> where Self: Process<X> {}

/// Wrap a `Fn` into a `SplitProcess`/`SplitInplace`
///
/// The closure receives both the mutable split state and the new input sample.
/// This is a compact way to prototype split-state processors before promoting
/// them to named types.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{FnSplitProcess, SplitProcess};
///
/// let proc = FnSplitProcess(|state: &mut i32, x: i32| {
///     *state += x;
///     *state
/// });
///
/// let mut state = 0;
/// assert_eq!(proc.process(&mut state, 2), 2);
/// assert_eq!(proc.process(&mut state, 3), 5);
/// ```
pub struct FnSplitProcess<F>(pub F);

impl<F: Fn(&mut S, X) -> Y, X: Copy, Y, S> SplitProcess<X, Y, S> for FnSplitProcess<F> {
    fn process(&self, state: &mut S, x: X) -> Y {
        (self.0)(state, x)
    }
}

impl<F, X: Copy, S> SplitInplace<X, S> for FnSplitProcess<F> where Self: SplitProcess<X, X, S> {}
