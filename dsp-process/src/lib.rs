#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]
use core::marker::PhantomData;

mod process;
pub use process::*;
mod basic;
mod containers;
pub use basic::*;
mod split;
pub use split::*;

/// Binary const assertions
pub(crate) struct Assert<const A: usize, const B: usize>;
impl<const A: usize, const B: usize> Assert<A, B> {
    /// Assert A>B
    pub const GREATER: () = assert!(A > B);
}

/// Processor-minor, data-major
///
/// The various Process tooling implementations for `Minor`
/// place the data loop as the outer-most loop (processor-minor, data-major).
/// This is optimal for processors with small or no state and configuration.
///
/// Chain of large processors are implemented through tuples and slices/arrays.
/// Those optimize well if the sizes obey configuration ~ state > data.
/// If they do not, use `Minor`.
///
/// Note that the major implementations only override the behavior
/// for `block()` and `inplace()`. `process()` is unaffected.
#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub struct Minor<C: ?Sized, U> {
    /// An intermediate data type
    _intermediate: PhantomData<U>,
    /// The inner configurations
    pub inner: C,
}

impl<C, U> Minor<C, U> {
    /// Create a new chain
    pub fn new(inner: C) -> Self {
        Self {
            inner,
            _intermediate: PhantomData,
        }
    }
}

impl<C, U, const N: usize> Minor<[C; N], U> {
    /// Borrowed Self
    pub fn as_ref(&self) -> Minor<&[C], U> {
        Minor::new(self.inner.as_ref())
    }

    /// Mutably borrowed Self
    pub fn as_mut(&mut self) -> Minor<&mut [C], U> {
        Minor::new(self.inner.as_mut())
    }
}

/// Fan out parallel input to parallel processors
#[derive(Clone, Copy, Debug, Default)]
pub struct Parallel<P>(pub P);

/// Data block transposition wrapper
///
/// Like [`Parallel`] but reinterpreting data as transpes `[[X; N]] <-> [[X]; N]`
/// such that `block()` and `inplace()` are lowered.
#[derive(Clone, Copy, Debug, Default)]
pub struct Transpose<C>(pub C);

/// Multiple channels to be processed with the same configuration
#[derive(Clone, Copy, Debug, Default)]
pub struct Channels<C>(pub C);

/// Parallel filter pair
///
/// This can be viewed as digital lattice filter or butterfly filter or complementary allpass pair
/// or polyphase interpolator.
/// Candidates for the branches are allpasses like Wdf or Ldi, polyphase banks for resampling or Hilbert filters.
///
/// Potentially required scaling with 0.5 gain is to be performed ahead of the filter or within each branch.
///
/// This uses the default sample-major implementation
/// and may lead to suboptimal cashing and register thrashing for large branches.
/// To avoid this, use `block()` and `inplace()` on a scratch buffer (input or output).
///
/// The corresponding state for this is `(((), (S0, S1)), ())`.
pub type Pair<C0, C1, X, I = Unsplit<Identity>, J = Unsplit<Add>> =
    Minor<((I, Parallel<(C0, C1)>), J), [X; 2]>;

#[cfg(test)]
mod test {
    use super::*;
    use dsp_fixedpoint::Q32;

    #[allow(unused)]
    const fn assert_process<T: Process<X, Y>, X: Copy, Y>() {}

    #[test]
    fn misc() {
        let y: i32 = (&Identity).process(3);
        assert_eq!(y, 3);
        assert_eq!(Split::stateless(Neg).as_mut().process(9), -9);
        //assert_eq!(Split::stateless(Neg).process(9), -9);
        assert_eq!((&Gain(Q32::<3>::new(32))).process(9), 9 * 4);
        assert_eq!((&Offset(7)).process(9), 7 + 9);
        assert_eq!(Minor::new((&Offset(7), &Offset(1))).process(9), 7 + 1 + 9);
        let mut xy = [3, 0, 0];
        let mut dly = Buffer::<_, 2>::default();
        dly.inplace(&mut xy);
        assert_eq!(xy, [0, 0, 3]);
        let y: i32 = Split::stateful(dly).as_mut().process(4);
        assert_eq!(y, 0);
        let mut f = Split::new(
            Pair::<_, _, i32>::new((
                (
                    Default::default(),
                    Parallel((Unsplit(Offset(3)), Unsplit(Gain(Q32::<1>::new(4))))),
                ),
                Default::default(),
            )),
            Default::default(),
        );
        let y: i32 = f.as_mut().process(5);
        assert_eq!(y, (5 + 3) + ((5 * 4) >> 1));
        let y: [i32; 5] = f.channels().as_mut().process([5; _]);
        assert_eq!(y, [(5 + 3) + ((5 * 4) >> 1); 5]);
    }
}
