#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

mod process;
pub use process::*;
mod basic;
pub use basic::*;
mod split;
pub use split::*;
mod adapters;
pub use adapters::*;
mod compose;
pub use compose::*;

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
pub type Pair<C0, C1, X, I = Unsplit<&'static Identity>, J = Unsplit<&'static Add>> =
    Minor<((I, Parallel<(C0, C1)>), J), [X; 2]>;

#[cfg(test)]
mod test {
    use super::*;
    use dsp_fixedpoint::Q32;

    #[test]
    fn misc() {
        let y: i32 = (&Identity).process(3);
        assert_eq!(y, 3);
        assert_eq!(Split::stateless(&Neg).as_mut().process(9), -9);
        //assert_eq!(Split::stateless(Neg).process(9), -9);
        assert_eq!((&Gain(Q32::<3>::new(32))).process(9), 9 * 4);
        assert_eq!((&Offset(7)).process(9), 7 + 9);
        let mut p = (Split::stateless(&Offset(7)) * Split::stateless(&Offset(1))).minor();
        p.as_mut().assert_process::<i8, _>();
        assert_eq!(p.as_mut().process(9), 7 + 1 + 9);
        let mut xy = [3, 0, 0];
        let mut dly = Buffer::<[_; 2]>::default();
        dly.inplace(&mut xy);
        assert_eq!(xy, [0, 0, 3]);
        let y: i32 = Split::stateful(dly).as_mut().process(4);
        assert_eq!(y, 0);
        let g = Gain(Q32::<1>::new(4));
        let mut f = Split::new(
            Pair::<_, _, _>::new((
                (
                    Unsplit(&Identity),
                    Parallel((Unsplit(&Offset(3)), Unsplit(&g))),
                ),
                Unsplit(&Add),
            )),
            Default::default(),
        );
        let y: i32 = f.as_mut().process(5);
        assert_eq!(y, (5 + 3) + ((5 * 4) >> 1));
        let y: [i32; 5] = f.channels().as_mut().process([5; _]);
        assert_eq!(y, [(5 + 3) + ((5 * 4) >> 1); 5]);
    }
}
