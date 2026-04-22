#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

mod process;
pub use process::*;
mod basic;
pub use basic::*;
mod adapters;
pub use adapters::*;
mod split;
pub use split::*;
mod compose;
pub use compose::*;

/// Parallel filter pair
///
/// This can be viewed as digital lattice filter or butterfly filter or complementary allpass pair
/// or polyphase interpolator.
/// Candidates for the branches are allpasses like Wdf or Ldi, polyphase banks for resampling or Hilbert filters.
///
/// Potentially required scaling with 0.5 gain is to be performed ahead of the filter, within each branch, or (with headroom) afterwards.
///
/// This uses the default configuration-minor/sample-major implementation
/// and may lead to suboptimal cashing and register thrashing for large branches.
/// To avoid this, use `block()` and `inplace()` on a scratch buffer ([`Major`] input or output).
///
/// The corresponding state for this is `((Unsplit<Identity>, (S0, S1)), Unsplit<Add>)`.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Add, Gain, Identity, Offset, Pair, Parallel, Process, Split, Unsplit};
///
/// let mut pair = Split::new(
///     Pair::<_, _, i32>::new((((), Parallel((Offset(3), Gain(4)))), ())),
///     ((Unsplit(Identity), Default::default()), Unsplit(Add)),
/// );
/// let y: i32 = pair.process(5);
/// assert_eq!(y, 28);
/// ```
pub type Pair<C0, C1, X, I = (), J = ()> = Minor<((I, Parallel<(C0, C1)>), J), [X; 2]>;

#[cfg(test)]
mod test {
    use super::*;
    use dsp_fixedpoint::Q32;

    #[test]
    fn basic() {
        assert_eq!(3, Identity.process(3));
        assert_eq!([7, 1], Butterfly.process([4, 3]));
        assert_eq!(Split::stateless(Gain(Q32::<3>::new(32))).process(9), 9 * 4);
        assert_eq!(Split::stateless(Offset(7)).process(9), 7 + 9);
    }

    #[test]
    fn stateless() {
        assert_eq!(Neg.process(9), -9);
        assert_eq!(Split::stateful(Neg).process(9), -9);

        let mut p = (Split::stateless(Offset(7)) * Split::stateless(Offset(1))).minor();
        p.assert_process::<i8, _>();
        assert_eq!(p.process(9), 7 + 1 + 9);
    }

    #[test]
    fn stateful() {
        let mut xy = [3, 0, 0];
        let mut dly = Buffer::<[_; 2]>::default();
        dly.inplace(&mut xy);
        assert_eq!(xy, [0, 0, 3]);
        let y: i32 = Split::stateful(dly).process(4);
        assert_eq!(y, 0);
    }

    #[test]
    fn pair() {
        let g = Gain(Q32::<1>::new(4));
        let mut f = Split::new(
            Pair::<_, _, _>::new((((), Parallel((Offset(3), g))), ())),
            ((Unsplit(Identity), Default::default()), Unsplit(Add)),
        );
        let y: i32 = f.process(5);
        assert_eq!(y, (5 + 3) + ((5 * 4) >> 1));

        let y: [i32; 5] = f.channels().process([5; _]);
        assert_eq!(y, [(5 + 3) + ((5 * 4) >> 1); 5]);
    }

    #[test]
    fn chunk_in_out() {
        let mut p = Split::stateless(ChunkInOut::<_, 2, 1>(FnSplitProcess(
            |_: &mut (), [x0, x1]: [i32; 2]| [x0 + x1],
        )));
        let y: [i32; 2] = p.process([1, 2, 3, 4]);
        assert_eq!(y, [3, 7]);
    }

    #[test]
    fn chunk_out_pod() {
        let mut p = Split::stateless(ChunkOutPod::<_, 2>(FnSplitProcess(|_: &mut (), x: i32| {
            [x, -x]
        })));
        let y: [i32; 4] = p.process([2, 3]);
        assert_eq!(y, [2, -2, 3, -3]);
    }

    #[test]
    fn buffer_blocks() {
        let mut dly = Buffer::<[_; 2]>::default();
        let mut y = [0; 5];
        dly.block(&[1, 2, 3, 4, 5], &mut y);
        assert_eq!(y, [0, 0, 1, 2, 3]);
        assert_eq!(dly.process([6, 7, 8]), [4, 5, 6]);

        let mut chunk = Buffer::<[_; 2]>::default();
        let mut y = [None; 5];
        <Buffer<[i32; 2]> as Process<i32, Option<[i32; 2]>>>::block(
            &mut chunk,
            &[1, 2, 3, 4, 5],
            &mut y,
        );
        assert_eq!(y, [None, Some([1, 2]), None, Some([3, 4]), None]);

        let mut stream = Buffer::<[_; 2]>::default();
        let mut y = [0; 5];
        <Buffer<[i32; 2]> as Process<Option<[i32; 2]>, i32>>::block(
            &mut stream,
            &[Some([1, 2]), None, Some([3, 4]), None, Some([5, 6])],
            &mut y,
        );
        assert_eq!(y, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn diffsum_block() {
        let mut nyq = Nyquist([10, 20]);
        let mut y = [0; 4];
        nyq.block(&[1, 2, 3, 4], &mut y);
        assert_eq!(y, [21, 12, 4, 6]);
        let mut xy = [5, 6, 7];
        nyq.inplace(&mut xy);
        assert_eq!(xy, [8, 10, 12]);

        let mut comb = Comb([10, 20]);
        let mut y = [0; 4];
        comb.block(&[1, 2, 3, 4], &mut y);
        assert_eq!(y, [-19, -8, 2, 2]);
        let mut xy = [5, 6, 7];
        comb.inplace(&mut xy);
        assert_eq!(xy, [2, 2, 2]);
    }
}
