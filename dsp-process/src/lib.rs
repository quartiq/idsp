#![cfg_attr(not(feature = "std"), no_std)]

mod process;
pub use process::*;
mod containers;
pub use containers::*;
mod split;
pub use split::*;
mod basic;
pub use basic::*;

/// Binary const assertions
pub(crate) struct Assert<const A: usize, const B: usize>;
impl<const A: usize, const B: usize> Assert<A, B> {
    /// Assert A>B
    pub const GREATER: () = assert!(A > B);
}

#[cfg(test)]
mod test {
    use crate::*;
    use dsp_fixedpoint::Q32;

    #[allow(unused)]
    const fn assert_process<T: Process<X, Y>, X: Copy, Y>() {}

    #[test]
    fn misc() {
        let y: i32 = (&Identity).process(3);
        assert_eq!(y, 3);
        assert_eq!(Split::stateless(Neg).as_mut().process(9), -9);
        assert_eq!((&Gain(Q32::<3>::new(32))).process(9), 9 * 4);
        assert_eq!((&Offset(7)).process(9), 7 + 9);
        assert_eq!(Minor::new((&Offset(7), &Offset(1))).process(9), 7 + 1 + 9);
        let mut xy = [3, 0, 0];
        let mut dly = Buffer::<_, 2>::default();
        dly.inplace(&mut xy);
        assert_eq!(xy, [0, 0, 3]);
        let y: i32 = Split::stateful(dly).as_mut().process(4);
        assert_eq!(y, 0);
        let f = Pair::<_, _, i32>::new((
            (
                Default::default(),
                Parallel((Stateless(Offset(3)), Stateless(Gain(Q32::<1>::new(4))))),
            ),
            Default::default(),
        ));
        let y: i32 = Split::new(&f, &mut Default::default()).process(5);
        assert_eq!(y, (5 + 3) + ((5 * 4) >> 1));
        let y: [i32; 5] = Split::new(Channels(f), Default::default())
            .as_mut()
            .process([5; _]);
        assert_eq!(y, [(5 + 3) + ((5 * 4) >> 1); 5]);
    }
}
