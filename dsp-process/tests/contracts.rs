//! State and output equivalence across processing shapes.

use dsp_process::{
    DecimatorError, FnSplitProcess, Inplace, LaneMajor, Process, Split, SplitInplace, SplitProcess,
    TryDecimator, View, ViewInplace, ViewMut, ViewProcess,
};

#[derive(Clone, Copy)]
struct Accumulate(i32);

impl SplitProcess<i32, i32, i32> for Accumulate {
    fn process(&self, state: &mut i32, x: i32) -> i32 {
        *state += self.0 * x;
        *state
    }
}

impl SplitInplace<i32, i32> for Accumulate {}

#[test]
fn serial_execution_forms() {
    let initial = Split::new((Accumulate(2), Accumulate(-1)), (3, -4));
    for len in 0..=17 {
        let input: Vec<i32> = (0..len).map(|i| i % 5 - 2).collect();
        let mut scalar = initial;
        let expected: Vec<_> = input.iter().map(|&x| scalar.process(x)).collect();
        let mut block = initial;
        let mut minor = initial.minor::<i32>();
        let mut major = initial.major::<[i32; 4]>();
        for stage in [&mut block as &mut dyn Inplace<i32>, &mut minor, &mut major] {
            let mut output = vec![0; input.len()];
            stage.block(&[], &mut []);
            for (x, y) in input.chunks(7).zip(output.chunks_mut(7)) {
                stage.block(x, y);
            }
            assert_eq!(output, expected);
        }
        assert_eq!(block.state, scalar.state);
        assert_eq!(minor.state, scalar.state);
        assert_eq!(major.state, scalar.state);
        let mut block = initial;
        let mut minor = initial.minor::<i32>();
        let mut major = initial.major::<[i32; 4]>();
        for stage in [&mut block as &mut dyn Inplace<i32>, &mut minor, &mut major] {
            let mut output = input.clone();
            stage.inplace(&mut output);
            assert_eq!(output, expected);
        }
        assert_eq!(block.state, scalar.state);
        assert_eq!(minor.state, scalar.state);
        assert_eq!(major.state, scalar.state);
    }
}

#[test]
fn lane_layouts() {
    for len in 0..=9 {
        let input: Vec<[i32; 2]> = (0..len).map(|i| [i - 3, 2 - i]).collect();
        let initial = Split::new(Accumulate(2), 7).lanes::<2>();
        let mut scalar = initial;
        let expected: Vec<_> = input.iter().map(|&x| scalar.process(x)).collect();
        let mut frame = initial.per_frame();
        let mut output = vec![[0; 2]; input.len()];
        frame.process_frames(View::from_frames(&input), ViewMut::from_frames(&mut output));
        assert_eq!(output, expected);
        assert_eq!(frame.state, scalar.state);
        let flat: Vec<_> = (0..2)
            .flat_map(|lane| input.iter().map(move |x| x[lane]))
            .collect();
        let want: Vec<_> = (0..2)
            .flat_map(|lane| expected.iter().map(move |x| x[lane]))
            .collect();
        let mut lane = initial;
        let mut output = vec![0; flat.len()];
        lane.process_view(
            View::<_, LaneMajor, 2>::from_flat(&flat, input.len()),
            ViewMut::<_, LaneMajor, 2>::from_flat(&mut output, input.len()),
        );
        assert_eq!(output, want);
        assert_eq!(lane.state, scalar.state);
        let mut lane = initial;
        let mut output = flat;
        lane.inplace_view(ViewMut::<_, LaneMajor, 2>::from_flat(
            &mut output,
            input.len(),
        ));
        assert_eq!(output, want);
        assert_eq!(lane.state, scalar.state);
    }
}

#[test]
fn decimator_advances_whole_chunk_on_error() {
    let stage = TryDecimator(FnSplitProcess(|state: &mut usize, tick: bool| {
        *state += 1;
        tick.then_some(*state)
    }));
    let mut state = 0;
    assert_eq!(
        stage.process(&mut state, [true, true, false, false]),
        Err(DecimatorError::ExtraTick)
    );
    assert_eq!(state, 4);
    assert_eq!(
        stage.process(&mut state, [false; 4]),
        Err(DecimatorError::NoTick)
    );
    assert_eq!(state, 8);
    assert_eq!(
        stage.process(&mut state, [false, true, false, false]),
        Ok(10)
    );
    assert_eq!(state, 12);
    assert_eq!(
        stage.process(&mut state, [true; 4]),
        Err(DecimatorError::ExtraTick)
    );
    assert_eq!(state, 16);
}

/// Check every split, including empty blocks, after each possible delay phase.
fn check_blocks<P, Y>(initial: P)
where
    P: Process<i32, Y> + Clone,
    Y: Copy + Default + PartialEq + core::fmt::Debug,
{
    let input: Vec<_> = (1..=17).collect();
    for phase in 0..4 {
        for len in 0..=12 {
            for split in 0..=len {
                let mut scalar = initial.clone();
                for &x in &input[..phase] {
                    scalar.process(x);
                }
                let mut block = scalar.clone();
                let input = &input[phase..];
                let expected: Vec<_> = input.iter().map(|&x| scalar.process(x)).collect();
                let mut output = vec![Y::default(); input.len()];
                block.block(&input[..split], &mut output[..split]);
                block.block(&[], &mut []);
                block.block(&input[split..len], &mut output[split..len]);
                for (&x, y) in input[len..].iter().zip(&mut output[len..]) {
                    *y = block.process(x);
                }
                assert_eq!(output, expected, "phase={phase}, len={len}, split={split}");
            }
        }
    }
}

#[test]
fn delay_and_comb_blocks() {
    use dsp_process::{Buffer, Comb, Nyquist};
    check_blocks::<_, i32>(Buffer::<[i32; 4]>::default());
    check_blocks::<_, Option<[i32; 4]>>(Buffer::<[i32; 4]>::default());
    check_blocks(Comb([4, 3, 2, 1]));
    check_blocks(Nyquist([4, 3, 2, 1]));
}

#[test]
fn delay_inplace_partitions() {
    use dsp_process::Buffer;
    let input: Vec<i32> = (1..=17).collect();
    for phase in 0..4 {
        for split in 0..=12 {
            let mut scalar = Buffer::<[i32; 4]>::default();
            for &x in &input[..phase] {
                let _: i32 = scalar.process(x);
            }
            let mut inplace = scalar;
            let expected: Vec<i32> = input[phase..].iter().map(|&x| scalar.process(x)).collect();
            let mut output = input[phase..].to_vec();
            inplace.inplace(&mut output[..split]);
            inplace.inplace(&mut []);
            inplace.inplace(&mut output[split..12]);
            for x in &mut output[12..] {
                *x = inplace.process(*x);
            }
            assert_eq!(output, expected);
        }
    }
}
