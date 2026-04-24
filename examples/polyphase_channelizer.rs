//! Four-channel static polyphase analysis bank.
//!
//! Showcases:
//! - `dsp_process::Split::per_frame()` as the clean fit for a maximally decimated
//!   DFT analysis bank: one 4-sample frame in, one 4-channel frame out
//! - tuple composition plus `Minor` to expose the graph as `polyphase bank -> DFT`
//! - a framewise polyphase FIR kernel with one shared circular head across all phases
//! - real prototype taps applied to I/Q lanes separately, because the FIR is linear
//! - `idsp::Complex::<f32>::from_angle_rad()` only in the fixture, for readable tones
//!
//! This intentionally does not use `Lanes`/`ByLane` in the hot kernel. The good
//! implementation updates all phases together once per frame, so the natural
//! state is one shared delay bank rather than four independent lane processors.

use bytemuck;
use dsp_process::{Split, SplitProcess, View, ViewMut};
use idsp::Complex;
use std::f32::consts::TAU;

const M: usize = 4; // DFT4
const TAPS: usize = 8;

type Iq = [f32; 2];
// One frame is the semantic sample of the hot graph: the bank consumes one
// decimated 4-sample input frame and emits one 4-channel output frame.
type Frame = [Iq; M];

// --- DSP graph ---

fn sinc(x: f32) -> f32 {
    if x == 0.0 { 1.0 } else { x.sin() / x }
}

fn prototype() -> [f32; M * TAPS] {
    let fc = 0.5 / M as f32 * 0.9;
    let mid = (M * TAPS - 1) as f32 * 0.5;
    let mut h = core::array::from_fn(|i| {
        let n = i as f32 - mid;
        let w = 0.54 - 0.46 * (TAU * i as f32 / (M * TAPS - 1) as f32).cos();
        2.0 * fc * sinc(TAU * fc * n) * w
    });
    let sum = h.iter().sum::<f32>();
    h.iter_mut().for_each(|h| *h /= sum);
    h
}

#[derive(Clone, Copy, Debug, Default)]
struct BankState {
    hist: [Frame; TAPS],
    head: usize,
}

#[derive(Clone, Copy, Debug)]
struct PolyphaseBank {
    coeff: [[f32; M]; TAPS],
}

impl SplitProcess<Frame, Frame, BankState> for PolyphaseBank {
    fn process(&self, state: &mut BankState, x: Frame) -> Frame {
        state.head = (state.head + TAPS - 1) % TAPS;
        state.hist[state.head] = x;

        let mut y = Frame::default();
        for tap in 0..TAPS {
            for ((y, c), h) in y
                .iter_mut()
                .zip(&self.coeff[tap])
                .zip(&state.hist[(state.head + tap) % TAPS])
            {
                y[0] += h[0] * c;
                y[1] += h[1] * c;
            }
        }
        y
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Dft4;

impl SplitProcess<Frame, Frame> for Dft4 {
    fn process(&self, _: &mut (), x: Frame) -> Frame {
        [
            [
                x[0][0] + x[1][0] + x[2][0] + x[3][0],
                x[0][1] + x[1][1] + x[2][1] + x[3][1],
            ],
            [
                x[0][0] + x[1][1] - x[2][0] - x[3][1],
                x[0][1] - x[1][0] - x[2][1] + x[3][0],
            ],
            [
                x[0][0] - x[1][0] + x[2][0] - x[3][0],
                x[0][1] - x[1][1] + x[2][1] - x[3][1],
            ],
            [
                x[0][0] - x[1][1] - x[2][0] + x[3][1],
                x[0][1] + x[1][0] - x[2][1] - x[3][0],
            ],
        ]
    }
}

fn run_graph(x: &[Iq]) -> Vec<Frame> {
    let coeff = bytemuck::cast(prototype());
    // `per_frame()` lifts the framewise graph into typed `View`/`ViewMut`
    // processing without rewriting the bank itself as a view processor.
    let mut bank = (Split::new(PolyphaseBank { coeff }, BankState::default())
        * Split::stateless(Dft4))
    .minor()
    .per_frame();
    // The backing storage stays frame-major; typed views make that boundary
    // layout explicit instead of smuggling it through raw slices.
    let (frames, []) = x.as_chunks() else {
        unreachable!()
    };
    let mut y = vec![Frame::default(); frames.len()];
    bank.process_frames(View::from_frames(frames), ViewMut::from_frames(&mut y));
    y
}

// --- Fixture ---

fn tone(freq: f32, n: usize) -> Vec<Iq> {
    (0..n)
        .map(|i| {
            let z = Complex::<f32>::from_angle_rad(TAU * freq * i as f32);
            [z.re(), z.im()]
        })
        .collect()
}

// --- Metrics / checks ---

fn channel_powers(freq: f32) -> [f32; M] {
    let y = run_graph(&tone(freq, 4096));
    let mut p = [0.0; M];
    let mut used = 0.0;
    for frame in &y[128..] {
        for (p, y) in p.iter_mut().zip(*frame) {
            *p += y[0] * y[0] + y[1] * y[1];
        }
        used += 1.0;
    }
    p.map(|p| p / used)
}

fn main() {
    for freq in [0.0, 0.25, 0.5, 0.75] {
        let p = channel_powers(freq);
        println!(
            "freq={freq:.2} powers=[{:.3}, {:.3}, {:.3}, {:.3}]",
            p[0], p[1], p[2], p[3]
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn argmax(x: &[f32]) -> usize {
        x.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    #[test]
    fn routes_center_tones_to_expected_bins() {
        for (freq, want) in [(0.0, 0), (0.25, 1), (0.5, 2), (0.75, 3)] {
            let p = channel_powers(freq);
            assert_eq!(argmax(&p), want);
            let strongest_other = p
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != want)
                .map(|(_, p)| *p)
                .fold(0.0, f32::max);
            assert!(p[want] > 10.0 * strongest_other);
        }
    }
}
