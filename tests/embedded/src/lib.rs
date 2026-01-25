#![no_std]
#![no_main]

use core::{
    hint::black_box,
    sync::atomic::{self, Ordering},
};
use cortex_m::peripheral::DWT;
use defmt::*;
use {defmt_rtt as _, panic_probe as _};

use dsp_process::{Inplace, Process, SplitInplace, SplitProcess};

fn time<F: FnMut()>(mut func: F) -> u32 {
    cortex_m::interrupt::free(|_cs| {
        // cache warming
        func();
        func();
        let c = DWT::cycle_count();
        atomic::compiler_fence(Ordering::SeqCst);
        func();
        atomic::compiler_fence(Ordering::SeqCst);
        DWT::cycle_count().wrapping_sub(c)
    })
}

#[inline(never)]
fn noinline<F: FnOnce()>(func: F) {
    func()
}

pub fn timeit<F: FnMut()>(mut func: F) -> u32 {
    time(|| {
        noinline(&mut func);
        noinline(&mut func);
    }) - time(|| {
        noinline(&mut func);
    })
}

#[derive(Default, Debug, Clone, defmt::Format)]
pub struct CyclesResults {
    pub single: u32,
    pub chunk: u32,
    pub slice: u32,
    pub inplace_chunk: u32,
    pub inplace_slice: u32,
}

impl CyclesResults {
    pub fn header() {
        info!(
            "Units: cycles per sample, chunk size: {=usize}, slice size: {=usize}",
            CHUNK, SLICE
        );
        info!("Name                  [single, chunk, slice, chunk inplace, slice inplace]");
    }

    pub fn show(&self, name: &str) {
        const PAD: &str = "                     ";
        let adj = name.len().min(PAD.len());
        info!(
            "{=str}{=str} {=[?]}",
            name[..adj],
            PAD[adj..],
            [
                self.single as f32,
                self.chunk as f32 / CHUNK as f32,
                self.slice as f32 / SLICE as f32,
                self.inplace_chunk as f32 / CHUNK as f32,
                self.inplace_slice as f32 / SLICE as f32,
            ]
        );
    }
}

pub const SLICE: usize = 1 << 10;
pub const CHUNK: usize = 4;

pub fn bench_process<P: Process<X, Y>, X: Copy + Default, Y: Copy + Default>(
    proc: &mut P,
) -> CyclesResults {
    let proc = black_box(proc);
    let x = black_box(X::default());
    let single = timeit(|| {
        black_box(proc.process(x));
    });
    let x = black_box([X::default(); SLICE]);
    let mut y = black_box([Y::default(); SLICE]);
    let slice = timeit(|| proc.block(&x, &mut y));
    let x = black_box([X::default(); CHUNK]);
    let mut y = black_box([Y::default(); CHUNK]);
    let chunk = timeit(|| proc.block(&x, &mut y));
    CyclesResults {
        single,
        chunk,
        slice,
        ..Default::default()
    }
}

pub fn bench_inplace<P: Inplace<X>, X: Copy + Default>(proc: &mut P) -> CyclesResults {
    let mut ret = bench_process::<_, X, X>(proc);
    let proc = black_box(proc);
    let mut y = black_box([X::default(); SLICE]);
    ret.inplace_slice = timeit(|| proc.inplace(&mut y));
    let mut y = black_box([X::default(); CHUNK]);
    ret.inplace_chunk = timeit(|| proc.inplace(&mut y));
    ret
}

// TODO: move and use

pub struct FnProcess<F>(pub F);
impl<F: FnMut(X) -> Y, X: Copy, Y> Process<X, Y> for FnProcess<F> {
    fn process(&mut self, x: X) -> Y {
        (self.0)(x)
    }
}
impl<F, X: Copy> Inplace<X> for FnProcess<F> where Self: Process<X> {}

pub struct FnSplitProcess<F>(pub F);
impl<F: Fn(&mut S, X) -> Y, X: Copy, Y, S> SplitProcess<X, Y, S> for FnSplitProcess<F> {
    fn process(&self, state: &mut S, x: X) -> Y {
        (self.0)(state, x)
    }
}
impl<F, X: Copy, S> SplitInplace<X, S> for FnSplitProcess<F> where Self: SplitProcess<X, X, S> {}
