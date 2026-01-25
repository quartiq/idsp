#![no_std]
#![no_main]

use core::{
    hint::black_box,
    sync::atomic::{self, Ordering},
};
use cortex_m::{asm, peripheral::DWT};
use defmt::*;
use num_traits::Float;
use {defmt_rtt as _, embassy_stm32 as _, panic_probe as _};

use dsp_fixedpoint::Q32;
use dsp_process::{Inplace, Process, Split};
use idsp::iir;

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

fn timeit<F: FnMut()>(mut func: F) -> u32 {
    let c1 = time(|| {
        noinline(&mut func);
    });
    let c2 = time(|| {
        noinline(&mut func);
        noinline(&mut func);
    });
    c2 - c1
}

#[derive(Default, Debug, Clone, defmt::Format)]
struct Cycles {
    block: u32,
    inplace: u32,
}

#[derive(Default, Debug, Clone, defmt::Format)]
struct CyclesResults {
    single: u32,
    chunk: Cycles,
    slice: Cycles,
}

impl CyclesResults {
    fn show(&self, name: &str) {
        const PAD: &str = "                     ";
        let adj = name.len().min(PAD.len());
        info!(
            "{=str}{=str} {=[?]}",
            name[..adj],
            PAD[adj..],
            [
                self.single as f32,
                self.chunk.block as f32 / CHUNK as f32,
                self.chunk.inplace as f32 / CHUNK as f32,
                self.slice.block as f32 / SLICE as f32,
                self.slice.inplace as f32 / SLICE as f32,
            ]
        );
    }
}

const SLICE: usize = 1 << 10;
const CHUNK: usize = 4;

fn bench_process<C, S, X>(proc: &mut Split<C, S>) -> CyclesResults
where
    Split<C, S>: Inplace<X>,
    X: Copy + Default,
{
    let proc = black_box(proc);
    let x = black_box(X::default());
    let single = timeit(|| {
        black_box(proc.process(x));
    });
    let mut xy = [black_box([x; SLICE]); 2];
    let [x, y] = xy.each_mut().map(|x| x.as_mut_slice());
    let slice = Cycles {
        block: timeit(|| proc.block(x, y)),
        inplace: timeit(|| proc.inplace(y)),
    };
    let ((x, []), (y, [])) = (x.as_chunks::<CHUNK>(), y.as_chunks_mut::<CHUNK>()) else {
        defmt::unreachable!()
    };
    // let chunked = Cycles {
    //     block: timeit(|| {
    //         for (x, y) in x.iter().zip(y.iter_mut()) {
    //             proc.block(x, y);
    //         }
    //     }),
    //     inplace: timeit(|| {
    //         for y in y.iter_mut() {
    //             proc.inplace(y);
    //         }
    //     }),
    // };
    let (x0, y0) = (&x[0], &mut y[0]);
    let chunk = Cycles {
        block: timeit(|| proc.block(x0, y0)),
        inplace: timeit(|| proc.inplace(y0)),
    };
    CyclesResults {
        single,
        chunk,
        slice,
    }
}

struct BiquadProcess<B>(B);
impl<B: biquad::Biquad<X>, X: Copy + Float> dsp_process::Process<X> for BiquadProcess<B> {
    fn process(&mut self, x: X) -> X {
        self.0.run(x)
    }
}
impl<B, X: Copy> dsp_process::Inplace<X> for BiquadProcess<B> where Self: dsp_process::Process<X> {}

fn coefficients_default<T: Float>() -> biquad::Coefficients<T> {
    biquad::Coefficients {
        a1: T::zero(),
        a2: T::zero(),
        b0: T::zero(),
        b1: T::zero(),
        b2: T::zero(),
    }
}

#[cortex_m_rt::entry]
fn main() -> ! {
    info!("Setup");
    let mut c = unwrap!(cortex_m::Peripherals::take());

    // Roughly halves cycles
    // c.SCB.enable_icache();
    // c.SCB.enable_dcache(&mut c.CPUID);

    c.DCB.enable_trace();
    c.DWT.enable_cycle_counter();

    info!("Starting");
    info!(
        "Units are cycles per sample. SLICE={=usize}, CHUNK={=usize}",
        SLICE, CHUNK
    );

    info!("Name                  [single, chunk, chunk inplace, slice, slice inplace]");

    bench_process(&mut Split::<iir::Biquad<Q32<29>>, iir::DirectForm1<i32>>::default())
        .show("idsp q32");

    bench_process(&mut Split::<
        iir::BiquadClamp<Q32<29>, i32>,
        iir::DirectForm1<i32>,
    >::default())
    .show("idsp clamp q32");

    bench_process(&mut Split::<iir::Biquad<Q32<29>>, iir::DirectForm1Wide>::default())
        .show("idsp wide q32");

    bench_process(&mut Split::<iir::Biquad<Q32<29>>, iir::DirectForm1Dither>::default())
        .show("idsp dither q32");

    bench_process(&mut Split::<
        iir::BiquadClamp<Q32<29>, i32>,
        iir::DirectForm1Wide,
    >::default())
    .show("idsp clamp wide q32");

    bench_process(&mut Split::<
        iir::BiquadClamp<Q32<29>, i32>,
        iir::DirectForm1Dither,
    >::default())
    .show("idsp clamp dither q32");

    bench_process(&mut Split::<iir::Biquad<f32>, iir::DirectForm1<f32>>::default())
        .show("idsp f32");

    bench_process(&mut Split::stateful(BiquadProcess(
        biquad::DirectForm1::new(coefficients_default::<f32>()),
    )))
    .show("biquad df1 f32");

    bench_process(&mut Split::<
        iir::Biquad<f32>,
        iir::DirectForm2Transposed<f32>,
    >::default())
    .show("idsp df2t f32");

    bench_process(&mut Split::stateful(BiquadProcess(
        biquad::DirectForm2Transposed::new(coefficients_default::<f32>()),
    )))
    .show("biquad df2t f32");

    bench_process(&mut Split::<iir::BiquadClamp<f32>, iir::DirectForm1<f32>>::default())
        .show("idsp clamp f32");

    bench_process(&mut Split::<iir::Biquad<f64>, iir::DirectForm1<f64>>::default())
        .show("idsp f64");

    bench_process(&mut Split::stateful(BiquadProcess(
        biquad::DirectForm1::new(coefficients_default::<f64>()),
    )))
    .show("biquad df1 f64");

    bench_process(&mut Split::<
        iir::Biquad<f64>,
        iir::DirectForm2Transposed<f64>,
    >::default())
    .show("idsp df2t f64");

    bench_process(&mut Split::stateful(BiquadProcess(
        biquad::DirectForm2Transposed::new(coefficients_default::<f64>()),
    )))
    .show("biquad df2t f64");

    bench_process(&mut Split::<iir::BiquadClamp<f64>, iir::DirectForm1<f64>>::default())
        .show("idsp clamp f64");

    use dsp_process::{Add, Identity, Pair, Parallel, Unsplit};
    use idsp::iir::wdf::Wdf;
    bench_process::<_, _, i32>(&mut Split::new(
        Pair::new((
            (
                (),
                Parallel((
                    [
                        defmt::unwrap!(Wdf::<_, 0xad>::quantize(&[-0.9, 0.9])),
                        defmt::unwrap!(Wdf::<_, 0xad>::quantize(&[-0.6, 0.7])),
                    ],
                    (
                        defmt::unwrap!(Wdf::<_, 0xad>::quantize(&[-0.7, 0.6])),
                        defmt::unwrap!(Wdf::<_, 0xa>::quantize(&[0.8])),
                    ),
                )),
            ),
            (),
        )),
        ((Unsplit(Identity), Default::default()), Unsplit(Add)),
    ))
    .show("idsp wdf-ca-7");

    // bench_process(&mut Split::<iir::Biquad<dsp_fixedpoint::Q16<13>>, iir::DirectForm1<i16>>::default())
    //     .show("idsp q16");
    // bench_process(&mut Split::<iir::Biquad<dsp_fixedpoint::Q64<61>>, iir::DirectForm1<i64>>::default())
    //     .show("idsp q64");

    info!("Done");

    asm::bkpt();
    loop {
        asm::nop();
    }
}
