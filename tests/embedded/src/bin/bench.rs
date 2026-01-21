#![no_std]
#![no_main]

use core::{
    hint::black_box,
    ptr,
    sync::atomic::{self, Ordering},
};
use cortex_m::{asm, peripheral::DWT};
use defmt::*;
// use embassy_stm32::{bind_interrupts, peripherals};
use embassy_stm32::Config;
use embassy_stm32::rcc::*;
use num_traits::Float;
use {defmt_rtt as _, panic_probe as _};

use dsp_fixedpoint::Q32;
use dsp_process::{Inplace, Process, Split};
use idsp::iir;

fn init() -> embassy_stm32::Peripherals {
    let mut config = Config::default();
    config.rcc.hsi = Some(HSIPrescaler::DIV1);
    config.rcc.csi = true;
    config.rcc.hsi48 = Some(Default::default());
    config.rcc.pll1 = Some(Pll {
        source: PllSource::HSI,
        prediv: PllPreDiv::DIV4,
        mul: PllMul::MUL50,
        divp: Some(PllDiv::DIV2),
        divq: Some(PllDiv::DIV8),
        divr: None,
        fracn: None,
    });
    config.rcc.pll2 = Some(Pll {
        source: PllSource::HSI,
        prediv: PllPreDiv::DIV4,
        mul: PllMul::MUL50,
        divp: Some(PllDiv::DIV8), // 100 Mhz
        divq: None,
        divr: None,
        fracn: None,
    });
    config.rcc.sys = Sysclk::PLL1_P; // 400 Mhz
    config.rcc.ahb_pre = AHBPrescaler::DIV2; // 200 Mhz
    config.rcc.apb1_pre = APBPrescaler::DIV2; // 100 Mhz
    config.rcc.apb2_pre = APBPrescaler::DIV2; // 100 Mhz
    config.rcc.apb3_pre = APBPrescaler::DIV2; // 100 Mhz
    config.rcc.apb4_pre = APBPrescaler::DIV2; // 100 Mhz
    config.rcc.voltage_scale = VoltageScale::Scale1;
    config.rcc.mux.adcsel = mux::Adcsel::PLL2_P;
    // config.rcc.supply_config = SupplyConfig::DirectSMPS;
    embassy_stm32::init(config)
}

/// Setup ITCM and load its code from flash.
///
/// For portability and maintainability this is implemented in Rust.
/// Since this is implemented in Rust the compiler may assume that bss and data are set
/// up already. There is no easy way to ensure this implementation will never need bss
/// or data. Hence we can't safely run this as the cortex-m-rt `pre_init` hook before
/// bss/data is setup.
///
/// Calling (through IRQ or directly) any code in ITCM before having called
/// this method is undefined.
fn load_itcm() {
    unsafe extern "C" {
        // ZST (`()`: not layout-stable. empty/zst struct in `repr(C)``: not "proper" C)
        unsafe static mut __sitcm: [u32; 0];
        unsafe static mut __eitcm: [u32; 0];
        unsafe static mut __siitcm: [u32; 0];
    }
    // NOTE(unsafe): Assuming the address symbols from the linker as well as
    // the source instruction data are all valid, this is safe as it only
    // copies linker-prepared data to where the code expects it to be.
    // Calling it multiple times is safe as well.

    // ITCM is enabled on reset on our CPU but might not be on others.
    // Keep for completeness.
    const ITCMCR: *mut u32 = 0xE000_EF90usize as _;
    unsafe {
        ptr::write_volatile(ITCMCR, ptr::read_volatile(ITCMCR) | 1);
    }

    // Ensure ITCM is enabled before loading.
    atomic::fence(Ordering::SeqCst);

    let sitcm = ptr::addr_of_mut!(__sitcm) as *mut u32;
    let eitcm = ptr::addr_of!(__eitcm) as *const u32;
    let siitcm = ptr::addr_of!(__siitcm) as *const u32;

    unsafe {
        let len = eitcm.offset_from(sitcm) as usize;
        // Load code into ITCM.
        ptr::copy(siitcm, sitcm, len);
    }

    // Ensure ITCM is loaded before potentially executing any instructions from it.
    atomic::fence(Ordering::SeqCst);
    cortex_m::asm::dsb();
    cortex_m::asm::isb();
}

//#[inline(never)]
//#[unsafe(link_section = ".itcm.timeit")]
fn time<F: FnMut()>(mut func: F) -> u32 {
    // cortex_m::interrupt::free(|_cs| {
    // cache warming
    func();
    func();
    func();
    let c = DWT::cycle_count();
    atomic::compiler_fence(Ordering::SeqCst);
    func();
    atomic::compiler_fence(Ordering::SeqCst);
    DWT::cycle_count().wrapping_sub(c)
    // })
}

#[inline(never)]
#[unsafe(link_section = ".itcm.timeit")]
fn noinline<F: FnOnce()>(func: F) {
    func()
}

// place the no-inline here to aid cache warming
// #[inline(never)]
// #[unsafe(link_section = ".itcm.timeit")]
fn timeit<F: FnMut()>(mut func: F) -> u32 {
    let c1 = time(|| {
        noinline(&mut func);
    });
    let c2 = time(|| {
        noinline(&mut func);
        noinline(&mut func);
    });
    let c3 = time(|| {
        noinline(&mut func);
        noinline(&mut func);
        noinline(&mut func);
    });
    let err = (c2 + c2) as f32 / (c1 + c3) as f32 - 1.0;
    defmt::assert!(err.abs() < 1e-2, "cycle error: {=f32}", err);
    c2 - c1
}

#[derive(Default, Debug, Clone, defmt::Format)]
struct Cycles {
    block: u32,
    inplace: u32,
}

#[derive(Default, Debug, Clone, defmt::Format)]
struct CyclesResults {
    slice: Cycles,
    chunked: Cycles,
    chunk: Cycles,
}

impl CyclesResults {
    fn show(&self, name: &str) {
        const PAD: &str = "                     ";
        info!(
            "{=str}{=str} {=[?]}",
            name[..name.len().min(PAD.len())],
            PAD[..(PAD.len() - name.len().min(PAD.len()))],
            [
                self.slice.block as f32 / BLOCK as f32,
                self.slice.inplace as f32 / BLOCK as f32,
                self.chunked.block as f32 / BLOCK as f32,
                self.chunked.inplace as f32 / BLOCK as f32,
                self.chunk.block as f32 / CHUNK as f32,
                self.chunk.inplace as f32 / CHUNK as f32,
            ]
        );
    }
}

const BLOCK: usize = 1 << 10;
const CHUNK: usize = 4;

fn bench_process<C, S, X>(proc: &mut Split<C, S>) -> CyclesResults
where
    for<'a> Split<&'a C, &'a mut S>: Inplace<X>,
    X: Copy + Default,
{
    let proc = black_box(proc);
    let mut xy = [black_box([X::default(); BLOCK]); 2];
    let [x, y] = xy.each_mut();
    let slice = Cycles {
        block: timeit(|| proc.as_mut().block(x, y)),
        inplace: timeit(|| proc.as_mut().inplace(y)),
    };
    let ((x, []), (y, [])) = (x.as_chunks::<CHUNK>(), y.as_chunks_mut::<CHUNK>()) else {
        defmt::unreachable!()
    };
    let chunked = Cycles {
        block: timeit(|| {
            for (x, y) in x.iter().zip(y.iter_mut()) {
                proc.as_mut().block(x, y);
            }
        }),
        inplace: timeit(|| {
            for y in y.iter_mut() {
                proc.as_mut().inplace(y);
            }
        }),
    };
    let (x0, y0) = (&x[0], &mut y[0]);
    let chunk = Cycles {
        block: timeit(|| proc.as_mut().block(x0, y0)),
        inplace: timeit(|| proc.as_mut().inplace(y0)),
    };
    CyclesResults {
        slice,
        chunked,
        chunk,
        ..Default::default()
    }
}

struct BiquadAdapter<B>(B);
impl<B: biquad::Biquad<X>, X: Copy + Float> dsp_process::Process<X> for BiquadAdapter<B> {
    fn process(&mut self, x: X) -> X {
        self.0.run(x)
    }
}
impl<B, X: Copy> dsp_process::Inplace<X> for BiquadAdapter<B> where Self: dsp_process::Process<X> {}

fn biquad_rs_bench_one<P, X>(proc: &mut P) -> CyclesResults
where
    P: biquad::Biquad<X>,
    X: Float + Default,
{
    let proc = black_box(proc);
    let mut xy = [black_box([X::default(); BLOCK]); 2];
    let [x, y] = xy.each_mut();
    let slice = Cycles {
        block: timeit(|| {
            for (x, y) in x.iter().zip(y.iter_mut()) {
                *y = proc.run(*x);
            }
        }),
        inplace: timeit(|| {
            for y in y.iter_mut() {
                *y = proc.run(*y);
            }
        }),
    };
    let ((x, []), (y, [])) = (x.as_chunks::<CHUNK>(), y.as_chunks_mut::<CHUNK>()) else {
        defmt::unreachable!()
    };
    let chunked = Cycles {
        block: timeit(|| {
            for (x, y) in x.iter().zip(y.iter_mut()) {
                for (x, y) in x.iter().zip(y.iter_mut()) {
                    *y = proc.run(*x);
                }
            }
        }),
        inplace: timeit(|| {
            for y in y.iter_mut() {
                for y in y.iter_mut() {
                    *y = proc.run(*y);
                }
            }
        }),
    };
    let (x0, y0) = (&x[0], &mut y[0]);
    let chunk = Cycles {
        block: timeit(|| {
            for (x, y) in x0.iter().zip(y0.iter_mut()) {
                *y = proc.run(*x);
            }
        }),
        inplace: timeit(|| {
            for y in y0.iter_mut() {
                *y = proc.run(*y);
            }
        }),
    };
    CyclesResults {
        slice,
        chunked,
        chunk,
        ..Default::default()
    }
}

fn coeff_def<T: Float>() -> biquad::Coefficients<T> {
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
    let _p = init();
    let mut c = unwrap!(cortex_m::Peripherals::take());

    load_itcm();

    c.SCB.enable_icache();
    c.SCB.enable_dcache(&mut c.CPUID);

    c.DCB.enable_trace();
    c.DWT.enable_cycle_counter();

    info!("Starting");

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

    biquad_rs_bench_one(&mut biquad::DirectForm1::new(coeff_def::<f32>())).show("biquad df1 f32");
    bench_process(&mut Split::stateful(BiquadAdapter(
        biquad::DirectForm1::new(coeff_def::<f32>()),
    )))
    .show("biquad adapter df1 f32");

    bench_process(&mut Split::<
        iir::Biquad<f32>,
        iir::DirectForm2Transposed<f32>,
    >::default())
    .show("idsp df2t f32");

    biquad_rs_bench_one(&mut biquad::DirectForm2Transposed::new(coeff_def::<f32>()))
        .show("biquad df2t f32");
    bench_process(&mut Split::stateful(BiquadAdapter(
        biquad::DirectForm2Transposed::new(coeff_def::<f32>()),
    )))
    .show("biquad adapter df2t f32");

    bench_process(&mut Split::<iir::BiquadClamp<f32>, iir::DirectForm1<f32>>::default())
        .show("idsp clamp f32");

    bench_process(&mut Split::<iir::Biquad<f64>, iir::DirectForm1<f64>>::default())
        .show("idsp f64");

    biquad_rs_bench_one(&mut biquad::DirectForm1::new(coeff_def::<f64>())).show("biquad df1 f64");
    bench_process(&mut Split::stateful(BiquadAdapter(
        biquad::DirectForm1::new(coeff_def::<f64>()),
    )))
    .show("biquad adapter df1 f64");

    bench_process(&mut Split::<iir::BiquadClamp<f64>, iir::DirectForm1<f64>>::default())
        .show("idsp clamp f64");

    use dsp_process::{Add, Identity, Pair, Parallel, Unsplit};
    use idsp::iir::wdf::Wdf;
    bench_process::<_, _, i32>(&mut Split::new(
        Pair::new((
            (
                Unsplit(&Identity),
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
            Unsplit(&Add),
        )),
        Default::default(),
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
