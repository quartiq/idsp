#![no_std]
#![no_main]

use core::hint::black_box;

use cortex_m::asm;
use defmt::*;
use {defmt_rtt as _, panic_probe as _};

use dsp_process::{FnProcess, Split};
use idsp::hbf;

use idsp_embedded_bench::*;

#[cortex_m_rt::entry]
fn main() -> ! {
    info!("Setup");
    let mut c = unwrap!(cortex_m::Peripherals::take());

    // c.SCB.enable_icache();
    // c.SCB.enable_dcache(&mut c.CPUID);

    c.DCB.enable_trace();
    c.DWT.enable_cycle_counter();

    info!("Starting");

    CyclesResults::header();

    bench_inplace(&mut Split::new(
        hbf::EvenSymmetric([0.0; 8]),
        [0.0; 16 + 32],
    ))
    .show("fir evensym");

    bench_process(
        &mut (Split::new(
            hbf::HBF_INT_CASCADE.inner.0.inner.0,
            hbf::HbfInt8::default(),
        ) * Split::stateful(FnProcess(|x: [f32; 8]| {
            black_box(x);
        })))
        .major::<[_; 32]>(),
    )
    .show("hbf int8");

    bench_process(
        &mut (Split::stateful(FnProcess(|_: ()| black_box([0.0; 8])))
            * Split::new(
                hbf::HBF_DEC_CASCADE.inner.1.inner.1,
                hbf::HbfDec8::default(),
            ))
        .major::<[_; 32]>(),
    )
    .show("hbf dec8");

    info!("Done");

    asm::bkpt();
    loop {
        asm::nop();
    }
}
