#![no_std]
#![no_main]

use cortex_m::asm;
use defmt::*;
use {defmt_rtt as _, panic_probe as _};

use dsp_process::FnProcess;
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
    bench_inplace(&mut FnProcess(core::convert::identity::<i32>)).show("identity");
    bench_inplace(&mut FnProcess(core::hint::black_box::<i32>)).show("black_box");

    bench_process(&mut FnProcess(|x| idsp::cossin(x))).show("cossin");
    bench_process(&mut FnProcess(|[x, y]: [_; 2]| idsp::atan2(y, x))).show("atan2");

    info!("Done");

    asm::bkpt();
    loop {
        asm::nop();
    }
}
