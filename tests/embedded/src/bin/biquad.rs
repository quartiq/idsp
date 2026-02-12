#![no_std]
#![no_main]

use cortex_m::asm;
use defmt::*;
use num_traits::Float;

use dsp_fixedpoint::Q32;
use dsp_process::{Add, Identity, Pair, Parallel, Unsplit};
use dsp_process::{Inplace, Process, Split};
use idsp::iir::{self, wdf::Wdf};

use idsp_embedded_bench::*;

struct BiquadProcess<B>(B);
impl<B: biquad::Biquad<X>, X: Copy + Float> Process<X> for BiquadProcess<B> {
    fn process(&mut self, x: X) -> X {
        self.0.run(x)
    }
}
impl<B, X: Copy> Inplace<X> for BiquadProcess<B> where Self: Process<X> {}

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

    // c.SCB.enable_icache();
    // c.SCB.enable_dcache(&mut c.CPUID);

    c.DCB.enable_trace();
    c.DWT.enable_cycle_counter();

    info!("Starting");

    CyclesResults::header();

    bench_inplace(&mut Split::<iir::Biquad<Q32<29>>, iir::DirectForm1<i32>>::default())
        .show("idsp q32");

    bench_inplace(&mut Split::<
        iir::BiquadClamp<Q32<29>, i32>,
        iir::DirectForm1<i32>,
    >::default())
    .show("idsp clamp q32");

    bench_inplace(&mut Split::<iir::Biquad<Q32<29>>, iir::DirectForm1Wide>::default())
        .show("idsp wide q32");

    bench_inplace(&mut Split::<iir::Biquad<Q32<29>>, iir::DirectForm1Dither>::default())
        .show("idsp dither q32");

    bench_inplace(&mut Split::<
        iir::BiquadClamp<Q32<29>, i32>,
        iir::DirectForm1Wide,
    >::default())
    .show("idsp clamp wide q32");

    bench_inplace(&mut Split::<
        iir::BiquadClamp<Q32<29>, i32>,
        iir::DirectForm1Dither,
    >::default())
    .show("idsp clamp dither q32");

    bench_inplace(&mut Split::<iir::Biquad<f32>, iir::DirectForm1<f32>>::default())
        .show("idsp f32");

    bench_inplace(&mut BiquadProcess(biquad::DirectForm1::new(
        coefficients_default::<f32>(),
    )))
    .show("biquad df1 f32");

    bench_inplace(&mut Split::<
        iir::Biquad<f32>,
        iir::DirectForm2Transposed<f32>,
    >::default())
    .show("idsp df2t f32");

    bench_inplace(&mut BiquadProcess(biquad::DirectForm2Transposed::new(
        coefficients_default::<f32>(),
    )))
    .show("biquad df2t f32");

    bench_inplace(&mut Split::<iir::BiquadClamp<f32>, iir::DirectForm1<f32>>::default())
        .show("idsp clamp f32");

    bench_inplace(&mut Split::<iir::Biquad<f64>, iir::DirectForm1<f64>>::default())
        .show("idsp f64");

    bench_inplace(&mut BiquadProcess(biquad::DirectForm1::new(
        coefficients_default::<f64>(),
    )))
    .show("biquad df1 f64");

    bench_inplace(&mut Split::<
        iir::Biquad<f64>,
        iir::DirectForm2Transposed<f64>,
    >::default())
    .show("idsp df2t f64");

    bench_inplace(&mut BiquadProcess(biquad::DirectForm2Transposed::new(
        coefficients_default::<f64>(),
    )))
    .show("biquad df2t f64");

    bench_inplace(&mut Split::<iir::BiquadClamp<f64>, iir::DirectForm1<f64>>::default())
        .show("idsp clamp f64");

    bench_inplace::<_, i32>(&mut Split::new(
        Pair::new((
            (
                (),
                Parallel((
                    [
                        unwrap!(Wdf::<_, 0xad>::quantize(&[-0.9, 0.9])),
                        unwrap!(Wdf::<_, 0xad>::quantize(&[-0.6, 0.7])),
                    ],
                    (
                        unwrap!(Wdf::<_, 0xad>::quantize(&[-0.7, 0.6])),
                        unwrap!(Wdf::<_, 0xa>::quantize(&[0.8])),
                    ),
                )),
            ),
            (),
        )),
        ((Unsplit(Identity), Default::default()), Unsplit(Add)),
    ))
    .show("idsp wdf-ca-7");

    // Filter an i32 array in place with a 19th order half band WDF.
    //
    // Gazsi 1985, Example 5
    let p = (
        (
            (
                Wdf::<1, 0x1>::default(),
                unwrap!(Wdf::<_, 0x1c>::quantize(&[-0.226119, 0.0])),
            ),
            [
                unwrap!(Wdf::<_, 0x1d>::quantize(&[-0.602422, 0.0])),
                unwrap!(Wdf::quantize(&[-0.83932, 0.0])),
                unwrap!(Wdf::quantize(&[-0.950847, 0.0])),
            ],
        ),
        (
            [
                unwrap!(Wdf::<_, 0x1c>::quantize(&[-0.063978, 0.0])),
                unwrap!(Wdf::quantize(&[-0.423068, 0.0])),
            ],
            [
                unwrap!(Wdf::<_, 0x1d>::quantize(&[-0.741327, 0.0])),
                unwrap!(Wdf::quantize(&[-0.905567, 0.0])),
                unwrap!(Wdf::quantize(&[-0.984721, 0.0])),
            ],
        ),
    );

    let mut f = (Split::stateful(Identity)
        * Split::new(p, Default::default()).parallel()
        * Split::stateful(Add))
    .minor::<[_; _]>();
    bench_inplace::<_, i32>(&mut f).show("idsp wdf-ca-19");

    bench_inplace(&mut Split::<
        iir::Biquad<dsp_fixedpoint::Q16<13>>,
        iir::DirectForm1<i16>,
    >::default())
    .show("idsp q16");
    bench_inplace(&mut Split::<
        iir::Biquad<dsp_fixedpoint::Q64<61>>,
        iir::DirectForm1<i64>,
    >::default())
    .show("idsp q64");

    bench_inplace(&mut Split::<idsp::Lowpass<1>, idsp::LowpassState<1>>::default())
        .show("lowpass1");
    bench_inplace(&mut Split::<idsp::Lowpass<2>, idsp::LowpassState<2>>::default())
        .show("lowpass2");

    info!("Done");

    asm::bkpt();
    loop {
        asm::nop();
    }
}
