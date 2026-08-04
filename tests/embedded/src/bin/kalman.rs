#![no_std]
#![no_main]

use cortex_m::asm;
use defmt::*;
use dsp_fixedpoint::Q32;
use dsp_process::Split;
use idsp::kalman::{
    ConstantVelocity, Direct, Estimate, Kalman, Observation, RandomWalk, Transition,
};
use num_traits::{ConstOne, ConstZero};

use idsp_embedded_bench::*;

const KINEMATIC_4: [[f32; 4]; 4] = [
    [1.0, 0.25, 0.03125, 1.0 / 384.0],
    [0.0, 1.0, 0.25, 0.03125],
    [0.0, 0.0, 1.0, 0.25],
    [0.0, 0.0, 0.0, 1.0],
];

#[cortex_m_rt::entry]
fn main() -> ! {
    info!("Setup");
    let mut c = unwrap!(cortex_m::Peripherals::take());
    c.DCB.enable_trace();
    c.DWT.enable_cycle_counter();

    info!("Starting");
    CyclesResults::header();

    bench_inplace(&mut Split::new(
        RandomWalk::<f32>::new(0.01, 1.0),
        Estimate::new([0.0], [[10.0]]),
    ))
    .show("kalman f32 n1 structural");

    let estimate = Estimate::new([0.0; 2], [[10.0, 0.0], [0.0, 1.0]]);
    bench_inplace(&mut Split::new(
        Kalman::new(
            Transition::new([[1.0, 1.0], [0.0, 1.0]], [[0.01, 0.0], [0.0, 0.01]]),
            Observation::new([1.0, 0.0], 1.0),
        ),
        estimate,
    ))
    .show("kalman f32 n2 dense");
    bench_inplace(&mut Split::new(
        Kalman::new(
            ConstantVelocity::new(1.0, [[0.01, 0.0], [0.0, 0.01]]),
            Direct::<f32, 0>::new(1.0),
        ),
        estimate,
    ))
    .show("kalman f32 n2 structural");

    let transition = Transition::new(
        KINEMATIC_4,
        [
            [0.01, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0],
            [0.0, 0.0, 0.0, 0.01],
        ],
    );
    let estimate = Estimate::new(
        [0.0; 4],
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ],
    );
    bench_inplace(&mut Split::new(
        Kalman::new(transition, Observation::new([1.0, 0.0, 0.0, 0.0], 1.0)),
        estimate,
    ))
    .show("kalman f32 n4 dense");
    bench_inplace(&mut Split::new(
        Kalman::new(transition, Direct::<f32, 0>::new(1.0)),
        estimate,
    ))
    .show("kalman f32 n4 direct");

    type C = Q32<28>;
    let transition = Transition::new(
        [[C::ONE, C::from_f32(0.25)], [C::ZERO, C::ONE]],
        [[16, 0], [0, 4]],
    );
    let estimate = Estimate::new([0, 40], [[256, 0], [0, 64]]);
    bench_inplace(&mut Split::new(
        Kalman::new(transition, Observation::new([C::ONE, C::ZERO], 64)),
        estimate,
    ))
    .show("kalman q32 n2 dense");
    bench_inplace(&mut Split::new(
        Kalman::new(
            ConstantVelocity::new(C::from_f32(0.25), [[16, 0], [0, 4]]),
            Direct::<C, 0, i32>::new(64),
        ),
        estimate,
    ))
    .show("kalman q32 n2 structural");

    let transition = Transition::new(
        KINEMATIC_4.map(|row| row.map(C::from_f32)),
        [[16, 0, 0, 0], [0, 4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    );
    let estimate = Estimate::new(
        [0, 40, 0, 0],
        [[256, 0, 0, 0], [0, 64, 0, 0], [0, 0, 16, 0], [0, 0, 0, 4]],
    );
    bench_inplace(&mut Split::new(
        Kalman::new(
            transition,
            Observation::new([C::ONE, C::ZERO, C::ZERO, C::ZERO], 64),
        ),
        estimate,
    ))
    .show("kalman q32 n4 dense");
    bench_inplace(&mut Split::new(
        Kalman::new(transition, Direct::<C, 0, i32>::new(64)),
        estimate,
    ))
    .show("kalman q32 n4 direct");

    info!("Done");
    asm::bkpt();
    loop {
        asm::nop();
    }
}
