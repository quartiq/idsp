use crate::Accu;
use core::num::Wrapping as W;
use dsp_process::SplitProcess;

/// Reciprocal PLL.
///
/// Consumes noisy, quantized timestamps of a reference signal and reconstructs
/// the phase and frequency of the update() invocations with respect to (and in units of
/// 1 << 32 of) that reference.
/// In other words, `update()` rate relative to reference frequency,
/// `u32::MAX + 1` corresponding to both being equal.
#[derive(Copy, Clone, Default)]
pub struct RPLL {
    x: W<i32>,  // previous timestamp
    ff: W<u32>, // current frequency estimate from frequency loop
    f: W<u32>,  // current frequency estimate from both frequency and phase loop
    y: W<i32>,  // current phase estimate
}

/// RPLL configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct RPLLConfig {
    /// 1 << dt2 is the counter rate to update() rate ratio
    pub dt2: u8,
    /// Frequency lock settling time.
    ///
    /// `1 << shift_frequency` is
    /// frequency lock settling time in counter periods. The settling time must be larger
    /// than the signal period to lock to.
    pub shift_frequency: u8,
    /// Phase lock settling time.
    ///
    /// Usually one less than `shift_frequency` (see there).
    pub shift_phase: u8,
}

impl SplitProcess<Option<W<i32>>, Accu<W<i32>>, RPLL> for RPLLConfig {
    /// Advance the RPLL and optionally supply a new timestamp.
    ///
    /// * input: Optional new timestamp.
    ///   There can be at most one timestamp per `update()` cycle (1 << dt2 counter cycles).
    ///
    /// Returns:
    /// A tuple containing the current phase (wrapping at the i32 boundary, pi) and
    /// frequency.
    fn process(&self, state: &mut RPLL, x: Option<W<i32>>) -> Accu<W<i32>> {
        debug_assert!(self.shift_frequency >= self.dt2);
        debug_assert!(self.shift_phase >= self.dt2);
        // Advance phase
        state.y += W(state.f.0 as i32);
        if let Some(x) = x {
            // Reference period in counter cycles
            let dx = x - state.x;
            // Store timestamp for next time.
            state.x = x;
            // Phase using the current frequency estimate
            let p_sig_64 = W(state.ff.0 as u64 * dx.0 as u64);
            // Add half-up rounding bias and apply gain/attenuation
            let p_sig = W(((p_sig_64 + W((1u32 << (self.shift_frequency - 1)) as u64))
                >> self.shift_frequency as usize)
                .0 as _);
            // Reference phase (1 << dt2 full turns) with gain/attenuation applied
            let p_ref = W(1u32 << (32 + self.dt2 - self.shift_frequency));
            // Update frequency lock
            state.ff += p_ref - p_sig;
            // Time in counter cycles between timestamp and "now"
            let dt = W((-x & W((1 << self.dt2) - 1)).0 as _);
            // Reference phase estimate "now"
            let y_ref = W(((state.f >> self.dt2 as usize) * dt).0 as _);
            // Phase error with gain
            let dy = (y_ref - state.y) >> (self.shift_phase - self.dt2) as usize;
            // Current frequency estimate from frequency lock and phase error
            state.f = state.ff + W(dy.0 as u32);
        }
        Accu::new(state.y, W(state.f.0 as _))
    }
}

impl RPLL {
    /// Return the current phase estimate
    pub fn phase(&self) -> W<i32> {
        self.y
    }

    /// Return the current frequency estimate
    pub fn frequency(&self) -> W<u32> {
        self.f
    }
}

#[cfg(test)]
mod test {
    use super::{RPLL, RPLLConfig};
    use core::num::Wrapping as W;
    use dsp_process::SplitProcess;
    use rand::{prelude::*, rngs::StdRng};
    use std::vec::Vec;

    #[test]
    fn make() {
        let _ = RPLL::default();
    }

    struct Harness {
        rpll: RPLL,
        rpll_config: RPLLConfig,
        noise: i32,
        period: i32,
        next: W<i32>,
        next_noisy: W<i32>,
        time: W<i32>,
        rng: StdRng,
    }

    impl Harness {
        fn default() -> Self {
            Self {
                rpll: RPLL::default(),
                rpll_config: RPLLConfig {
                    dt2: 8,
                    shift_frequency: 9,
                    shift_phase: 8,
                },
                noise: 0,
                period: 333,
                next: W(111),
                next_noisy: W(111),
                time: W(0),
                rng: StdRng::seed_from_u64(42),
            }
        }

        fn run(&mut self, n: usize) -> (Vec<f32>, Vec<f32>) {
            assert!(self.period >= 1 << self.rpll_config.dt2);
            assert!(self.period < 1 << self.rpll_config.shift_frequency);
            assert!(self.period < 1 << self.rpll_config.shift_phase + 1);

            let mut y = Vec::<f32>::new();
            let mut f = Vec::<f32>::new();
            for _ in 0..n {
                let timestamp = if self.time - self.next_noisy >= W(0) {
                    assert!(self.time - self.next_noisy < W(1 << self.rpll_config.dt2));
                    self.next += self.period;
                    let timestamp = self.next_noisy;
                    let p_noise = self.rng.random_range(-self.noise..=self.noise);
                    self.next_noisy = self.next + W(p_noise);
                    Some(timestamp)
                } else {
                    None
                };
                let _accu = self.rpll_config.process(&mut self.rpll, timestamp);
                let (yi, fi) = (self.rpll.phase(), self.rpll.frequency());

                let y_ref = W(
                    ((self.time - self.next).0 as i64 * (1i64 << 32) / self.period as i64) as i32,
                );
                // phase error
                y.push((yi - y_ref).0 as f32 / 2f32.powi(32));

                let p_ref = 1 << 32 + self.rpll_config.dt2;
                let p_sig = fi.0 as u64 * self.period as u64;
                // relative frequency error
                f.push(
                    p_sig.wrapping_sub(p_ref) as i64 as f32
                        / 2f32.powi(32 + self.rpll_config.dt2 as i32),
                );

                // advance time
                self.time += W(1 << self.rpll_config.dt2);
            }
            (y, f)
        }

        fn measure(&mut self, n: usize, limits: [f32; 4]) {
            let t_settle = (1 << self.rpll_config.shift_frequency - self.rpll_config.dt2 + 4)
                + (1 << self.rpll_config.shift_phase - self.rpll_config.dt2 + 4);
            self.run(t_settle);

            let (y, f) = self.run(n);
            // println!("{:?} {:?}", f, y);

            let fm = f.iter().copied().sum::<f32>() / f.len() as f32;
            let fs = f.iter().map(|f| (*f - fm).powi(2)).sum::<f32>().sqrt() / f.len() as f32;
            let ym = y.iter().copied().sum::<f32>() / y.len() as f32;
            let ys = y.iter().map(|y| (*y - ym).powi(2)).sum::<f32>().sqrt() / y.len() as f32;

            println!("f: {:.2e}±{:.2e}; y: {:.2e}±{:.2e}", fm, fs, ym, ys);

            let m = [fm, fs, ym, ys];

            print!("relative: ");
            for i in 0..m.len() {
                let rel = m[i].abs() / limits[i].abs();
                print!("{:.2e} ", rel);
                assert!(
                    rel <= 1.,
                    "idx {}, have |{:.2e}| > limit {:.2e}",
                    i,
                    m[i],
                    limits[i]
                );
            }
            println!();
        }
    }

    #[test]
    fn default() {
        let mut h = Harness::default();

        h.measure(1 << 16, [1e-11, 4e-8, 2e-8, 2e-8]);
    }

    #[test]
    fn noisy() {
        let mut h = Harness::default();
        h.noise = 10;
        h.rpll_config.shift_frequency = 23;
        h.rpll_config.shift_phase = 22;

        h.measure(1 << 16, [3e-9, 3e-6, 5e-4, 2e-4]);
    }

    #[test]
    fn narrow_fast() {
        let mut h = Harness::default();
        h.period = 990;
        h.next = W(351);
        h.next_noisy = h.next;
        h.noise = 5;
        h.rpll_config.shift_frequency = 23;
        h.rpll_config.shift_phase = 22;

        h.measure(1 << 16, [2e-9, 2e-6, 1e-3, 1e-4]);
    }

    #[test]
    fn narrow_slow() {
        let mut h = Harness::default();
        h.period = 1818181;
        h.next = W(35281);
        h.next_noisy = h.next;
        h.noise = 1000;
        h.rpll_config.shift_frequency = 23;
        h.rpll_config.shift_phase = 22;

        h.measure(1 << 16, [2e-5, 6e-4, 2e-4, 2e-4]);
    }

    #[test]
    fn wide_fast() {
        let mut h = Harness::default();
        h.period = 990;
        h.next = W(351);
        h.next_noisy = h.next;
        h.noise = 5;
        h.rpll_config.shift_frequency = 10;
        h.rpll_config.shift_phase = 9;

        h.measure(1 << 16, [2e-6, 3e-2, 2e-5, 2e-2]);
    }

    #[test]
    fn wide_slow() {
        let mut h = Harness::default();
        h.period = 1818181;
        h.next = W(35281);
        h.next_noisy = h.next;
        h.noise = 1000;
        h.rpll_config.shift_frequency = 21;
        h.rpll_config.shift_phase = 20;

        h.measure(1 << 16, [2e-4, 6e-3, 2e-4, 2e-3]);
    }

    #[test]
    fn batch_fast_narrow() {
        let mut h = Harness::default();
        h.rpll_config.dt2 = 8 + 3;
        h.period = 2431;
        h.next = W(35281);
        h.next_noisy = h.next;
        h.noise = 100;
        h.rpll_config.shift_frequency = 23;
        h.rpll_config.shift_phase = 23;

        h.measure(1 << 16, [1e-8, 2e-5, 6e-4, 6e-4]);
    }
}
