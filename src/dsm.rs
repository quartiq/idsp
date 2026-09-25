use dsp_process::Process;

/// Delta-sigma modulator
///
/// * MASH-(1)^K architecture
/// * `0 <= K <= 8` (`K=0` is valid but the output will be the constant quantized 0)
/// * The output range is `1 - (1 << K - 1)..=(1 << K - 1)`.
/// * Given constant input `x0`, the average output is `x0/(1 << 32)`.
/// * The noise goes up as `K * 20 dB/decade`.
///
/// ```
/// # use idsp::Dsm;
/// # use dsp_process::Process;
/// let mut d = Dsm::<3>::default();
/// let x = 0x87654321;
/// let n = 1 << 20;
/// let y = (0..n).map(|_| d.process(x) as f32).sum::<f32>() / n as f32;
/// let m = x as f32 / (1u64 << 32) as f32;
/// assert!((y / m - 1.0).abs() < (1.0 / n as f32).sqrt(), "{y} != {m}");
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Dsm<const K: usize> {
    a: [u32; K],
    c: [i8; K],
}

impl<const K: usize> Default for Dsm<K> {
    fn default() -> Self {
        const { assert!(K <= 8, "DSM order must be at most eight") }
        Self {
            a: [0; K],
            c: [0; K],
        }
    }
}

impl<const K: usize> Process<u32, i8> for Dsm<K> {
    /// Ingest input sample, emit new output.
    ///
    /// # Arguments
    /// * `x`: New input sample
    ///
    /// # Returns
    /// New output
    fn process(&mut self, x: u32) -> i8 {
        if K == 0 {
            return 0;
        }
        let mut d = 0i8;
        let mut c = false;
        self.a.iter_mut().fold(x, |x, a| {
            (*a, c) = a.overflowing_add(x);
            d = (d << 1) | c as i8;
            *a
        });
        self.c.iter_mut().take(K - 1).fold(d & 1, |mut y, c| {
            d >>= 1;
            (y, *c) = ((d & 1) + y - *c, y);
            y
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<const K: usize>() {
        for input in [0, 1, 0x1234_5678, 0x8000_0000, u32::MAX] {
            let mut dsm = Dsm::<K>::default();
            let mut sum = 0i64;
            for _ in 0..65536 {
                let y = dsm.process(input) as i64;
                if K == 0 {
                    assert_eq!(y, 0);
                } else {
                    assert!((1 - (1 << (K - 1))..=1 << (K - 1)).contains(&y));
                }
                sum += y;
            }
            if K > 0 {
                assert!((sum as f64 / 65536.0 - input as f64 / 4294967296.0).abs() < 0.002);
            }
        }
    }

    #[test]
    fn order_boundaries() {
        check::<0>();
        check::<1>();
        check::<8>();
    }
}
