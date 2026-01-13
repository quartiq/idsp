fn divi(mut y: u32, mut x: u32) -> u32 {
    debug_assert!(y <= x);
    let z = y.leading_zeros().min(15);
    y <<= z;
    x += (1 << (15 - z)) - 1;
    x >>= 16 - z;
    if x == 0 {
        0 // x == y == 0
    } else {
        ((y / x) << 15) + (1 << 14)
    }
}

fn atani(x: u32) -> u32 {
    const A: [i32; 6] = [
        0x0517c2cd,
        -0x06c6496b,
        0x0fbdb021,
        -0x25b32e0a,
        0x43b34c81,
        -0x3bc823dd,
    ];
    let x2 = ((x as i64 * x as i64) >> 32) as i32;
    let r = A
        .iter()
        .rev()
        .fold(0, |r, a| ((r as i64 * x2 as i64) >> 32) as i32 + a);
    ((r as i64 * x as i64) >> 28) as _
}

/// 2-argument arctangent function.
///
/// This implementation uses all integer arithmetic for fast
/// computation.
///
/// # Arguments
///
/// * `y` - Y-axis component.
/// * `x` - X-axis component.
///
/// # Returns
///
/// The angle between the x-axis and the ray to the point (x,y). The
/// result range is from i32::MIN to i32::MAX, where i32::MIN
/// represents -pi and, equivalently, +pi. i32::MAX represents one
/// count less than +pi.
pub fn atan2(mut y: i32, mut x: i32) -> i32 {
    let mut k = 0u32;
    if y < 0 {
        y = y.saturating_neg();
        k ^= u32::MAX;
    }
    if x < 0 {
        x = x.saturating_neg();
        k ^= u32::MAX >> 1;
    }
    if y > x {
        (y, x) = (x, y);
        k ^= u32::MAX >> 2;
    }
    let r = atani(divi(y as _, x as _));
    (r ^ k) as _
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn atan2_error() {
        const N: isize = 201;
        for i in 0..N {
            let p = ((1. - 2. * i as f64 / N as f64) * i32::MIN as f64) as i32;
            let pf = p as f64 / i32::MIN as f64 * -PI;
            let y = (pf.sin() * i32::MAX as f64) as i32;
            let x = (pf.cos() * i32::MAX as f64) as i32;
            let _p0 = (y as f64).atan2(x as f64);
            let pp = atan2(y, x);
            let pe = -(pp as f64 / i32::MIN as f64);
            println!(
                "y:{:.5e}, x:{:.5e}, p/PI:{:.5e}: pe:{:.5e}, pe*PI-p0:{:.5e}",
                y as f64 / i32::MAX as f64,
                x as f64 / i32::MAX as f64,
                pf / PI,
                pe,
                pe * PI - pf
            );
        }
    }

    fn angle_to_axis(angle: f64) -> f64 {
        let angle = angle % (PI / 2.);
        (PI / 2. - angle).min(angle)
    }

    #[test]
    fn atan2_absolute_error() {
        const N: usize = 321;
        let mut test_vals = [0i32; N + 2];
        let scale = (1i64 << 31) as f64;
        for i in 0..N {
            test_vals[i] = (scale * (-1. + 2. * i as f64 / N as f64)) as i32;
        }

        assert!(test_vals.contains(&i32::MIN));
        test_vals[N] = i32::MAX;
        test_vals[N + 1] = 0;

        let mut rms_err = 0f64;
        let mut abs_err = 0f64;
        let mut rel_err = 0f64;

        for &x in test_vals.iter() {
            for &y in test_vals.iter() {
                let want = (y as f64).atan2(x as f64);
                let have = atan2(y, x) as f64 * (PI / scale);
                let err = (have - want).abs();
                abs_err = abs_err.max(err);
                rms_err += err * err;
                if err > 3e-5 {
                    println!("{:.5e}/{:.5e}: {:.5e} vs {:.5e}", y, x, have, want);
                    println!("y/x {} {}", y, x);
                    rel_err = rel_err.max(err / angle_to_axis(want));
                }
            }
        }
        rms_err = rms_err.sqrt() / test_vals.len() as f64;
        println!("max abs err: {:.2e}", abs_err);
        println!("rms abs err: {:.2e}", rms_err);
        println!("max rel err: {:.2e}", rel_err);
        assert!(abs_err < 1.2e-5);
        assert!(rms_err < 4.2e-6);
        assert!(rel_err < 1e-12);
    }
}
