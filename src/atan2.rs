use dsp_fixedpoint::Q32;

include!(concat!(env!("OUT_DIR"), "/atan2_divi_table.rs"));

/// Fixed point unsigned multiplication without roudning bias
#[inline(always)]
fn mul_q31(x: u32, y: u32) -> u32 {
    ((x as u64 * y as u64) >> 31) as u32
}

/// Divide y/x with y <= x
fn divi(y: u32, x: u32) -> u32 {
    debug_assert!(y <= x);
    if x == 0 {
        return 0;
    }
    // Normalize `x` to Q1.31 on [1, 2), interpolate a reciprocal seed from a
    // small LUT, and refine it with one Newton step.
    let shift = x.leading_zeros();
    let y = y << shift;
    let x = x << shift;
    const FRAC_BITS: u32 = 31 - ATAN2_DIVI_DEPTH as u32;
    let rem = x & ((1 << FRAC_BITS) - 1);
    let idx = ((x << 1) >> (1 + FRAC_BITS)) as usize;
    let (base, slope) = ATAN2_DIVI_RECIP[idx];
    let step = ((slope as i64 * rem as i64) >> FRAC_BITS) as u32;
    let r0 = base.wrapping_add(step);
    mul_q31(y, mul_q31(r0, mul_q31(x, r0).wrapping_neg()))
}

/// Polynomial approximation to atan(x) to 11th order
fn atani(x: u32) -> u32 {
    const ATANI: [Q32<32>; 6] = [
        Q32::new(0x0517c2cd),
        Q32::new(-0x06c6496b),
        Q32::new(0x0fbdb021),
        Q32::new(-0x25b32e0a),
        Q32::new(0x43b34c81),
        Q32::new(-0x3bc823dd),
    ];
    // Evaluate the odd polynomial in x * P(x^2/4).
    let x2 = Q32::new(((x as i64 * x as i64) >> 32) as _);
    let r = ATANI
        .iter()
        .copied()
        .rev()
        .fold(Q32::new(0), |r, a| (r * x2) + a);
    (((r.inner as i64) * (x as i64)) >> 28) as u32
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
        assert!(abs_err < 2.3e-6);
        assert!(rms_err < 1.3e-6);
        assert!(rel_err < 1e-15);
    }

    #[test]
    fn atan2_small_equal_inputs() {
        let scale = PI / (1i64 << 31) as f64;
        for v in 1..1024 {
            let have = atan2(v, v) as f64 * scale;
            let want = PI / 4.0;
            assert!((have - want).abs() < 2.3e-6, "{v}: {have} vs {want}");
        }
    }

    #[test]
    fn atan2_small_vectors_do_not_break_near_origin() {
        let scale = PI / (1i64 << 31) as f64;
        let mut max_err = 0.0f64;
        for x in 1..512 {
            for y in 0..=x {
                let have = atan2(y, x) as f64 * scale;
                let want = (y as f64).atan2(x as f64);
                max_err = max_err.max((have - want).abs());
            }
        }
        assert!(max_err < 2.3e-6, "{max_err}");
    }

    #[test]
    fn atan2_zero_axis_is_exact() {
        assert_eq!(atan2(0, 1), 0);
        assert_eq!(atan2(0, i32::MAX), 0);
        assert_eq!(atan2(1, 0), 0x3fff_ffff);
        assert_eq!(atan2(i32::MAX, 0), 0x3fff_ffff);
    }
}
