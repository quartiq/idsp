/// 2-argument arctangent function.
///
/// This implementation uses all integer arithmetic for fast
/// computation. It is designed to have high accuracy near the axes
/// and lower away from the axes. It is additionally designed so that
/// the error changes slowly with respect to the angle.
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
pub fn atan2(y: i32, x: i32) -> i32 {
    // Map into first octant
    let sign_x = x < 0;
    let mut x = if sign_x { x.wrapping_neg() } else { x } as u32;
    let sign_y = y < 0;
    let mut y = if sign_y { y.wrapping_neg() } else { y } as u32;
    let y_greater = y > x;
    if y_greater {
        (x, y) = (y, x);
    }

    // Normalize for maximum ratio dynamic range
    let z = (x.leading_zeros() as i32).min(15);
    x >>= 15 - z;
    if x == 0 {
        return 0;
    }

    // Compute ratio
    let r = (y << z) / x;
    debug_assert!(r <= 1 << 16);

    // This approximation is a very common one. It is mentione in
    // https://math.stackexchange.com/a/1105038/583981 and
    // Sreeraman Rajan, Sichun Wang, Robert Inkol, and Alain Joyal:
    // Efficient Approximations for the Arctangent Function,
    // IEEE Signal Processing Magazine [108] May 2006
    // This version is rewritten and optimized for fixed point.
    // One could do another order of `r + r*(1 - r)*(0.2447 + 0.0663*r)/PI_4`
    // but that would only bring a factor of 2 smaller max and rms error.
    const K: u32 = (0.2732 / core::f32::consts::FRAC_PI_4 * (1 << 15) as f32) as _;
    let mut angle = ((r << 14) + (K * ((r * ((1 << 15) - r)) >> 16))) as i32;

    // Unmap octant
    if y_greater {
        angle = (1 << 30) - angle;
    }
    if sign_x {
        angle = i32::MAX - angle;
    }
    // Bitflip ends up in slightly faster assembly than negation
    if sign_y {
        !angle
    } else {
        angle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn angle_to_axis(angle: f64) -> f64 {
        let angle = angle % (PI / 2.);
        (PI / 2. - angle).min(angle)
    }

    #[test]
    fn atan2_absolute_error() {
        const N: usize = 321;
        let mut test_vals = [0i32; N + 4];
        let scale = (1i64 << 31) as f64;
        for i in 0..N {
            test_vals[i] = (scale * (-1. + 2. * i as f64 / N as f64)) as i32;
        }

        assert!(test_vals.contains(&i32::MIN));
        test_vals[N] = i32::MAX;
        test_vals[N + 1] = 0;
        test_vals[N + 2] = -1;
        test_vals[N + 3] = 1;

        let mut rms_err = 0f64;
        let mut abs_err = 0f64;
        let mut rel_err = 0f64;

        for &x in test_vals.iter() {
            for &y in test_vals.iter() {
                let want = (y as f64 / scale).atan2(x as f64 / scale);
                let have = atan2(y, x) as f64 * PI / scale;

                let err = (have - want).abs();
                abs_err = abs_err.max(err);
                rms_err += err * err;
                if err > 3e-5 {
                    rel_err = rel_err.max(err / angle_to_axis(want));
                }
            }
        }
        rms_err = rms_err.sqrt() / test_vals.len() as f64;
        println!("max abs err: {:.2e}", abs_err);
        println!("rms abs err: {:.2e}", rms_err);
        println!("max rel err: {:.2e}", rel_err);
        assert!(abs_err < 3.8e-3);
        assert!(rms_err < 2.7e-3);
        assert!(rel_err < 0.6);
    }
}
