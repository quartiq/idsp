// https://www.st.com/resource/en/design_tip/dt0085-coordinate-rotation-digital-computer-algorithm-cordic-to-compute-trigonometric-and-hyperbolic-functions-stmicroelectronics.pdf

include!(concat!(env!("OUT_DIR"), "/cordic_tables.rs"));

const ROTATE: bool = false;
const DEROTATE: bool = true;
const CIRCULAR: u8 = 0;
const HYPERBOLIC: u8 = 1;
const LINEAR: u8 = 2;

/// Generic CORDIC
#[inline]
fn cordic<const VECTORING: bool, const COORD: u8>(
    mut x: i32,
    mut y: i32,
    mut z: i32,
    iter: Option<usize>,
) -> (i32, i32) {
    // Microrotation table
    let a = match COORD {
        CIRCULAR => &CORDIC_CIRCULAR,
        _ => &CORDIC_HYPERBOLIC,
    };
    // MSB
    let left = if VECTORING {
        x < 0
    } else {
        z.wrapping_sub(i32::MIN >> 1) < 0
    };
    if left {
        x = -x;
        y = -y;
        z = z.wrapping_sub(i32::MIN);
    }
    // Hyperbolic repetition marker
    let mut k = 4;
    for (mut i, &(mut a)) in a[..iter.unwrap_or(a.len())].iter().enumerate() {
        // For linear mode the microrotations are computed, not looked up
        if COORD == LINEAR {
            a = (i32::MIN as u32 >> i) as _;
        }
        // Hyperbolic starts at i = 1
        if COORD == HYPERBOLIC {
            i += 1;
        }
        // Hyperbolic repeats some rotations for convergence
        let repeat = if COORD == HYPERBOLIC && i == k {
            k = 3 * i + 1;
            2
        } else {
            1
        };
        for _ in 0..repeat {
            // "sigma"
            let lower = if VECTORING { y <= 0 } else { z >= 0 };
            let (dx, dy) = (y >> i, x >> i);
            if lower {
                if COORD == CIRCULAR {
                    x -= dx;
                } else if COORD == HYPERBOLIC {
                    x += dx;
                }
                y += dy;
                z = z.wrapping_sub(a);
            } else {
                if COORD == CIRCULAR {
                    x += dx;
                } else if COORD == HYPERBOLIC {
                    x -= dx;
                }
                y -= dy;
                z = z.wrapping_add(a);
            }
        }
    }
    (x, if VECTORING { z } else { y })
}

/// Returns `F*(x*cos(z*PI) - y*sin(z*PI)), F*(x*sin(z*PI) + y*cos(z*PI))`
pub fn cos_sin(x: i32, y: i32, z: i32) -> (i32, i32) {
    cordic::<ROTATE, CIRCULAR>(x, y, z, None)
}

/// Returns `F*sqrt(x**2 + y**2), z + atan2(y, x)/PI`
pub fn sqrt_atan2(x: i32, y: i32, z: i32) -> (i32, i32) {
    cordic::<DEROTATE, CIRCULAR>(x, y, z, None)
}

/// Returns `x, y + x*z`
pub fn mul(x: i32, y: i32, z: i32) -> i32 {
    cordic::<ROTATE, LINEAR>(x, y, z, None).1
}

/// Returns `x, z + y/x`
pub fn div(x: i32, y: i32, z: i32) -> i32 {
    cordic::<DEROTATE, LINEAR>(x, y, z, None).1
}

/// Returns `G*(x*cosh(z) + y*sinh(z)), G*(x*sinh(z) + y*cosh(z))`
pub fn cosh_sinh(x: i32, y: i32, z: i32) -> (i32, i32) {
    cordic::<ROTATE, HYPERBOLIC>(x, y, z, None)
}

/// Returns `G*sqrt(x**2 - y**2), z + atanh2(y, x)`
pub fn sqrt_atanh2(x: i32, y: i32, z: i32) -> (i32, i32) {
    cordic::<DEROTATE, HYPERBOLIC>(x, y, z, None)
}

#[cfg(test)]
mod test {
    use core::f64::consts::PI;
    use log;
    use quickcheck_macros::quickcheck;
    use rand::{prelude::*, rngs::StdRng};

    use super::*;

    const Q31: f64 = (1i64 << 31) as _;

    fn f2i(x: f64) -> i32 {
        (x * Q31).round() as i64 as _
    }
    fn i2f(x: i32) -> f64 {
        (x as f64 / Q31) as _
    }

    const F: f64 = CORDIC_CIRCULAR_GAIN.recip();
    const G: f64 = CORDIC_HYPERBOLIC_GAIN.recip();

    fn cos_sin_err(x: f64, y: f64, z: f64) -> f64 {
        let xy = cos_sin(f2i(x * F), f2i(y * F), f2i(z));
        let xy = (i2f(xy.0), i2f(xy.1));
        let (s, c) = (z * PI).sin_cos();
        let xy0 = (c * x - s * y, s * x + c * y);
        let (dx, dy) = (xy.0 - xy0.0, xy.1 - xy0.1);
        let dr = (dx.powi(2) + dy.powi(2)).sqrt();
        let da = i2f(f2i(xy.1.atan2(xy.0) / PI - y.atan2(x) / PI - z));
        log::debug!("{dx:.10},{dy:.10} ~ {dr:.10}@{da:.10}");
        dr * Q31
    }

    fn sqrt_atan2_err(x: f64, y: f64) -> f64 {
        let (r, z) = sqrt_atan2(f2i(x * F), f2i(y * F), 0);
        let (r, z) = (i2f(r), i2f(z));
        let r0 = (x.powi(2) + y.powi(2)).sqrt();
        let z0 = y.atan2(x) / PI;
        let da = i2f(f2i(z - z0));
        let dr = ((r - r0).powi(2) + ((da * PI).sin() * r0).powi(2)).sqrt();
        log::debug!("{dr:.10}@{da:.10}");
        dr * Q31
    }

    fn sqrt_atanh2_err(x: f64, y: f64) -> f64 {
        let (r, z) = sqrt_atanh2(f2i(x * G), f2i(y * G), 0);
        let (r, z) = (i2f(r), i2f(z));
        let r0 = (x.powi(2) - y.powi(2)).sqrt();
        let z0 = (y / x).atanh();
        let da = i2f(f2i(z - z0));
        let dr = (r - r0).abs();
        log::debug!("{dr:.10}@{da:.10}");
        dr * Q31
    }

    #[test]
    fn basic_rot() {
        assert!((CORDIC_CIRCULAR_GAIN - 1.64676025812107).abs() < 1e-14,);

        cos_sin_err(0.50, 0.2, 0.123);
        cos_sin_err(0.01, 0.0, -0.35);
        cos_sin_err(0.605, 0.0, 0.35);
        cos_sin_err(-0.3, 0.4, 0.55);
        cos_sin_err(-0.3, -0.4, -0.55);
        cos_sin_err(-0.3, -0.4, 0.8);
        cos_sin_err(-0.3, -0.4, -0.8);
    }

    fn test_values(n: usize) -> Vec<i32> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut n: Vec<_> = core::iter::from_fn(|| Some(rng.random())).take(n).collect();
        n.extend([
            0,
            1,
            -1,
            0xf,
            -0xf,
            0x55555555,
            -0x55555555,
            0x5aaaaaaa,
            -0x5aaaaaaa,
            0x7fffffff,
            -0x7fffffff,
            1 << 29,
            -1 << 29,
            1 << 30,
            -1 << 30,
            i32::MIN,
            i32::MAX,
        ]);
        n
    }

    #[test]
    fn meanmax_rot() {
        let n = test_values(50);
        let mut mean = 0.0;
        let mut max: f64 = 0.0;
        for x in n.iter() {
            for y in n.iter() {
                for z in n.iter() {
                    let (x, y) = (i2f(*x), i2f(*y));
                    if 1.0 - x.powi(2) - y.powi(2) <= 1e-9 {
                        continue;
                    }
                    let dr = cos_sin_err(x, y, i2f(*z));
                    mean += dr;
                    max = max.max(dr);
                }
            }
        }
        mean /= n.len().pow(3) as f64;
        log::info!("{mean} {max}");
        assert!(mean < 5.0);
        assert!(max < 24.0);
    }

    #[test]
    fn meanmax_vect() {
        let n = test_values(300);
        let mut mean = 0.0;
        let mut max: f64 = 0.0;
        for x in n.iter() {
            for y in n.iter() {
                let (x, y) = (i2f(*x), i2f(*y));
                if 1.0 - x.powi(2) - y.powi(2) <= 1e-9 {
                    continue;
                }
                let dr = sqrt_atan2_err(x, y);
                mean += dr;
                max = max.max(dr);
            }
        }
        mean /= n.len().pow(2) as f64;
        log::info!("{mean} {max}");
        assert!(mean < 8.0);
        assert!(max < 30.0);
    }

    #[quickcheck]
    fn check_rot(x: i32, y: i32, z: i32) -> bool {
        let (x, y) = (i2f(x), i2f(y));
        if CORDIC_CIRCULAR_GAIN.powi(2) * (x.powi(2) + y.powi(2)) >= 1.0 {
            return true;
        }
        cos_sin_err(x, y, i2f(z)) <= 22.0
    }

    #[quickcheck]
    fn check_vect(x: i32, y: i32) -> bool {
        let (x, y) = (i2f(x), i2f(y));
        if CORDIC_CIRCULAR_GAIN.powi(2) * (x.powi(2) + y.powi(2)) >= 1.0 {
            return true;
        }
        sqrt_atan2_err(x, y) <= 29.0
    }

    #[quickcheck]
    fn check_hyp_vect(x: i32, y: i32) -> bool {
        let (x, y) = (i2f(x), i2f(y));
        if CORDIC_HYPERBOLIC_GAIN.powi(2) * (x.powi(2) - y.powi(2)) >= 1.0 {
            return true;
        }
        if y.abs() > x.abs() || (y / x).atanh() >= 1.1 {
            return true;
        }

        sqrt_atanh2_err(x, y);
        true
    }

    #[test]
    fn test_atanh() {
        sqrt_atanh2_err(0.8, 0.1);
        sqrt_atanh2_err(0.8, 0.1);
        sqrt_atanh2_err(0.8, 0.1);
        sqrt_atanh2_err(0.99, 0.0);
    }
}
