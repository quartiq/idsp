//! Numeric conversion and division contracts against widened references.

use dsp_fixedpoint::{FromRatio, P8, P64, Q8, Q64};
use num_traits::{FromPrimitive, ToPrimitive};

fn conversions<const F: i8>() {
    let scale = 2f64.powi(F as i32);
    for raw in i8::MIN..=i8::MAX {
        let value = raw as f64 / scale;
        let q = Q8::<F>::from_bits(raw);
        assert_eq!(q.to_i64(), Some(value.floor() as i64));
        assert_eq!(q.to_u64(), (value >= 0.0).then_some(value.floor() as u64));
    }
    for raw in u8::MIN..=u8::MAX {
        let q = P8::<F>::from_bits(raw);
        assert_eq!(q.to_i64(), Some((raw as f64 / scale).trunc() as i64));
    }
    for integer in -1024..=1024i64 {
        let raw = (integer as f64 * scale).floor();
        let signed = (-128.0..128.0).contains(&raw).then_some(raw as i8);
        let unsigned = (0.0..256.0).contains(&raw).then_some(raw as u8);
        assert_eq!(Q8::<F>::from_i64(integer).map(|q| q.into_bits()), signed);
        assert_eq!(P8::<F>::from_i64(integer).map(|q| q.into_bits()), unsigned);
        if integer >= 0 {
            assert_eq!(
                Q8::<F>::from_u64(integer as u64).map(|q| q.into_bits()),
                signed
            );
            assert_eq!(
                P8::<F>::from_u64(integer as u64).map(|q| q.into_bits()),
                unsigned
            );
        }
        // Exercise rounding ties and range boundaries in the floating paths.
        let value = integer as f64 / (2.0 * scale);
        let raw = (value * scale).round();
        let expected = (-128.0..128.0).contains(&raw).then_some(raw as i8);
        assert_eq!(
            <Q8<F> as FromPrimitive>::from_f64(value).map(|q| q.into_bits()),
            expected
        );
        assert_eq!(
            <Q8<F> as FromPrimitive>::from_f32(value as f32).map(|q| q.into_bits()),
            expected
        );
        let expected = (0.0..256.0).contains(&raw).then_some(raw as u8);
        assert_eq!(
            <P8<F> as FromPrimitive>::from_f64(value).map(|q| q.into_bits()),
            expected
        );
    }
}

fn ratios<const F: i8>() {
    for numerator in i8::MIN..=i8::MAX {
        for denominator in i8::MIN..=i8::MAX {
            if denominator == 0 {
                continue;
            }
            let scaled = if F >= 0 {
                ((numerator as i32) << F) / denominator as i32
            } else {
                numerator as i32 / ((denominator as i32) << -F)
            };
            if let Ok(expected) = i8::try_from(scaled) {
                assert_eq!(
                    Q8::<F>::from_ratio(numerator, denominator).into_bits(),
                    expected
                );
                let mut mixed = Q8::<3>::from_bits(numerator);
                mixed /= Q8::<F>::from_bits(denominator);
                assert_eq!(mixed.into_bits(), expected);
                assert_eq!(numerator / Q8::<F>::from_bits(denominator), expected);
                assert_eq!(
                    (Q8::<F>::from_bits(numerator) / Q8::<F>::from_bits(denominator)).into_bits(),
                    expected
                );
            }
        }
    }
    for numerator in u8::MIN..=u8::MAX {
        for denominator in 1..=u8::MAX {
            let scaled = if F >= 0 {
                ((numerator as u32) << F) / denominator as u32
            } else {
                numerator as u32 / ((denominator as u32) << -F)
            };
            if let Ok(expected) = u8::try_from(scaled) {
                assert_eq!(
                    P8::<F>::from_ratio(numerator, denominator).into_bits(),
                    expected
                );
            }
        }
    }
}

#[test]
fn small_width_reference() {
    conversions::<-7>();
    conversions::<-2>();
    conversions::<0>();
    conversions::<4>();
    conversions::<7>();
    conversions::<9>();
    ratios::<-7>();
    ratios::<-2>();
    ratios::<0>();
    ratios::<4>();
    ratios::<7>();
}

#[test]
fn wide_and_nonfinite_boundaries() {
    assert_eq!(Q8::<4>::from_i64(8), None);
    assert_eq!(Q8::<-2>::from_i64(256).unwrap().into_bits(), 64);
    assert_eq!(Q8::<-2>::from_bits(100).to_i64(), Some(400));
    assert_eq!(
        Q64::<0>::from_i128(i64::MIN as i128).unwrap().into_bits(),
        i64::MIN
    );
    assert_eq!(Q64::<0>::from_i128(i64::MAX as i128 + 1), None);
    assert_eq!(
        P64::<0>::from_u128(u64::MAX as u128).unwrap().into_bits(),
        u64::MAX
    );
    assert_eq!(P64::<0>::from_u128(u64::MAX as u128 + 1), None);
    assert_eq!(Q64::<-64>::from_bits(-1).to_i128(), Some(-(1i128 << 64)));
    assert_eq!(Q64::<-64>::from_bits(1).to_i64(), None);
    assert_eq!(
        P64::<-64>::from_bits(u64::MAX).to_u128(),
        Some((u64::MAX as u128) << 64)
    );
    assert_eq!(Q64::<-127>::from_bits(-1).to_i128(), Some(i128::MIN));
    assert_eq!(Q64::<-127>::from_bits(1).to_i128(), None);
    assert_eq!(Q64::<127>::from_i128(1), None);
    assert_eq!(Q64::<127>::from_i128(0).unwrap().into_bits(), 0);
    assert_eq!(Q64::<-127>::from_i128(i128::MIN).unwrap().into_bits(), -1);
    assert_eq!(P64::<-127>::from_u128(u128::MAX).unwrap().into_bits(), 1);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        assert!(<Q8<4> as FromPrimitive>::from_f64(value).is_none());
        assert!(<P8<4> as FromPrimitive>::from_f64(value).is_none());
    }
    assert!(<Q64<0> as FromPrimitive>::from_f64(2f64.powi(63)).is_none());
    assert!(<P64<0> as FromPrimitive>::from_f64(2f64.powi(64)).is_none());
}
