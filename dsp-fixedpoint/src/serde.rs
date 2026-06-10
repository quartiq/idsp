//! Serde adapters for fixed-point wire formats.
//!
//! The direct `Q` serde implementation is transparent and uses the raw
//! representation. Use [`as_f32`] or [`as_f64`] when the wire format should
//! carry the scaled value through serde's floating-point data model instead.

use ::serde::{Deserialize, Deserializer, Serializer};
use num_traits::{AsPrimitive, ToPrimitive};

use crate::Q;

/// Serialize and lossy-deserialize `Q` through serde's `f32` data model.
///
/// This maps `Q::<i32, i64, 3>::new(1)` to the JSON number `0.125` and back.
pub mod as_f32 {
    use super::*;

    /// Serialize as a scaled `f32` value.
    pub fn serialize<S, T, A, const F: i8>(q: &Q<T, A, F>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        Q<T, A, F>: ToPrimitive,
    {
        serializer.serialize_f32(
            q.to_f32()
                .ok_or_else(|| ::serde::ser::Error::custom("fixed-point value is not finite"))?,
        )
    }

    /// Deserialize from a scaled `f32` value.
    pub fn deserialize<'de, D, T, A, const F: i8>(deserializer: D) -> Result<Q<T, A, F>, D::Error>
    where
        D: Deserializer<'de>,
        f32: AsPrimitive<Q<T, A, F>>,
        Q<T, A, F>: Copy + 'static,
    {
        f32::deserialize(deserializer).map(Q::from_f32)
    }
}

/// Serialize and lossy-deserialize `Q` through serde's `f64` data model.
///
/// This maps `Q::<i32, i64, 3>::new(1)` to the JSON number `0.125` and back.
pub mod as_f64 {
    use super::*;

    /// Serialize as a scaled `f64` value.
    pub fn serialize<S, T, A, const F: i8>(q: &Q<T, A, F>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        Q<T, A, F>: ToPrimitive,
    {
        serializer.serialize_f64(
            q.to_f64()
                .ok_or_else(|| ::serde::ser::Error::custom("fixed-point value is not finite"))?,
        )
    }

    /// Deserialize from a scaled `f64` value.
    pub fn deserialize<'de, D, T, A, const F: i8>(deserializer: D) -> Result<Q<T, A, F>, D::Error>
    where
        D: Deserializer<'de>,
        f64: AsPrimitive<Q<T, A, F>>,
        Q<T, A, F>: Copy + 'static,
    {
        f64::deserialize(deserializer).map(Q::from_f64)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ::serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, PartialEq, Serialize)]
    struct F32 {
        #[serde(with = "as_f32")]
        value: crate::Q<i32, i64, 3>,
    }

    #[derive(Debug, Deserialize, PartialEq, Serialize)]
    struct F64 {
        #[serde(with = "as_f64")]
        value: crate::Q<i32, i64, 3>,
    }

    #[test]
    fn as_f32_uses_scaled_value() {
        let value = F32 {
            value: crate::Q::new(1),
        };
        let mut out = [0; 32];
        let len = serde_json_core::to_slice(&value, &mut out).unwrap();
        let json = &out[..len];
        assert_eq!(json, br#"{"value":0.125}"#);
        assert_eq!(serde_json_core::from_slice::<F32>(json).unwrap().0, value);
    }

    #[test]
    fn as_f64_uses_scaled_value() {
        let value = F64 {
            value: crate::Q::new(1),
        };
        let mut out = [0; 32];
        let len = serde_json_core::to_slice(&value, &mut out).unwrap();
        let json = &out[..len];
        assert_eq!(json, br#"{"value":0.125}"#);
        assert_eq!(serde_json_core::from_slice::<F64>(json).unwrap().0, value);
    }
}
