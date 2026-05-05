//! Parameter validation errors for IIR builders.

/// Builder parameter validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A required finite parameter was not finite.
    #[error("parameter `{0}` must be finite")]
    NonFinite(&'static str),
    /// A parameter must be strictly positive.
    #[error("parameter `{0}` must be positive")]
    NonPositive(&'static str),
    /// A normalized parameter was outside its allowed range.
    #[error("parameter `{0}` is out of range")]
    OutOfRange(&'static str),
    /// A min/max pair was inverted.
    #[error("range `{0}` is inverted")]
    InvertedRange(&'static str),
    /// A gain/limit sign pairing was inconsistent.
    #[error("parameter `{0}` has incompatible sign")]
    SignMismatch(&'static str),
}
