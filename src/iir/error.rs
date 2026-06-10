//! Parameter validation errors for IIR builders.

/// Builder parameter validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A required finite parameter was not finite.
    NonFinite(&'static str),
    /// A parameter must be strictly positive.
    NonPositive(&'static str),
    /// A normalized parameter was outside its allowed range.
    OutOfRange(&'static str),
    /// A min/max pair was inverted.
    InvertedRange(&'static str),
    /// A gain/limit sign pairing was inconsistent.
    SignMismatch(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite(name) => write!(f, "parameter `{name}` must be finite"),
            Self::NonPositive(name) => write!(f, "parameter `{name}` must be positive"),
            Self::OutOfRange(name) => write!(f, "parameter `{name}` is out of range"),
            Self::InvertedRange(name) => write!(f, "range `{name}` is inverted"),
            Self::SignMismatch(name) => write!(f, "parameter `{name}` has incompatible sign"),
        }
    }
}

impl core::error::Error for Error {}
