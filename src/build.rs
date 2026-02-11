/// Build a value using a context
///
/// This is similar to the `Into` trait but allows lossy
/// (rounding, clamping, quantization) and non-value-preserving conversions.
/// The only semantic constraint is that the conversion is obvious and unambiguous.
pub trait Build<C> {
    /// The context of the convesion.
    ///
    /// E.g. units.
    type Context;

    /// Perform the conversion
    fn build(&self, ctx: &Self::Context) -> C;
}
