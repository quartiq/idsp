use super::tools::macc_i32;

/// Integer FIR filter
///
/// See `dsp::iir::IIR` for general implementation details.
/// Offset and limiting disabled to suit lowpass applications.
/// Coefficient scaling fixed and optimized.
#[derive(Copy, Clone, Default, Debug, StringSet, Deserialize)]
pub struct IIR {
    pub ba: Vec5,
    // pub y_offset: i32,
    // pub y_min: i32,
    // pub y_max: i32,
}

impl IIR {
    /// Coefficient fixed point format: signed Q2.30.
    /// Tailored to low-passes, PI, II etc.
    pub const SHIFT: u32 = 30;

    /// Feed a new input value into the filter, update the filter state, and
    /// return the new output. Only the state `xy` is modified.
    ///
    /// # Arguments
    /// * `xy` - Current filter state.
    /// * `x0` - New input.
    pub fn update(&self, xy: &mut Vec5, x0: i32) -> i32 {
        let n = self.ba.len();
        debug_assert!(xy.len() == n);
        // `xy` contains       x0 x1 y0 y1 y2
        // Increment time      x1 x2 y1 y2 y3
        // Shift               x1 x1 x2 y1 y2
        // This unrolls better than xy.rotate_right(1)
        xy.copy_within(0..n - 1, 1);
        // Store x0            x0 x1 x2 y1 y2
        xy[0] = x0;
        // Compute y0 by multiply-accumulate
        let y0 = macc(0, xy, &self.ba, IIR::SHIFT);
        // Limit y0
        // let y0 = y0.max(self.y_min).min(self.y_max);
        // Store y0            x0 x1 y0 y1 y2
        xy[n / 2] = y0;
        y0
    }
}

#[cfg(test)]
mod test {
    use super::{Coeff, Vec5};

    #[test]
    fn lowpass_gen() {
        let ba = Vec5::lowpass(1e-5, 1. / 2f64.sqrt(), 2.);
        println!("{:?}", ba);
    }
}
