#[pyo3::pymodule]
mod _idsp {
    use crate::iir::{Process, Sos, State as SosState, StatefulRef, Wdf2, Wdf2State};
    use numpy::{PyReadonlyArray2, PyReadwriteArray1};
    use pyo3::{exceptions::PyTypeError, prelude::*};

    /// Quantize a (N, 6) array of second order section coefficients to Q29 and filter an i32 array with it in place.
    #[pyfunction]
    fn sos<'py>(
        sos: PyReadonlyArray2<'py, f64>,
        mut xy: PyReadwriteArray1<'py, i32>,
    ) -> PyResult<()> {
        let sos = sos
            .as_array()
            .outer_iter()
            .map(|sos| {
                let sos: &[_; 3 * 2] = sos
                    .as_slice()
                    .ok_or(PyTypeError::new_err("order"))?
                    .try_into()
                    .or(Err(PyTypeError::new_err("shape")))?;
                let sos: &[[_; 3]; 2] = bytemuck::cast_ref(sos);
                Ok(Sos::<29>::from(sos))
            })
            .collect::<Result<Vec<_>, PyErr>>()?;
        let xy = xy.as_slice_mut().unwrap();
        let mut state = vec![SosState::default(); sos.len()];
        for (sos, state) in sos.iter().zip(state.iter_mut()) {
            StatefulRef(sos, state).process_in_place(xy);
        }
        Ok(())
    }

    /// Filter an i32 array in place with a 19th order half band WDF.
    ///
    /// Gazsi 1985, Example 5
    #[pyfunction]
    fn wdf<'py>(mut xy: PyReadwriteArray1<'py, i32>) -> PyResult<()> {
        struct Nineteen {
            a: (Wdf2<0x1c>, Wdf2<0x1d>, Wdf2<0x1d>, Wdf2<0x1d>, Wdf2<0x01>),
            b: (Wdf2<0x1c>, Wdf2<0x1c>, Wdf2<0x1d>, Wdf2<0x1d>, Wdf2<0x1d>),
        }

        impl Process for StatefulRef<'_, Nineteen, [Wdf2State; 10]> {
            fn process(&mut self, x0: i32) -> i32 {
                let mut xa = StatefulRef(&self.0.a.0, &mut self.1[0]).process(x0);
                xa = StatefulRef(&self.0.a.1, &mut self.1[1]).process(xa);
                xa = StatefulRef(&self.0.a.2, &mut self.1[2]).process(xa);
                xa = StatefulRef(&self.0.a.3, &mut self.1[3]).process(xa);
                xa = StatefulRef(&self.0.a.4, &mut self.1[4]).process(xa);
                let mut xb = StatefulRef(&self.0.b.0, &mut self.1[5]).process(x0);
                xb = StatefulRef(&self.0.b.1, &mut self.1[6]).process(xb);
                xb = StatefulRef(&self.0.b.2, &mut self.1[7]).process(xb);
                xb = StatefulRef(&self.0.b.3, &mut self.1[8]).process(xb);
                xb = StatefulRef(&self.0.b.4, &mut self.1[9]).process(xb);
                xa + xb
            }
        }

        let f = Nineteen {
            a: (
                Wdf2::quantize([-0.226119, 0.0]).unwrap(),
                Wdf2::quantize([-0.602422, 0.0]).unwrap(),
                Wdf2::quantize([-0.839323, 0.0]).unwrap(),
                Wdf2::quantize([-0.950847, 0.0]).unwrap(),
                Wdf2::default(),
            ),
            b: (
                Wdf2::quantize([-0.063978, 0.0]).unwrap(),
                Wdf2::quantize([-0.423068, 0.0]).unwrap(),
                Wdf2::quantize([-0.741327, 0.0]).unwrap(),
                Wdf2::quantize([-0.905567, 0.0]).unwrap(),
                Wdf2::quantize([-0.984721, 0.0]).unwrap(),
            ),
        };
        let mut s: [Wdf2State; 10] = Default::default();
        let x = xy.as_slice_mut().or(Err(PyTypeError::new_err("order")))?;
        StatefulRef(&f, &mut s).process_in_place(x);
        Ok(())
    }
}
