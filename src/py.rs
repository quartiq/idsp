#[pyo3::pymodule]
mod _idsp {
    use crate::iir::{Process, StatefulRef, Wdf2, Wdf2State};
    use numpy::{
        PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    };
    use pyo3::{exceptions::PyTypeError, prelude::*};

    /// Cosine and sine of a phase
    #[pyfunction]
    fn cossin<'py>(
        py: Python<'py>,
        p: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let p = p.as_slice().or(Err(PyTypeError::new_err("order")))?;
        let xy = PyArray2::zeros(py, [p.len(), 2], false);
        for (p, xy) in p.iter().zip(
            xy.readwrite()
                .as_slice_mut()
                .or(Err(PyTypeError::new_err("order")))?
                .as_chunks_mut::<2>()
                .0,
        ) {
            *xy = crate::cossin(*p).into();
        }
        Ok(xy)
    }

    /// atan2(y, x) of a [[x, y]] array
    #[pyfunction]
    fn atan2<'py>(
        py: Python<'py>,
        xy: PyReadonlyArray2<'py, i32>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let xy = xy.as_slice().or(Err(PyTypeError::new_err("order")))?;
        let p = PyArray1::zeros(py, [xy.len() / 2], false);
        for (xy, p) in xy.as_chunks::<2>().0.iter().zip(
            p.readwrite()
                .as_slice_mut()
                .or(Err(PyTypeError::new_err("order")))?,
        ) {
            *p = crate::atan2(xy[1], xy[0]);
        }
        Ok(p)
    }

    /// Quantize a (N, 6) array of second order section coefficients to Q29
    /// and filter an array with it in place.
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
                Ok(crate::iir::Sos::<29>::from(sos))
            })
            .collect::<Result<Vec<_>, PyErr>>()?;
        let xy = xy.as_slice_mut().unwrap();
        let mut state = vec![crate::iir::State::default(); sos.len()];
        for (sos, state) in sos.iter().zip(state.iter_mut()) {
            StatefulRef(sos, state).process_in_place(xy);
        }
        Ok(())
    }

    /// Quantize a (N, 6) array of second order section coefficients to Q29,
    /// add offset/min/max,
    /// and filter an array with it
    /// on a wide error feedback state in place.
    #[pyfunction]
    fn sos_clamp_wide<'py>(
        sos: PyReadonlyArray2<'py, f64>,
        mut xy: PyReadwriteArray1<'py, i32>,
    ) -> PyResult<()> {
        let sos = sos
            .as_array()
            .outer_iter()
            .map(|s| {
                let s: &[_; 2 * 3 + 3] = s
                    .as_slice()
                    .ok_or(PyTypeError::new_err("order"))?
                    .try_into()
                    .or(Err(PyTypeError::new_err("shape")))?;
                let mut sos =
                    crate::iir::SosClamp::<29>::from(&[[s[0], s[1], s[2]], [s[3], s[4], s[5]]]);
                sos.u = s[6] as _;
                sos.min = s[7] as _;
                sos.max = s[8] as _;
                Ok(sos)
            })
            .collect::<Result<Vec<_>, PyErr>>()?;
        let xy = xy.as_slice_mut().unwrap();
        let mut state = vec![crate::iir::StateWide::default(); sos.len()];
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
