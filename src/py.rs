#[pyo3::pymodule]
mod _idsp {
    use dsp_fixedpoint::Q32;
    use dsp_process::SplitInplace;
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
            .map(|s| {
                let s: &[_; 3 * 2] = s
                    .as_slice()
                    .ok_or(PyTypeError::new_err("order"))?
                    .try_into()
                    .or(Err(PyTypeError::new_err("shape")))?;
                Ok(crate::iir::Biquad::<Q32<29>>::from([
                    [s[0], s[1], s[2]],
                    [s[3], s[4], s[5]],
                ]))
            })
            .collect::<Result<Vec<_>, PyErr>>()?;
        let mut state = vec![crate::iir::DirectForm1::default(); sos.len()];
        let xy = xy.as_slice_mut().or(Err(PyTypeError::new_err("order")))?;
        sos.inplace(&mut state, xy);
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
                let mut sos = crate::iir::BiquadClamp::<Q32<29>, i32>::from([
                    [s[0], s[1], s[2]],
                    [s[3], s[4], s[5]],
                ]);
                sos.u = s[6].round() as _;
                sos.min = s[7].round() as _;
                sos.max = s[8].round() as _;
                Ok(sos)
            })
            .collect::<Result<Vec<_>, PyErr>>()?;
        let mut state = vec![crate::iir::DirectForm1Wide::default(); sos.len()];
        let xy = xy.as_slice_mut().or(Err(PyTypeError::new_err("order")))?;
        sos.inplace(&mut state, xy);
        Ok(())
    }
}
