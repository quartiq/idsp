#[pyo3::pymodule(name = "_idsp")]
mod _idsp {
    use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
    use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
    use pyo3::prelude::*;

    #[pyfunction]
    fn wdf<'py>(x: &Bound<'py, PyArrayDyn<i32>>) -> PyResult<()> {
        use crate::iir::{Process, StatefulRef, Wdf2, Wdf2State};

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
        let x = unsafe { x.as_slice_mut() };
        StatefulRef(&f, &mut s).process_in_place(x.unwrap());
        Ok(())
    }
}
