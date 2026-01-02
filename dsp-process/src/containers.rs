use crate::{Inplace, Intermediate, Minor, Parallel, Process, Transpose, Unsplit};

//////////// UNSPLIT ////////////

impl<X: Copy, Y, P: Process<X, Y>> Process<X, Y> for Unsplit<P> {
    fn process(&mut self, x: X) -> Y {
        self.0.process(x)
    }
}

impl<X: Copy, P: Inplace<X>> Inplace<X> for Unsplit<P> {
    fn inplace(&mut self, xy: &mut [X]) {
        self.0.inplace(xy)
    }
}

//////////// MAJOR ////////////

/// Type sequence must be X->Y->Y
impl<X: Copy, Y: Copy, P0: Process<X, Y>, P1: Inplace<Y>> Process<X, Y> for (P0, P1) {
    fn process(&mut self, x: X) -> Y {
        self.1.process(self.0.process(x))
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        self.0.block(x, y);
        self.1.inplace(y);
    }
}

impl<X: Copy, P0: Inplace<X>, P1: Inplace<X>> Inplace<X> for (P0, P1) {
    fn inplace(&mut self, xy: &mut [X]) {
        self.0.inplace(xy);
        self.1.inplace(xy);
    }
}

/// Slices can be zero length but must have input and output types the same
impl<X: Copy, P: Inplace<X>> Process<X> for [P] {
    fn process(&mut self, x: X) -> X {
        self.iter_mut().fold(x, |x, p| p.process(x))
    }

    fn block(&mut self, x: &[X], y: &mut [X]) {
        if let Some((p0, p)) = self.split_first_mut() {
            p0.block(x, y);
            for p in p.iter_mut() {
                p.inplace(y);
            }
        } else {
            y.copy_from_slice(x);
        }
    }
}

impl<X: Copy, P: Inplace<X>> Inplace<X> for [P] {
    fn inplace(&mut self, xy: &mut [X]) {
        for p in self.iter_mut() {
            p.inplace(xy)
        }
    }
}

/// Arrays must be non-empty but that first item can transform X->Y
impl<X: Copy, Y: Copy, P: Process<X, Y> + Inplace<Y>, const N: usize> Process<X, Y> for [P; N] {
    fn process(&mut self, x: X) -> Y {
        const { assert!(N > 0) }
        let (p0, p) = self.split_first_mut().unwrap();
        p.iter_mut().fold(p0.process(x), |x, p| p.process(x))
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        const { assert!(N > 0) }
        let (p0, p) = self.split_first_mut().unwrap();
        p0.block(x, y);
        for p in p.iter_mut() {
            p.inplace(y);
        }
    }
}

impl<X: Copy, P: Inplace<X>, const N: usize> Inplace<X> for [P; N] {
    fn inplace(&mut self, xy: &mut [X]) {
        self.as_mut().inplace(xy)
    }
}

//////////// MINOR ////////////

/// X->U->Y
impl<X: Copy, U: Copy, Y, P0: Process<X, U>, P1: Process<U, Y>> Process<X, Y>
    for Minor<(P0, P1), U>
{
    fn process(&mut self, x: X) -> Y {
        self.inner.1.process(self.inner.0.process(x))
    }
}

/// X->X->X...
impl<X: Copy, P: Process<X>> Process<X> for Minor<[P], X> {
    fn process(&mut self, x: X) -> X {
        self.inner.iter_mut().fold(x, |x, p| p.process(x))
    }
}

/// X->Y->Y...
impl<X: Copy, Y: Copy, P: Process<X, Y> + Process<Y>, const N: usize> Process<X, Y>
    for Minor<[P; N], Y>
{
    fn process(&mut self, x: X) -> Y {
        const { assert!(N > 0) }
        let (p0, p) = self.inner.split_first_mut().unwrap();
        p.iter_mut().fold(p0.process(x), |x, p| p.process(x))
    }
}

impl<X: Copy, U, P> Inplace<X> for Minor<P, U> where Self: Process<X> {}

//////////// PARALLEL ////////////

impl<X0: Copy, X1: Copy, Y0, Y1, P0: Process<X0, Y0>, P1: Process<X1, Y1>>
    Process<(X0, X1), (Y0, Y1)> for Parallel<(P0, P1)>
{
    fn process(&mut self, x: (X0, X1)) -> (Y0, Y1) {
        (self.0.0.process(x.0), self.0.1.process(x.1))
    }
}

impl<X: Copy, Y, P0: Process<X, Y>, P1: Process<X, Y>> Process<[X; 2], [Y; 2]>
    for Parallel<(P0, P1)>
{
    fn process(&mut self, x: [X; 2]) -> [Y; 2] {
        [self.0.0.process(x[0]), self.0.1.process(x[1])]
    }
}

impl<X: Copy, Y: Copy + Default, P: Process<X, Y>, const N: usize> Process<[X; N], [Y; N]>
    for Parallel<[P; N]>
{
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        let mut y = [Y::default(); N];
        for (c, (x, y)) in self.0.iter_mut().zip(x.into_iter().zip(y.iter_mut())) {
            *y = c.process(x);
        }
        y
    }
}

impl<X: Copy, P> Inplace<X> for Parallel<P> where Self: Process<X> {}

//////////// TRANSPOSE ////////////

impl<X: Copy, Y, C0: Process<X, Y>, C1: Process<X, Y>> Process<[X; 2], [Y; 2]>
    for Transpose<(C0, C1)>
{
    fn process(&mut self, x: [X; 2]) -> [Y; 2] {
        [self.0.0.process(x[0]), self.0.1.process(x[1])]
    }

    fn block(&mut self, x: &[[X; 2]], y: &mut [[Y; 2]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        let (x0, x1) = x.as_flattened().split_at(n);
        let (y0, y1) = y.as_flattened_mut().split_at_mut(n);
        self.0.0.block(x0, y0);
        self.0.1.block(x1, y1);
    }
}

impl<X: Copy, C0, C1> Inplace<[X; 2]> for Transpose<(C0, C1)>
where
    C0: Inplace<X>,
    C1: Inplace<X>,
{
    fn inplace(&mut self, xy: &mut [[X; 2]]) {
        let n = xy.len();
        let (xy0, xy1) = xy.as_flattened_mut().split_at_mut(n);
        self.0.0.inplace(xy0);
        self.0.1.inplace(xy1);
    }
}

impl<X: Copy, Y: Copy + Default, C: Process<X, Y>, const N: usize> Process<[X; N], [Y; N]>
    for Transpose<[C; N]>
{
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        let mut y = [Y::default(); N];
        for (c, (x, y)) in self.0.iter_mut().zip(x.into_iter().zip(y.iter_mut())) {
            *y = c.process(x);
        }
        y
    }

    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; N]]) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        for (c, (x, y)) in self.0.iter_mut().zip(
            x.as_flattened()
                .chunks_exact(n)
                .zip(y.as_flattened_mut().chunks_exact_mut(n)),
        ) {
            c.block(x, y)
        }
    }
}

impl<X: Copy + Default, C: Inplace<X>, const N: usize> Inplace<[X; N]> for Transpose<[C; N]> {
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        let n = xy.len();
        for (c, xy) in self
            .0
            .iter_mut()
            .zip(xy.as_flattened_mut().chunks_exact_mut(n))
        {
            c.inplace(xy)
        }
    }
}

//////////// INTERMEDIATE ////////////

impl<X: Copy, U: Copy + Default, Y, P1: Process<X, U>, P2: Process<U, Y>, const N: usize>
    Process<X, Y> for Intermediate<(P1, P2), [U; N]>
{
    fn process(&mut self, x: X) -> Y {
        self.inner.1.process(self.inner.0.process(x))
    }

    fn block(&mut self, x: &[X], y: &mut [Y]) {
        let mut u = [U::default(); N];
        let (x, xr) = x.as_chunks::<N>();
        let (y, yr) = y.as_chunks_mut::<N>();
        for (x, y) in x.iter().zip(y) {
            self.inner.0.block(x, &mut u);
            self.inner.1.block(&u, y);
        }
        debug_assert_eq!(xr.len(), yr.len());
        self.inner.0.block(xr, &mut u[..xr.len()]);
        self.inner.1.block(&u[..xr.len()], yr);
    }
}

impl<X: Copy, U: Copy + Default, P1: Process<X, U>, P2: Process<U, X>, const N: usize> Inplace<X>
    for Intermediate<(P1, P2), [U; N]>
{
    fn inplace(&mut self, xy: &mut [X]) {
        let mut u = [U::default(); N];
        let (xy, xyr) = xy.as_chunks_mut::<N>();
        for xy in xy {
            self.inner.0.block(xy, &mut u);
            self.inner.1.block(&u, xy);
        }
        self.inner.0.block(xyr, &mut u[..xyr.len()]);
        self.inner.1.block(&u[..xyr.len()], xyr);
    }
}
