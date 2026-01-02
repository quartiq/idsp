use crate::{Inplace, Process, SplitInplace, SplitProcess};

/// Adapts an interpolator to output chunk mode
#[derive(Clone, Debug, Default)]
pub struct Interpolator<P>(pub P);
impl<X: Copy, Y, P: Process<Option<X>, Y>, const N: usize> Process<X, [Y; N]> for Interpolator<P> {
    fn process(&mut self, x: X) -> [Y; N] {
        core::array::from_fn(|i| self.0.process((i == 0).then_some(x)))
    }
}
impl<X: Copy, P> Inplace<X> for Interpolator<P> where Self: Process<X> {}
impl<X: Copy, Y, C: SplitProcess<Option<X>, Y, S>, S, const N: usize> SplitProcess<X, [Y; N], S>
    for Interpolator<C>
{
    fn process(&self, state: &mut S, x: X) -> [Y; N] {
        core::array::from_fn(|i| self.0.process(state, (i == 0).then_some(x)))
    }
}
impl<X: Copy, C, S> SplitInplace<X, S> for Interpolator<C> where Self: SplitProcess<X, X, S> {}

/// Adapts a decimator to input chunk mode
///
/// Synchronizes to the inner tick by discarding samples after tick.
/// Panics if tick does not match N
#[derive(Clone, Debug, Default)]
pub struct Decimator<P>(pub P);
impl<X: Copy, Y, P: Process<X, Option<Y>>, const N: usize> Process<[X; N], Y> for Decimator<P> {
    fn process(&mut self, x: [X; N]) -> Y {
        x.into_iter().find_map(|x| self.0.process(x)).unwrap()
    }
}
impl<X: Copy, P> Inplace<X> for Decimator<P> where Self: Process<X> {}
impl<X: Copy, Y, C: SplitProcess<X, Option<Y>, S>, S, const N: usize> SplitProcess<[X; N], Y, S>
    for Decimator<C>
{
    fn process(&self, state: &mut S, x: [X; N]) -> Y {
        x.into_iter()
            .find_map(|x| self.0.process(state, x))
            .unwrap()
    }
}
impl<X: Copy, C, S> SplitInplace<X, S> for Decimator<C> where Self: SplitProcess<X, X, S> {}

/// Chunked processing
///
/// Adapts a X->Y processor into a [X; N] -> [Y; N] processor
/// by flattening input and output.
#[derive(Clone, Debug, Default)]
pub struct Chunk<P>(pub P);
impl<P: Process<X, Y>, X: Copy, Y, const N: usize> Process<[X; N], [Y; N]> for Chunk<P> {
    fn process(&mut self, x: [X; N]) -> [Y; N] {
        x.map(|x| self.0.process(x))
    }

    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; N]]) {
        self.0.block(x.as_flattened(), y.as_flattened_mut())
    }
}
impl<P: Inplace<X>, X: Copy, const N: usize> Inplace<[X; N]> for Chunk<P> {
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        self.0.inplace(xy.as_flattened_mut())
    }
}
impl<C: SplitProcess<X, Y, S>, S, X: Copy, Y, const N: usize> SplitProcess<[X; N], [Y; N], S>
    for Chunk<C>
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; N] {
        x.map(|x| self.0.process(state, x))
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; N]]) {
        self.0.block(state, x.as_flattened(), y.as_flattened_mut())
    }
}
impl<C: SplitInplace<X, S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S> for Chunk<C> {
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        self.0.inplace(state, xy.as_flattened_mut())
    }
}

/// Adapts a [X; R] -> Y processor to [X; N=R*M]->[Y; M]
#[derive(Clone, Debug, Default)]
pub struct ChunkIn<P, const R: usize>(pub P);
impl<P: Process<[X; R], Y>, X: Copy, Y, const N: usize, const R: usize, const M: usize>
    Process<[X; N], [Y; M]> for ChunkIn<P, R>
{
    fn process(&mut self, x: [X; N]) -> [Y; M] {
        const { assert!(R * M == N) }
        let (x, []) = x.as_chunks() else {
            unreachable!()
        };
        core::array::from_fn(|i| self.0.process(x[i]))
    }

    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(R * M == N) }
        let (x, []) = x.as_flattened().as_chunks() else {
            unreachable!()
        };
        self.0.block(x, y.as_flattened_mut())
    }
}
impl<P: Inplace<[X; 1]>, X: Copy, const N: usize> Inplace<[X; N]> for ChunkIn<P, 1>
where
    Self: Process<[X; N]>,
{
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(xy)
    }
}
impl<C: SplitProcess<[X; R], Y, S>, S, X: Copy, Y, const N: usize, const R: usize, const M: usize>
    SplitProcess<[X; N], [Y; M], S> for ChunkIn<C, R>
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(R * M == N) }
        let (x, []) = x.as_chunks() else {
            unreachable!()
        };
        core::array::from_fn(|i| self.0.process(state, x[i]))
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(R * M == N) }
        let (x, []) = x.as_flattened().as_chunks() else {
            unreachable!()
        };
        self.0.block(state, x, y.as_flattened_mut())
    }
}
impl<C: SplitInplace<[X; 1], S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S>
    for ChunkIn<C, 1>
where
    Self: SplitProcess<[X; N], [X; N], S>,
{
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(state, xy)
    }
}

/// Adapts a X -> [Y; R] processor to [X; N]->[Y; M = R*N]
#[derive(Clone, Debug, Default)]
pub struct ChunkOut<P, const R: usize>(pub P);
impl<P: Process<X, [Y; R]>, X: Copy, Y, const N: usize, const R: usize, const M: usize>
    Process<[X; N], [Y; M]> for ChunkOut<P, R>
where
    [Y; M]: Default,
{
    fn process(&mut self, x: [X; N]) -> [Y; M] {
        const { assert!(R * N == M) }
        // TODO: bytemuck?
        // x.map(|x| self.0.process(x)).flatten()
        let mut y = <[Y; M]>::default();
        let (yy, []) = y.as_chunks_mut() else {
            unreachable!()
        };
        for (x, y) in x.into_iter().zip(yy) {
            *y = self.0.process(x);
        }
        y
    }

    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(R * N == M) }
        let (y, []) = y.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.block(x.as_flattened(), y)
    }
}
impl<P: Inplace<[X; 1]>, X: Copy, const N: usize> Inplace<[X; N]> for ChunkOut<P, 1>
where
    Self: Process<[X; N]>,
{
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(xy)
    }
}
impl<C: SplitProcess<X, [Y; R], S>, S, X: Copy, Y, const N: usize, const R: usize, const M: usize>
    SplitProcess<[X; N], [Y; M], S> for ChunkOut<C, R>
where
    [Y; M]: Default,
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(R * N == M) }
        // TODO: bytemuck?
        // x.map(|x| self.0.process(x)).flatten()
        let mut y = <[Y; M]>::default();
        let (yy, []) = y.as_chunks_mut() else {
            unreachable!()
        };
        for (x, y) in x.into_iter().zip(yy) {
            *y = self.0.process(state, x);
        }
        y
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(R * N == M) }
        let (y, []) = y.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.block(state, x.as_flattened(), y)
    }
}
impl<C: SplitInplace<[X; 1], S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S>
    for ChunkOut<C, 1>
where
    Self: SplitProcess<[X; N], [X; N], S>,
{
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(state, xy)
    }
}

/// Adapts a [X; Q] -> [Y; R] processor to [X; N = Q*_]->[Y; M = R*_]
#[derive(Clone, Debug, Default)]
pub struct ChunkInOut<P, const Q: usize, const R: usize>(pub P);
impl<P, X: Copy, Y, const Q: usize, const N: usize, const R: usize, const M: usize>
    Process<[X; N], [Y; M]> for ChunkInOut<P, Q, R>
where
    [Y; M]: Default,
    P: Process<[X; Q], [Y; R]>,
{
    fn process(&mut self, x: [X; N]) -> [Y; M] {
        const { assert!(N.is_multiple_of(Q)) }
        const { assert!(M.is_multiple_of(R)) }
        // TODO: bytemuck?
        let mut y = <[Y; M]>::default();
        let (yy, []) = y.as_chunks_mut() else {
            unreachable!()
        };
        let (x, []) = x.as_chunks() else {
            unreachable!()
        };
        for (x, y) in x.iter().zip(yy) {
            *y = self.0.process(*x);
        }
        y
    }

    fn block(&mut self, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(N.is_multiple_of(Q)) }
        const { assert!(M.is_multiple_of(R)) }
        let (x, []) = x.as_flattened().as_chunks() else {
            unreachable!()
        };
        let (y, []) = y.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.block(x, y)
    }
}
impl<P: Inplace<[X; 1]>, X: Copy, const N: usize> Inplace<[X; N]> for ChunkInOut<P, 1, 1>
where
    Self: Process<[X; N]>,
{
    fn inplace(&mut self, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(xy)
    }
}
impl<C, S, X: Copy, Y, const Q: usize, const N: usize, const R: usize, const M: usize>
    SplitProcess<[X; N], [Y; M], S> for ChunkInOut<C, Q, R>
where
    [Y; M]: Default,
    C: SplitProcess<[X; Q], [Y; R], S>,
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(N.is_multiple_of(Q)) }
        const { assert!(M.is_multiple_of(R)) }
        // TODO: bytemuck?
        let mut y = <[Y; M]>::default();
        let (yy, []) = y.as_chunks_mut() else {
            unreachable!()
        };
        let (x, []) = x.as_chunks() else {
            unreachable!()
        };
        for (x, y) in x.iter().zip(yy) {
            *y = self.0.process(state, *x);
        }
        y
    }

    fn block(&self, state: &mut S, x: &[[X; N]], y: &mut [[Y; M]]) {
        const { assert!(N.is_multiple_of(Q)) }
        const { assert!(M.is_multiple_of(R)) }
        let (x, []) = x.as_flattened().as_chunks() else {
            unreachable!()
        };
        let (y, []) = y.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.block(state, x, y)
    }
}
impl<C: SplitInplace<[X; 1], S>, S, X: Copy, const N: usize> SplitInplace<[X; N], S>
    for ChunkInOut<C, 1, 1>
where
    Self: SplitProcess<[X; N], [X; N], S>,
{
    fn inplace(&self, state: &mut S, xy: &mut [[X; N]]) {
        let (xy, []) = xy.as_flattened_mut().as_chunks_mut() else {
            unreachable!()
        };
        self.0.inplace(state, xy)
    }
}
