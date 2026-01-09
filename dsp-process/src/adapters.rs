use crate::{SplitInplace, SplitProcess};

/// Adapts an interpolator to output chunk mode
///
/// The inner processor is called with `Some(x)` once and then `None` `N-1` times
#[derive(Clone, Debug, Default)]
pub struct Interpolator<P>(pub P);
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
/// Panics if tick does not match `N`.
#[derive(Clone, Debug, Default)]
pub struct Decimator<P>(pub P);
impl<X: Copy, Y, C: SplitProcess<X, Option<Y>, S>, S, const N: usize> SplitProcess<[X; N], Y, S>
    for Decimator<C>
{
    fn process(&self, state: &mut S, x: [X; N]) -> Y {
        const { assert!(N > 0) }
        x.into_iter()
            .find_map(|x| self.0.process(state, x))
            .unwrap()
    }
}
impl<X: Copy, C, S> SplitInplace<X, S> for Decimator<C> where Self: SplitProcess<X, X, S> {}

/// Map `Option` and `Result`
#[derive(Clone, Debug, Default)]
pub struct Map<P>(pub P);
impl<X: Copy, Y, C: SplitProcess<X, Y, S>, S> SplitProcess<Option<X>, Option<Y>, S> for Map<C> {
    fn process(&self, state: &mut S, x: Option<X>) -> Option<Y> {
        x.map(|x| self.0.process(state, x))
    }
}
impl<X: Copy, Y, C: SplitProcess<X, Y, S>, S, E: Copy> SplitProcess<Result<X, E>, Result<Y, E>, S>
    for Map<C>
{
    fn process(&self, state: &mut S, x: Result<X, E>) -> Result<Y, E> {
        x.map(|x| self.0.process(state, x))
    }
}
impl<X: Copy, C: SplitInplace<X, S>, S> SplitInplace<X, S> for Map<C> where
    Self: SplitProcess<X, X, S>
{
}

/// Chunked processing
///
/// Adapt` a `X->Y` processor into a `[X; N] -> [Y; N]` processor
/// by flattening input and output.
#[derive(Debug, Copy, Clone, Default)]
pub struct Chunk<P>(pub P);
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

/// Chunked input
///
/// Adapt a `[X; R] -> Y` processor to `[X; N=R*M]->[Y; M]` for any `M`
/// by flattening and re-chunking input.
#[derive(Debug, Copy, Clone, Default)]
pub struct ChunkIn<P, const R: usize>(pub P);
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

/// Chunked output
///
/// Adapt a `X -> [Y; R]` processor to `[X; N]->[Y; M = R*N]` for any `N`
/// by flattening and re-chunking output.
#[derive(Debug, Copy, Clone, Default)]
pub struct ChunkOut<P, const R: usize>(pub P);
impl<C, S, X: Copy, Y: Default + Copy, const N: usize, const R: usize, const M: usize>
    SplitProcess<[X; N], [Y; M], S> for ChunkOut<C, R>
where
    C: SplitProcess<X, [Y; R], S>,
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(R * N == M) }
        // TODO: bytemuck y
        // x.map(|x| self.0.process(x)).as_flattened().as_array()
        let mut y = [Y::default(); M];
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

/// Chunked input and output
///
/// Adapt a `[X; Q] -> [Y; R]` processor to `[X; N = Q*I]->[Y; M = R*I]` for any `I`
/// by flattening and re-chunking input and output.
#[derive(Debug, Copy, Clone, Default)]
pub struct ChunkInOut<P, const Q: usize, const R: usize>(pub P);
impl<
    C,
    S,
    X: Copy,
    Y: Default + Copy,
    const Q: usize,
    const N: usize,
    const R: usize,
    const M: usize,
> SplitProcess<[X; N], [Y; M], S> for ChunkInOut<C, Q, R>
where
    C: SplitProcess<[X; Q], [Y; R], S>,
{
    fn process(&self, state: &mut S, x: [X; N]) -> [Y; M] {
        const { assert!(N.is_multiple_of(Q)) }
        const { assert!(M.is_multiple_of(R)) }
        // TODO: bytemuck y
        let mut y = [Y::default(); M];
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
