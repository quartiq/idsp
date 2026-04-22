use core::marker::PhantomData;

use crate::{Split, SplitInplace, SplitProcess};

/// Frame-major block layout marker.
///
/// A frame-major block corresponds to the ordinary `[[T; C]]` interpretation:
/// frames are contiguous and each frame holds `C` channel values.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameMajor;

/// Channel-major block layout marker.
///
/// A channel-major block stores `C` contiguous channel slices of equal length.
/// This is useful for multi-channel processing where each channel is processed
/// in a long contiguous run.
#[derive(Clone, Copy, Debug, Default)]
pub struct ChannelMajor;

/// Immutable typed view of a DSP block.
///
/// `L` describes how the flat storage should be interpreted. The view never
/// allocates or physically transposes memory.
#[derive(Clone, Copy, Debug)]
pub struct Block<'a, T, L, const C: usize> {
    flat: &'a [T],
    frames: usize,
    _layout: PhantomData<L>,
}

/// Mutable typed view of a DSP block.
#[derive(Debug)]
pub struct BlockMut<'a, T, L, const C: usize> {
    flat: &'a mut [T],
    frames: usize,
    _layout: PhantomData<L>,
}

/// Apply a chunk-based processor frame by frame to a frame-major block.
///
/// This is the bridge between chunk semantics and typed block views:
/// `SplitProcess<[X; Q], [Y; R], S>` becomes block processing from
/// `Block<FrameMajor, Q>` to `BlockMut<FrameMajor, R>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Frames<C>(pub C);

impl<'a, T, L, const C: usize> Block<'a, T, L, C> {
    /// Flat underlying storage.
    #[must_use]
    pub fn flat(self) -> &'a [T] {
        self.flat
    }

    /// Number of frames in the block.
    #[must_use]
    pub const fn frames(self) -> usize {
        self.frames
    }

    /// Reinterpret the same backing storage under another layout marker.
    #[must_use]
    pub fn as_layout<M>(self) -> Block<'a, T, M, C> {
        Block {
            flat: self.flat,
            frames: self.frames,
            _layout: PhantomData,
        }
    }
}

impl<'a, T, L, const C: usize> BlockMut<'a, T, L, C> {
    /// Flat underlying storage.
    #[must_use]
    pub fn flat(&self) -> &[T] {
        self.flat
    }

    /// Mutable flat underlying storage.
    #[must_use]
    pub fn flat_mut(&mut self) -> &mut [T] {
        self.flat
    }

    /// Number of frames in the block.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Reinterpret the same backing storage under another layout marker.
    #[must_use]
    pub fn as_layout<M>(self) -> BlockMut<'a, T, M, C> {
        let Self { flat, frames, .. } = self;
        BlockMut {
            flat,
            frames,
            _layout: PhantomData,
        }
    }
}

impl<'a, T, const C: usize> Block<'a, T, FrameMajor, C> {
    /// Borrow a conventional frame-major block.
    #[must_use]
    pub fn from_frames(frames: &'a [[T; C]]) -> Self {
        Self {
            flat: frames.as_flattened(),
            frames: frames.len(),
            _layout: PhantomData,
        }
    }

    /// Borrow the block as exact frames.
    #[must_use]
    pub fn as_frames(self) -> &'a [[T; C]] {
        let (frames, []) = self.flat.as_chunks::<C>() else {
            unreachable!()
        };
        frames
    }

    /// Borrow one frame.
    #[must_use]
    pub fn frame(self, i: usize) -> &'a [T; C] {
        &self.as_frames()[i]
    }
}

impl<'a, T, const C: usize> BlockMut<'a, T, FrameMajor, C> {
    /// Borrow a mutable conventional frame-major block.
    #[must_use]
    pub fn from_frames(frames: &'a mut [[T; C]]) -> Self {
        let frames_len = frames.len();
        Self {
            flat: frames.as_flattened_mut(),
            frames: frames_len,
            _layout: PhantomData,
        }
    }

    /// Borrow the block as exact frames.
    #[must_use]
    pub fn as_frames(&self) -> &[[T; C]] {
        let (frames, []) = self.flat.as_chunks::<C>() else {
            unreachable!()
        };
        frames
    }

    /// Borrow the block as exact mutable frames.
    #[must_use]
    pub fn as_frames_mut(&mut self) -> &mut [[T; C]] {
        let (frames, []) = self.flat.as_chunks_mut::<C>() else {
            unreachable!()
        };
        frames
    }

    /// Borrow one frame.
    #[must_use]
    pub fn frame(&self, i: usize) -> &[T; C] {
        &self.as_frames()[i]
    }

    /// Borrow one frame mutably.
    #[must_use]
    pub fn frame_mut(&mut self, i: usize) -> &mut [T; C] {
        &mut self.as_frames_mut()[i]
    }
}

impl<'a, T, const C: usize> Block<'a, T, ChannelMajor, C> {
    /// Borrow a channel-major flat block.
    ///
    /// `frames` must satisfy `flat.len() == frames * C`.
    #[must_use]
    pub fn from_flat(flat: &'a [T], frames: usize) -> Self {
        assert_eq!(flat.len(), frames * C);
        Self {
            flat,
            frames,
            _layout: PhantomData,
        }
    }

    /// Borrow one contiguous channel slice.
    #[must_use]
    pub fn channel(self, i: usize) -> &'a [T] {
        let start = i * self.frames;
        &self.flat[start..start + self.frames]
    }
}

impl<'a, T, const C: usize> BlockMut<'a, T, ChannelMajor, C> {
    /// Borrow a mutable channel-major flat block.
    ///
    /// `frames` must satisfy `flat.len() == frames * C`.
    #[must_use]
    pub fn from_flat(flat: &'a mut [T], frames: usize) -> Self {
        assert_eq!(flat.len(), frames * C);
        Self {
            flat,
            frames,
            _layout: PhantomData,
        }
    }

    /// Borrow one contiguous channel slice.
    #[must_use]
    pub fn channel(&self, i: usize) -> &[T] {
        let start = i * self.frames;
        &self.flat[start..start + self.frames]
    }

    /// Borrow one contiguous mutable channel slice.
    #[must_use]
    pub fn channel_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.frames;
        &mut self.flat[start..start + self.frames]
    }
}

/// Explicit block-processing API over typed block views.
///
/// This is the block-view companion to [`crate::Process`]. It is primarily useful for
/// processors that care about block layout rather than just slice length.
pub trait BlockProcess<X, Y = X> {
    /// Process one typed input block into one typed output block.
    fn process_block(&mut self, x: X, y: Y);
}

/// Explicit in-place block-processing API over typed block views.
pub trait BlockInplace<X> {
    /// Process one typed block in place.
    fn inplace_block(&mut self, xy: X);
}

/// Explicit frame-wise processing API over typed frame-major block views.
///
/// This is the companion to chunk-style processors where one processing step
/// consumes one whole frame rather than one scalar element.
pub trait FrameProcess<X, Y = X> {
    /// Process one frame-major block into another frame-major block.
    fn process_frames(&mut self, x: X, y: Y);
}

/// Explicit in-place frame-wise processing API.
pub trait FrameInplace<X> {
    /// Process one frame-major block in place frame by frame.
    fn inplace_frames(&mut self, xy: X);
}

/// Split-state block-processing API over typed block views.
pub trait SplitBlockProcess<X, Y = X, S: ?Sized = ()> {
    /// Process one typed input block into one typed output block.
    fn process_block(&self, state: &mut S, x: X, y: Y);
}

/// Split-state in-place block-processing API over typed block views.
pub trait SplitBlockInplace<X, S: ?Sized = ()> {
    /// Process one typed block in place.
    fn inplace_block(&self, state: &mut S, xy: X);
}

/// Split-state frame-wise processing API over frame-major block views.
pub trait SplitFrameProcess<X, Y = X, S: ?Sized = ()> {
    /// Process one frame-major block into another frame-major block.
    fn process_frames(&self, state: &mut S, x: X, y: Y);
}

/// Split-state in-place frame-wise processing API.
pub trait SplitFrameInplace<X, S: ?Sized = ()> {
    /// Process one frame-major block in place frame by frame.
    fn inplace_frames(&self, state: &mut S, xy: X);
}

impl<'a, 'b, X: Copy, Y, S: ?Sized, T, const C: usize>
    SplitBlockProcess<Block<'a, X, FrameMajor, C>, BlockMut<'b, Y, FrameMajor, C>, S> for T
where
    T: SplitProcess<X, Y, S>,
{
    fn process_block(
        &self,
        state: &mut S,
        x: Block<'a, X, FrameMajor, C>,
        mut y: BlockMut<'b, Y, FrameMajor, C>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        SplitProcess::block(self, state, x.flat(), y.flat_mut());
    }
}

impl<'a, X: Copy, S: ?Sized, T, const C: usize> SplitBlockInplace<BlockMut<'a, X, FrameMajor, C>, S>
    for T
where
    T: SplitInplace<X, S>,
{
    fn inplace_block(&self, state: &mut S, mut xy: BlockMut<'a, X, FrameMajor, C>) {
        SplitInplace::inplace(self, state, xy.flat_mut());
    }
}

impl<X, Y, C, S> BlockProcess<X, Y> for Split<C, S>
where
    C: SplitBlockProcess<X, Y, S>,
{
    fn process_block(&mut self, x: X, y: Y) {
        self.config.process_block(&mut self.state, x, y);
    }
}

impl<X, C, S> BlockInplace<X> for Split<C, S>
where
    C: SplitBlockInplace<X, S>,
{
    fn inplace_block(&mut self, xy: X) {
        self.config.inplace_block(&mut self.state, xy);
    }
}

impl<'a, 'b, X: Copy, Y, C, S, const Q: usize, const R: usize>
    SplitFrameProcess<Block<'a, X, FrameMajor, Q>, BlockMut<'b, Y, FrameMajor, R>, S> for Frames<C>
where
    C: SplitProcess<[X; Q], [Y; R], S>,
{
    fn process_frames(
        &self,
        state: &mut S,
        x: Block<'a, X, FrameMajor, Q>,
        mut y: BlockMut<'b, Y, FrameMajor, R>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        self.0.block(state, x.as_frames(), y.as_frames_mut());
    }
}

impl<'a, X: Copy, C, S, const N: usize> SplitFrameInplace<BlockMut<'a, X, FrameMajor, N>, S>
    for Frames<C>
where
    C: SplitInplace<[X; N], S>,
{
    fn inplace_frames(&self, state: &mut S, mut xy: BlockMut<'a, X, FrameMajor, N>) {
        self.0.inplace(state, xy.as_frames_mut());
    }
}

impl<X, Y, C, S> FrameProcess<X, Y> for Split<C, S>
where
    C: SplitFrameProcess<X, Y, S>,
{
    fn process_frames(&mut self, x: X, y: Y) {
        self.config.process_frames(&mut self.state, x, y);
    }
}

impl<X, C, S> FrameInplace<X> for Split<C, S>
where
    C: SplitFrameInplace<X, S>,
{
    fn inplace_frames(&mut self, xy: X) {
        self.config.inplace_frames(&mut self.state, xy);
    }
}
