use core::marker::PhantomData;

use crate::{Split, SplitInplace, SplitProcess};

/// Frame-major view layout marker.
///
/// A frame-major view corresponds to the ordinary `[[T; L]]` interpretation:
/// frames are contiguous and each frame holds `L` lane values.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameMajor;

/// Lane-major view layout marker.
///
/// A lane-major view stores `L` contiguous lane slices of equal length. This is
/// useful when each lane should be processed as one long contiguous run.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaneMajor;

/// Immutable typed view of a DSP slice.
///
/// `Layout` describes how the flat storage should be interpreted. The view
/// never allocates or physically transposes memory.
#[derive(Clone, Copy, Debug)]
pub struct View<'a, T, Layout, const L: usize> {
    flat: &'a [T],
    frames: usize,
    _layout: PhantomData<Layout>,
}

/// Mutable typed view of a DSP slice.
#[derive(Debug)]
pub struct ViewMut<'a, T, Layout, const L: usize> {
    flat: &'a mut [T],
    frames: usize,
    _layout: PhantomData<Layout>,
}

/// Apply a chunk-based processor frame by frame to a frame-major view.
///
/// This is the bridge between chunk semantics and typed views:
/// `SplitProcess<[X; Q], [Y; R], S>` becomes per-frame processing from
/// `View<FrameMajor, Q>` to `ViewMut<FrameMajor, R>`.
///
/// Use this when each frame is one logical chunk sample and the backing storage
/// is already frame-major. For lane-major layout-sensitive processing, use
/// [`crate::Lanes`] or [`crate::ByLane`] with explicit [`LaneMajor`] views
/// instead.
#[derive(Clone, Copy, Debug, Default)]
pub struct PerFrame<C>(pub C);

impl<'a, T, Layout, const L: usize> View<'a, T, Layout, L> {
    /// Flat underlying storage.
    #[must_use]
    pub fn flat(self) -> &'a [T] {
        self.flat
    }

    /// Number of frames in the view.
    #[must_use]
    pub const fn frames(self) -> usize {
        self.frames
    }

    /// Reinterpret the same backing storage under another layout marker.
    #[must_use]
    pub fn as_layout<Other>(self) -> View<'a, T, Other, L> {
        View {
            flat: self.flat,
            frames: self.frames,
            _layout: PhantomData,
        }
    }
}

impl<'a, T, Layout, const L: usize> ViewMut<'a, T, Layout, L> {
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

    /// Number of frames in the view.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Reinterpret the same backing storage under another layout marker.
    #[must_use]
    pub fn as_layout<Other>(self) -> ViewMut<'a, T, Other, L> {
        let Self { flat, frames, .. } = self;
        ViewMut {
            flat,
            frames,
            _layout: PhantomData,
        }
    }
}

impl<'a, T, const L: usize> View<'a, T, FrameMajor, L> {
    /// Borrow a conventional frame-major view.
    #[must_use]
    pub fn from_frames(frames: &'a [[T; L]]) -> Self {
        Self {
            flat: frames.as_flattened(),
            frames: frames.len(),
            _layout: PhantomData,
        }
    }

    /// Borrow the view as exact frames.
    #[must_use]
    pub fn as_frames(self) -> &'a [[T; L]] {
        let (frames, []) = self.flat.as_chunks::<L>() else {
            unreachable!()
        };
        frames
    }

    /// Borrow one frame.
    #[must_use]
    pub fn frame(self, i: usize) -> &'a [T; L] {
        &self.as_frames()[i]
    }
}

impl<'a, T, const L: usize> ViewMut<'a, T, FrameMajor, L> {
    /// Borrow a mutable conventional frame-major view.
    #[must_use]
    pub fn from_frames(frames: &'a mut [[T; L]]) -> Self {
        let frames_len = frames.len();
        Self {
            flat: frames.as_flattened_mut(),
            frames: frames_len,
            _layout: PhantomData,
        }
    }

    /// Borrow the view as exact frames.
    #[must_use]
    pub fn as_frames(&self) -> &[[T; L]] {
        let (frames, []) = self.flat.as_chunks::<L>() else {
            unreachable!()
        };
        frames
    }

    /// Borrow the view as exact mutable frames.
    #[must_use]
    pub fn as_frames_mut(&mut self) -> &mut [[T; L]] {
        let (frames, []) = self.flat.as_chunks_mut::<L>() else {
            unreachable!()
        };
        frames
    }

    /// Borrow one frame.
    #[must_use]
    pub fn frame(&self, i: usize) -> &[T; L] {
        &self.as_frames()[i]
    }

    /// Borrow one frame mutably.
    #[must_use]
    pub fn frame_mut(&mut self, i: usize) -> &mut [T; L] {
        &mut self.as_frames_mut()[i]
    }
}

impl<'a, T, const L: usize> View<'a, T, LaneMajor, L> {
    /// Borrow a lane-major flat view.
    ///
    /// `frames` must satisfy `flat.len() == frames * L`.
    #[must_use]
    pub fn from_flat(flat: &'a [T], frames: usize) -> Self {
        assert_eq!(flat.len(), frames * L);
        Self {
            flat,
            frames,
            _layout: PhantomData,
        }
    }

    /// Borrow one contiguous lane slice.
    #[must_use]
    pub fn lane(self, i: usize) -> &'a [T] {
        let start = i * self.frames;
        &self.flat[start..start + self.frames]
    }
}

impl<'a, T, const L: usize> ViewMut<'a, T, LaneMajor, L> {
    /// Borrow a mutable lane-major flat view.
    ///
    /// `frames` must satisfy `flat.len() == frames * L`.
    #[must_use]
    pub fn from_flat(flat: &'a mut [T], frames: usize) -> Self {
        assert_eq!(flat.len(), frames * L);
        Self {
            flat,
            frames,
            _layout: PhantomData,
        }
    }

    /// Borrow one contiguous lane slice.
    #[must_use]
    pub fn lane(&self, i: usize) -> &[T] {
        let start = i * self.frames;
        &self.flat[start..start + self.frames]
    }

    /// Borrow one contiguous mutable lane slice.
    #[must_use]
    pub fn lane_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.frames;
        &mut self.flat[start..start + self.frames]
    }
}

/// Explicit processing API over typed views.
///
/// This is the typed-view companion to [`crate::Process`]. It is primarily
/// useful for processors that care about view layout rather than just slice
/// length.
///
/// # Examples
///
/// ```rust
/// use dsp_process::{Offset, Split, View, ViewMut, ViewProcess};
///
/// let mut p = Split::stateless(Offset(3));
/// let x = View::from_frames(&[[1, 2], [3, 4]]);
/// let mut y = [[0; 2]; 2];
/// let yv = ViewMut::from_frames(&mut y);
/// ViewProcess::process_view(&mut p, x, yv);
/// assert_eq!(y, [[4, 5], [6, 7]]);
/// ```
pub trait ViewProcess<X, Y = X> {
    /// Process one typed input view into one typed output view.
    fn process_view(&mut self, x: X, y: Y);
}

/// Explicit in-place processing API over typed views.
pub trait ViewInplace<X> {
    /// Process one typed view in place.
    fn inplace_view(&mut self, xy: X);
}

/// Split-state processing API over typed views.
pub trait SplitViewProcess<X, Y = X, S: ?Sized = ()> {
    /// Process one typed input view into one typed output view.
    fn process_view(&self, state: &mut S, x: X, y: Y);
}

/// Split-state in-place processing API over typed views.
pub trait SplitViewInplace<X, S: ?Sized = ()> {
    /// Process one typed view in place.
    fn inplace_view(&self, state: &mut S, xy: X);
}

impl<'a, 'b, X, Y, S: ?Sized, T, const L: usize>
    SplitViewProcess<View<'a, X, FrameMajor, L>, ViewMut<'b, Y, FrameMajor, L>, S> for T
where
    X: Copy,
    T: SplitProcess<X, Y, S>,
{
    fn process_view(
        &self,
        state: &mut S,
        x: View<'a, X, FrameMajor, L>,
        mut y: ViewMut<'b, Y, FrameMajor, L>,
    ) {
        debug_assert_eq!(x.frames(), y.frames());
        SplitProcess::block(self, state, x.flat(), y.flat_mut());
    }
}

impl<'a, X, S: ?Sized, T, const L: usize> SplitViewInplace<ViewMut<'a, X, FrameMajor, L>, S> for T
where
    X: Copy,
    T: SplitInplace<X, S>,
{
    fn inplace_view(&self, state: &mut S, mut xy: ViewMut<'a, X, FrameMajor, L>) {
        SplitInplace::inplace(self, state, xy.flat_mut());
    }
}

impl<X, Y, C, S> ViewProcess<X, Y> for Split<C, S>
where
    C: SplitViewProcess<X, Y, S>,
{
    fn process_view(&mut self, x: X, y: Y) {
        self.config.process_view(&mut self.state, x, y);
    }
}

impl<X, C, S> ViewInplace<X> for Split<C, S>
where
    C: SplitViewInplace<X, S>,
{
    fn inplace_view(&mut self, xy: X) {
        self.config.inplace_view(&mut self.state, xy);
    }
}

impl<C, S> Split<PerFrame<C>, S> {
    /// Process a frame-major view frame by frame.
    ///
    /// ```rust
    /// use dsp_process::{ChunkInOut, FnSplitProcess, Split, View, ViewMut};
    ///
    /// let mut p = Split::stateless(ChunkInOut::<_, 2, 1>(FnSplitProcess(
    ///     |_: &mut (), [x0, x1]: [i32; 2]| [x0 + x1],
    /// )))
    /// .per_frame();
    /// let x = View::from_frames(&[[1, 2], [3, 4]]);
    /// let mut y = [[0; 1]; 2];
    /// let yv = ViewMut::from_frames(&mut y);
    /// p.process_frames(x, yv);
    /// assert_eq!(y, [[3], [7]]);
    /// ```
    pub fn process_frames<'a, 'b, X, Y, const Q: usize, const R: usize>(
        &mut self,
        x: View<'a, X, FrameMajor, Q>,
        mut y: ViewMut<'b, Y, FrameMajor, R>,
    ) where
        X: Copy,
        C: SplitProcess<[X; Q], [Y; R], S>,
    {
        debug_assert_eq!(x.frames(), y.frames());
        self.config
            .0
            .block(&mut self.state, x.as_frames(), y.as_frames_mut());
    }

    /// Process a frame-major view in place frame by frame.
    pub fn inplace_frames<'a, X, const L: usize>(&mut self, mut xy: ViewMut<'a, X, FrameMajor, L>)
    where
        X: Copy,
        C: SplitInplace<[X; L], S>,
    {
        self.config.0.inplace(&mut self.state, xy.as_frames_mut());
    }
}
