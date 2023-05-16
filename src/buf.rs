use std::marker::PhantomData;
use std::ops::{Bound, Index, IndexMut, RangeBounds};

use crate::vec::{Vec3, Mat4x4};
use crate::Pixel;

pub type PixelBuf<'a> = MatrixSliceMut<'a, Pixel>;

#[derive(Clone, Copy)]
pub struct MatrixSlice<'a, E> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    ptr: *const E,
    _marker: PhantomData<&'a [E]>,
}

impl<'a, E> MatrixSlice<'a, E> {
    pub fn new(buf: &'a [E], width: usize, height: usize) -> Self {
        assert_eq!(buf.len(), width * height);
        let ptr = buf.as_ptr();
        MatrixSlice {
            width,
            height,
            stride: width,
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[E] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.width * self.height) }
    }
}

pub struct MatrixSliceMut<'a, E> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    ptr: *mut E,
    _marker: PhantomData<&'a mut [E]>,
}

impl<'a, E> MatrixSliceMut<'a, E> {
    pub fn new(buf: &'a mut [E], width: usize, height: usize) -> Self {
        assert_eq!(buf.len(), width * height);
        let ptr = buf.as_mut_ptr();
        MatrixSliceMut {
            width,
            height,
            stride: width,
            ptr,
            _marker: PhantomData,
        }
    }

    pub unsafe fn from_ptr(ptr: *mut E, width: usize, height: usize) -> Self {
        MatrixSliceMut {
            width,
            height,
            stride: width,
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[E] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.width * self.height) }
    }

    pub fn as_slice_mut(&self) -> &mut [E] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.width * self.height) }
    }

    pub fn ndc_to_screen(&self) -> Mat4x4 {
        /*
        p
            // flip y axis and scale range from [-1, 1] -> [-0.5, 0.5]
            .hom_scale(Vec3::from([0.5, -0.5, 1.0]))
            // translate range from [-0.5, 0.5] -> [0.0, 1.0]
            .hom_translate(Vec3::from([0.5, 0.5, 0.0]))
            // scale range from [0.0, 1.0] -> [0.0, width]
            .hom_scale(Vec3::from([self.width as f32, self.height as f32, 1.0]))
        */

        Vec3::from([self.width as f32, self.height as f32, 1.0]).to_scale()
            * Vec3::from([0.5, 0.5, 0.0]).to_translation()
            * Vec3::from([0.5, -0.5, 1.0]).to_scale()
    }

    pub fn borrow<'b>(&'b mut self) -> MatrixSliceMut<'b, E> {
        MatrixSliceMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    pub fn slice_mut<'b, B: RangeBounds<usize>>(
        &'b mut self,
        rows: B,
        cols: B,
    ) -> MatrixSliceMut<'b, E> {
        let start_row = match rows.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start + 1,
            Bound::Unbounded => 0,
        };
        let end_row = match rows.end_bound() {
            Bound::Included(&end) => end + 1,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.height,
        };
        let start_col = match cols.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start + 1,
            Bound::Unbounded => 0,
        };
        let end_col = match cols.end_bound() {
            Bound::Included(&end) => end + 1,
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.width,
        };
        if end_row > self.height || end_col > self.width {
            panic!("out of bounds");
        }
        MatrixSliceMut {
            width: end_col - start_col,
            height: end_row - start_row,
            stride: self.stride,
            ptr: unsafe { self.ptr.add(start_row * self.stride + start_col) },
            _marker: PhantomData,
        }
    }
}

impl<'a, E> Index<(usize, usize)> for MatrixSliceMut<'a, E> {
    type Output = E;

    fn index(&self, (x, y): (usize, usize)) -> &E {
        if x >= self.width || y >= self.height {
            panic!("out of bounds");
        }
        unsafe {
            let idx = y * self.stride + x;
            &*self.ptr.add(idx)
        }
    }
}

impl<'a, E> IndexMut<(usize, usize)> for MatrixSliceMut<'a, E> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut E {
        if x >= self.width || y >= self.height {
            panic!("out of bounds");
        }
        unsafe {
            let idx = y * self.stride + x;
            &mut *self.ptr.add(idx)
        }
    }
}
