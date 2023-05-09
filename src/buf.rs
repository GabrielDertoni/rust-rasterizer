use std::ops::{Index, IndexMut};
use std::marker::PhantomData;

use crate::{Pixel, ScreenPos};

pub type PixelBuf<'a> = MatrixSliceMut<'a, Pixel>;

pub struct MatrixSliceMut<'a, E> {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    ptr: *mut E,
    _marker: PhantomData<&'a [E]>,
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

    pub fn as_slice(&self) -> &[E] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.width * self.height) }
    }

    pub fn as_slice_mut(&self) -> &mut [E] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.width * self.height) }
    }

    pub fn ndc_to_screen(&self, [x, y, z]: [f32; 3]) -> ScreenPos {
        (
            ((x + 1.0) * self.width  as f32 / 2.0) as i32,
            ((1.0 - y) * self.height as f32 / 2.0) as i32,
            z
        )
    }

    pub fn borrow<'b>(&mut self) -> MatrixSliceMut<'b, E>
    where
        'a: 'b
    {
        MatrixSliceMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            ptr: self.ptr,
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
            let idx = y * self.width + x;
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
            let idx = y * self.width + x;
            &mut *self.ptr.add(idx)
        }
    }
}
