use std::marker::PhantomData;
use std::ops::{Bound, Index, IndexMut, RangeBounds};
use std::simd::{
    LaneCount, Mask, Simd, SimdConstPtr, SimdElement, SimdFloat, SimdPartialOrd, SupportedLaneCount,
};

use crate::vec::{Mat4x4, Vec, Vec2xN, Vec3, Vec4xN};
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
        assert!(buf.as_ptr().is_aligned_to(std::mem::align_of::<u32>()));
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

    pub fn borrow<'b>(&'b self) -> MatrixSlice<'b, E> {
        MatrixSlice {
            width: self.width,
            height: self.height,
            stride: self.stride,
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    pub fn ndc_to_screen(&self) -> Mat4x4 {
        Vec3::from([self.width as f32, self.height as f32, 1.0]).to_scale()
            * Vec3::from([0.5, 0.5, 0.0]).to_translation()
            * Vec3::from([0.5, -0.5, 1.0]).to_scale()
    }

    pub fn channel1(&self) -> MatrixSlice<'a, [E; 1]> {
        unsafe { std::mem::transmute_copy(self) }
    }
}

impl<'a, T, const N: usize> MatrixSlice<'a, [T; N]> {
    #[inline]
    pub fn flatten(self) -> MatrixSlice<'a, T> {
        MatrixSlice {
            width: self.width * N,
            height: self.height,
            stride: self.stride * N,
            ptr: self.ptr.cast(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn simd_index_soa<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
        mask: Mask<isize, LANES>,
    ) -> [Simd<T, LANES>; N]
    where
        LaneCount<LANES>: SupportedLaneCount,
        T: SimdElement + Default,
    {
        self.simd_index_soa_or(x, y, mask, Simd::splat(Default::default()))
    }

    #[inline]
    pub fn simd_index_soa_or<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
        mask: Mask<isize, LANES>,
        or: Simd<T, LANES>,
    ) -> [Simd<T, LANES>; N]
    where
        LaneCount<LANES>: SupportedLaneCount,
        T: SimdElement,
    {
        let inbounds = x.simd_le(Simd::splat(self.width)) & y.simd_le(Simd::splat(self.height));
        debug_assert!(inbounds.all(), "out of bounds");

        let mask = mask & inbounds;

        let flat: MatrixSlice<T> = self.borrow().flatten();
        let x = x * Simd::splat(N);

        std::array::from_fn(move |off| unsafe {
            flat.simd_index_select_or_unchecked(mask, x + Simd::splat(off), y, or)
        })
    }

    #[inline]
    pub unsafe fn simd_index_soa_or_unchecked<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
        mask: Mask<isize, LANES>,
        or: Simd<T, LANES>,
    ) -> [Simd<T, LANES>; N]
    where
        LaneCount<LANES>: SupportedLaneCount,
        T: SimdElement,
    {
        let flat: MatrixSlice<T> = self.borrow().flatten();
        let x = x * Simd::splat(N);

        std::array::from_fn(move |off| unsafe {
            flat.simd_index_select_or_unchecked(mask, x + Simd::splat(off), y, or)
        })
    }

    /// Index the matrix slice with uv coordinates. The access is clamped if the index is out of bounds.
    #[inline]
    pub fn index_uv<const LANES: usize>(
        &self,
        uv: Vec2xN<LANES>,
        mask: Mask<i32, LANES>,
    ) -> Vec<Simd<T, LANES>, N>
    where
        LaneCount<LANES>: SupportedLaneCount,
        T: SimdElement + Default,
    {
        let [u, v] = uv.to_array();

        let u = u.simd_clamp(Simd::splat(0.), Simd::splat(1.));
        let v = v.simd_clamp(Simd::splat(0.), Simd::splat(1.));

        let x = (u * Simd::splat(self.width as f32)).cast::<usize>();
        let y = ((Simd::splat(1.) - v) * Simd::splat(self.height as f32)).cast::<usize>();

        Vec::from(self.simd_index_soa(x, y, mask.cast()))
    }

    /// Index the matrix slice with uv coordinates. The access is repeated if the index is out of bounds.
    #[inline]
    pub fn index_uv_repeat<const LANES: usize>(
        &self,
        uv: Vec2xN<LANES>,
        mask: Mask<i32, LANES>,
    ) -> Vec<Simd<T, LANES>, N>
    where
        LaneCount<LANES>: SupportedLaneCount,
        T: SimdElement + Default,
    {
        let [u, v] = uv.to_array();

        let u = u % Simd::splat(1.);
        let v = v % Simd::splat(1.);

        // `.to_int()` will return 0 for `false` and `-1` for true
        let u = u - u.is_sign_negative().to_int().cast();
        let v = v - v.is_sign_negative().to_int().cast();

        let x = (u * Simd::splat(self.width as f32)).cast::<usize>();
        let y = ((Simd::splat(1.) - v) * Simd::splat(self.height as f32)).cast::<usize>();

        Vec::from(self.simd_index_soa(x, y, mask.cast()))
    }

    /// Index the matrix slice with uv coordinates. `or` is returned if the access is out of bounds
    #[inline]
    pub fn index_uv_or<const LANES: usize>(
        &self,
        uv: Vec2xN<LANES>,
        mask: Mask<i32, LANES>,
        or: Simd<T, LANES>,
    ) -> Vec<Simd<T, LANES>, N>
    where
        LaneCount<LANES>: SupportedLaneCount,
        T: SimdElement,
    {
        let [u, v] = uv.to_array();

        let mask = mask & u.is_sign_positive() & v.is_sign_positive();

        let x = (u * Simd::splat(self.width as f32)).cast::<usize>();
        let y = ((Simd::splat(1.) - v) * Simd::splat(self.height as f32)).cast::<usize>();

        Vec::from(self.simd_index_soa_or(x, y, mask.cast(), or))
    }
}

impl<'a, T: SimdElement, const N: usize> MatrixSlice<'a, [T; N]> {
    #[inline]
    pub fn index_uv_or_4(
        &self,
        uv: Vec2xN<4>,
        mask: Mask<i32, 4>,
        or: Simd<T, 4>,
    ) -> Vec<Simd<T, 4>, N> {
        use core::arch::x86_64;

        let [u, v] = uv.to_array();

        let mask = mask & u.is_sign_positive() & v.is_sign_positive();

        let x = u * Simd::splat(self.width as f32);
        let x: Simd<i32, 4> = unsafe { x86_64::_mm_cvtps_epi32(x86_64::__m128::from(x)).into() };
        let x = x.cast::<usize>();

        let y = (Simd::splat(1.) - v) * Simd::splat(self.height as f32);
        let y: Simd<i32, 4> = unsafe { x86_64::_mm_cvtps_epi32(x86_64::__m128::from(y)).into() };
        let y = y.cast::<usize>();

        let inbounds = x.simd_le(Simd::splat(self.width)) & y.simd_le(Simd::splat(self.height));
        debug_assert!(inbounds.all(), "out of bounds");

        let mask = mask.cast() & inbounds;

        let flat: MatrixSlice<T> = self.borrow().flatten();
        let x = x * Simd::splat(N);

        Vec::from(std::array::from_fn(move |off| unsafe {
            let idx = y * Simd::splat(self.stride) + x + Simd::splat(off);
            Simd::gather_select_unchecked(flat.as_slice(), mask, idx, or)
        }))
    }
}

impl<'a, E> MatrixSlice<'a, E>
where
    E: SimdElement,
{
    #[inline]
    pub fn simd_index<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
    ) -> Simd<E, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
        E: Default,
    {
        self.simd_index_or(x, y, Simd::splat(Default::default()))
    }

    #[inline]
    pub fn simd_index_or<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
        or: Simd<E, LANES>,
    ) -> Simd<E, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let inbounds = x.simd_le(Simd::splat(self.width)) & y.simd_le(Simd::splat(self.height));
        debug_assert!(inbounds.all(), "out of bounds");

        let idx = y * Simd::splat(self.stride) + x;
        unsafe { Simd::gather_select_unchecked(self.as_slice(), inbounds, idx, or) }
    }

    #[inline]
    pub fn simd_index_select_or<const LANES: usize>(
        &self,
        mask: Mask<isize, LANES>,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
        or: Simd<E, LANES>,
    ) -> Simd<E, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let inbounds = x.simd_le(Simd::splat(self.width)) & y.simd_le(Simd::splat(self.height));
        debug_assert!(inbounds.all(), "out of bounds");
        unsafe { self.simd_index_select_or_unchecked(mask & inbounds, x, y, or) }
    }

    #[inline]
    pub unsafe fn simd_index_select_or_unchecked<const LANES: usize>(
        &self,
        mask: Mask<isize, LANES>,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
        or: Simd<E, LANES>,
    ) -> Simd<E, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let idx = y * Simd::splat(self.stride) + x;
        Simd::gather_select_unchecked(self.as_slice(), mask, idx, or)
    }
}

type Texture<'a> = MatrixSlice<'a, Pixel>;

impl<'a> Texture<'a> {
    /*
    #[inline]
    pub fn texture_idx<const LANES: usize>(
        &self,
        uv: Vec2xN<LANES>,
        mask: Mask<i32, LANES>,
    ) -> Vec4xN<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let mask = mask.cast();
        let [u, v] = uv.to_array();

        let x = (u * Simd::splat(self.width as f32)).cast::<usize>();
        let y = ((Simd::splat(1.) - v) * Simd::splat(self.height as f32)).cast::<usize>();

        let ptr = Simd::splat(self.ptr.cast::<u32>());

        // Multiply x by 4, this is necessary since we are indexing `colors` now
        let x = x << Simd::splat(2);

        let or = Simd::splat(0);

        let idx = y * Simd::splat(self.stride) + x;
        let colors = Simd::gather_select_ptr(ptr.wrapping_add(idx), mask, or);

        Vec::from(colors.to_array())
            .map(|el| Simd::from_array(el.to_ne_bytes()))
            .simd_transpose_4()

        let u8_max = Simd::splat(255.0_f32);
        Vec4xN::from([
            r.cast() / u8_max,
            g.cast() / u8_max,
            b.cast() / u8_max,
            a.cast() / u8_max,
        ])
    }
    */

    #[inline]
    pub fn texture_idx_4(&self, uv: Vec2xN<4>, mask: Mask<i32, 4>) -> Vec4xN<4> {
        let mask = mask.cast();
        let [u, v] = uv.to_array();

        let u = u.simd_clamp(Simd::splat(0.), Simd::splat(1.));
        let v = v.simd_clamp(Simd::splat(0.), Simd::splat(1.));

        let x = (u * Simd::splat(self.width as f32)).cast::<usize>();
        let y = ((Simd::splat(1.) - v) * Simd::splat(self.height as f32)).cast::<usize>();

        let ptr = Simd::splat(self.ptr.cast::<u32>());
        let idx = y * Simd::splat(self.stride) + x;
        let colors =
            unsafe { Simd::gather_select_ptr(ptr.wrapping_add(idx), mask, Simd::splat(0)) };

        let u8_max = Simd::splat(255.0_f32);
        Vec::from(colors.to_array())
            .map(|el| Simd::from_array(el.to_ne_bytes()))
            .simd_transpose_4()
            .map(|el| el.cast::<f32>() / u8_max)
    }
}

impl<'a, E> Index<(usize, usize)> for MatrixSlice<'a, E> {
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
