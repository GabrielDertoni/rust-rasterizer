#![feature(
    slice_as_chunks,
    iter_next_chunk,
    portable_simd,
    int_roundings,
    split_array,
    str_split_whitespace_remainder,
    pointer_is_aligned,
    never_type
)]

pub(crate) mod common;

pub mod buf;
pub mod vert_buf;
pub mod obj;
pub mod pipeline;
pub mod prim2d;
pub mod prim3d;
pub mod config;
pub mod simd_config;
pub mod texture;
pub mod utils;
pub mod vec;
pub mod math_utils;

// TODO: Move to separate crate
// pub mod frag_shaders;
pub mod shaders;

pub type ScreenPos = (i32, i32, f32);
pub type Pixel = [u8; 4];

use core::panic;
use std::simd::{LaneCount, Mask, Simd, SimdElement, SupportedLaneCount};

use texture::BorrowedMutTexture;
use vec::{Vec, Vec2, Vec3, Vec4, Vec4xN};
use vec::{Vec2i, Vec3xN};

pub fn triangles_iter<'a, V>(
    vert: &'a [V],
    tris: &'a [[u32; 3]],
) -> impl Iterator<Item = [&'a V; 3]> + 'a {
    tris.iter()
        .map(|&[p0, p1, p2]| [&vert[p0 as usize], &vert[p1 as usize], &vert[p2 as usize]])
}

pub fn clear_color(mut pixels: BorrowedMutTexture<[u8; 4]>, color: u32) {
    for pixel in pixels.as_slice_mut() {
        *pixel = color.to_be_bytes();
    }
}

#[derive(Clone, Copy, IntoSimd, Attributes)]
pub struct Vertex {
    #[position]
    pub position: Vec4,
    pub normal: Vec3,
    pub uv: Vec2,
}

pub struct FragmentShaderSimdDefaultImpl<F>(F);

impl<F, A, const LANES: usize> FragmentShaderSimd<AttributesSimdDefaultImpl<A>, LANES>
    for FragmentShaderSimdDefaultImpl<F>
where
    LaneCount<LANES>: SupportedLaneCount,
    F: FragmentShader<A>,
    A: Attributes + IntoSimd + Copy,
{
    fn exec(
        &self,
        mask: Mask<i32, LANES>,
        pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: AttributesSimdDefaultImplSimd<A::Simd<LANES>>,
    ) -> Vec4xN<LANES> {
        Vec4xN::from_array(std::array::from_fn(|lane| {
            if mask.test(lane) {
                self.0.exec(pixel_coords.index(lane), attrs.index(lane).0)
            } else {
                Vec4::zero()
            }
        }))
    }
}

pub enum FragmentShaderImpl<'a, Scalar: ?Sized, Simd: ?Sized> {
    Scalar(&'a Scalar),
    Simd(&'a Simd),
}

impl<'a, Scalar: ?Sized, Simd: ?Sized> FragmentShaderImpl<'a, Scalar, Simd> {
    pub fn unwrap_scalar(self) -> &'a Scalar {
        match self {
            FragmentShaderImpl::Scalar(scalar) => scalar,
            _ => panic!("expected a scalar impl"),
        }
    }

    pub fn unwrap_simd(self) -> &'a Simd {
        match self {
            FragmentShaderImpl::Simd(simd) => simd,
            _ => panic!("expected a simd impl"),
        }
    }
}

impl<'a, Scalar: ?Sized> FragmentShaderImpl<'a, Scalar, !> {
    pub fn scalar(scalar: &'a Scalar) -> Self {
        FragmentShaderImpl::Scalar(scalar)
    }
}

impl<'a, Simd: ?Sized> FragmentShaderImpl<'a, !, Simd> {
    pub fn simd(simd: &'a Simd) -> Self {
        FragmentShaderImpl::Simd(simd)
    }
}

pub trait FragmentShader<A: Attributes> {
    fn exec(&self, pixel_coords: Vec2i, attrs: A) -> Vec4;

    fn into_impl(&self) -> FragmentShaderImpl<Self, !> {
        FragmentShaderImpl::Scalar(self)
    }

    fn scalar_impl(&self) -> FragmentShaderImpl<Self, !> {
        self.into_impl()
    }
}

pub trait FragmentShaderSimd<A: AttributesSimd<LANES>, const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn exec(
        &self,
        mask: Mask<i32, LANES>,
        pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: A::Simd<LANES>,
    ) -> Vec4xN<LANES>;

    fn into_impl(&self) -> FragmentShaderImpl<!, Self> {
        FragmentShaderImpl::Simd(self)
    }

    fn simd_impl(&self) -> FragmentShaderImpl<!, Self> {
        self.into_impl()
    }
}

impl<A: Attributes> FragmentShader<A> for ! {
    fn exec(&self, _: Vec2i, _: A) -> Vec4 {
        match *self {}
    }
}

impl<A: AttributesSimd<LANES>, const LANES: usize> FragmentShaderSimd<A, LANES> for !
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn exec(&self, _: Mask<i32, LANES>, _: Vec<Simd<i32, LANES>, 2>, _: A::Simd<LANES>) -> Vec4xN<LANES> {
        match *self {}
    }
}

pub trait VertexShader<Vertex /*: IntoSimd*/> {
    type Output: Attributes;

    fn exec(&self, vertex: Vertex) -> Self::Output;
}

#[macro_export]
macro_rules! vertex_shader_simd {
    ([$($capture:ident : $capture_ty:ty),*] |$vert:ident : $ty:ty| -> $ret:ty { $($body:tt)* }) => {{
        struct Shader {
            $($capture: $capture_ty),*
        }

        impl $crate::VertexShader<$ty> for Shader {
            type Output = $ret;

            #[inline(always)]
            fn exec(&self, vertex: Vertex) -> Self::Output {
                use $crate::StructureOfArray;

                self.exec_simd::<1>($crate::IntoSimd::splat(vertex)).index(0)
            }

            #[inline]
            fn exec_simd<const LANES: usize>(
                &self,
                $vert: <$ty as $crate::IntoSimd>::Simd<LANES>,
            ) -> <Self::Output as $crate::IntoSimd>::Simd<LANES>
            where
                std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
            {
                let Shader { $($capture),* } = self;
                $($body)*
            }
        }

        Shader { $($capture),* }
    }};
}

impl<F, Vertex, Output> VertexShader<Vertex> for F
where
    Vertex: IntoSimd,
    F: Fn(Vertex) -> Output,
    Output: Attributes,
{
    type Output = Output;

    #[inline(always)]
    fn exec(&self, vertex: Vertex) -> Self::Output {
        (self)(vertex)
    }
}

pub use macros::Attributes;

pub trait Attributes {
    fn interpolate(p0: &Self, p1: &Self, p2: &Self, w: Vec<f32, 3>) -> Self;

    fn position(&self) -> &Vec4;
    fn position_mut(&mut self) -> &mut Vec4;
}

impl Attributes for Vec4 {
    fn interpolate(&p0: &Self, &p1: &Self, &p2: &Self, w: Vec<f32, 3>) -> Self {
        w.x * p0 + w.y * p1 + w.z * p2
    }

    fn position(&self) -> &Vec4 {
        self
    }

    fn position_mut(&mut self) -> &mut Vec4 {
        self
    }
}

pub use macros::AttributesSimd;

pub trait AttributesSimd<const LANES: usize>: IntoSimd
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn interpolate(p0: &Self, p1: &Self, p2: &Self, w: Vec3xN<LANES>) -> Self::Simd<LANES>;
}

impl<const LANES: usize> AttributesSimd<LANES> for Vec4
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn interpolate(p0: &Self, p1: &Self, p2: &Self, w: Vec3xN<LANES>) -> Self::Simd<LANES> {
        w.x * p0.splat() + w.y * p1.splat() + w.z * p2.splat()
    }
}

pub struct AttributesSimdDefaultImpl<A>(A);

pub struct AttributesSimdDefaultImplSimd<S>(S);

impl<A: IntoSimd + Copy> IntoSimd for AttributesSimdDefaultImpl<A> {
    type Simd<const LANES: usize> = AttributesSimdDefaultImplSimd<A::Simd<LANES>>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        AttributesSimdDefaultImplSimd(self.0.splat())
    }
}

impl<S, const LANES: usize> StructureOfArray<LANES>
    for AttributesSimdDefaultImplSimd<S>
where
    S: StructureOfArray<LANES>,
    S::Structure: Copy,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Structure = AttributesSimdDefaultImpl<S::Structure>;

    fn from_array(array: [Self::Structure; LANES]) -> Self {
        AttributesSimdDefaultImplSimd(S::from_array(std::array::from_fn(|lane| array[lane].0)))
    }

    fn index(&self, i: usize) -> Self::Structure {
        AttributesSimdDefaultImpl(self.0.index(i))
    }
}

impl<A, const LANES: usize> AttributesSimd<LANES> for AttributesSimdDefaultImpl<A>
where
    A: Attributes + IntoSimd + Copy,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn interpolate(p0: &Self, p1: &Self, p2: &Self, w: Vec3xN<LANES>) -> Self::Simd<LANES> {
        AttributesSimdDefaultImplSimd::from_array(std::array::from_fn(|lane| {
            AttributesSimdDefaultImpl(A::interpolate(&p0.0, &p1.0, &p2.0, w.index(lane)))
        }))
    }
}

pub trait VertexBuf {
    type Vertex;

    fn index(&self, index: usize) -> Self::Vertex;

    fn len(&self) -> usize;
}

impl<'a, B: VertexBuf + ?Sized> VertexBuf for &'a B {
    type Vertex = B::Vertex;

    #[inline(always)]
    fn index(&self, index: usize) -> Self::Vertex {
        (*self).index(index)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        (*self).len()
    }
}

impl<V: Copy> VertexBuf for [V] {
    type Vertex = V;

    #[inline(always)]
    fn index(&self, index: usize) -> Self::Vertex {
        self[index]
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

impl<V: Copy> VertexBuf for std::vec::Vec<V> {
    type Vertex = V;

    #[inline(always)]
    fn index(&self, index: usize) -> Self::Vertex {
        self[index]
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

pub use macros::IntoSimd;

pub trait IntoSimd: Sized {
    /// SoA representation of `Self`.
    type Simd<const LANES: usize>: StructureOfArray<LANES, Structure = Self>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;
}

pub trait StructureOfArray<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Structure;

    fn from_array(array: [Self::Structure; LANES]) -> Self;
    fn index(&self, i: usize) -> Self::Structure;
}

impl<T: SimdElement> IntoSimd for T {
    type Simd<const LANES: usize> = Simd<T, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    #[inline(always)]
    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        Simd::splat(self)
    }
}

impl<T, const LANES: usize> StructureOfArray<LANES> for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement,
{
    type Structure = T;

    #[inline(always)]
    fn from_array(array: [Self::Structure; LANES]) -> Self {
        Simd::from_array(array)
    }

    #[inline(always)]
    fn index(&self, i: usize) -> Self::Structure {
        self[i]
    }
}
