#![feature(
    slice_as_chunks,
    iter_next_chunk,
    portable_simd,
    int_roundings,
    split_array,
    str_split_whitespace_remainder,
)]

pub mod buf;
pub mod obj;
pub mod prim3d;
pub mod vec;
pub mod utils;

pub type ScreenPos = (i32, i32, f32);
pub type Pixel = [u8; 4];

use std::simd::{Simd, LaneCount, SupportedLaneCount, Mask};

use vec::{Vec, Vec2, Vec3, Vec4};
use obj::Index;

pub fn triangles_iter<'a, V>(
    vert: &'a [V],
    tris: &'a [[u32; 3]],
) -> impl Iterator<Item = [&'a V; 3]> + 'a {
    tris.iter()
        .map(|&[p0, p1, p2]| [&vert[p0 as usize], &vert[p1 as usize], &vert[p2 as usize]])
}

pub fn clear_color(pixels: buf::PixelBuf, color: u32) {
    for pixel in pixels.as_slice_mut() {
        *pixel = color.to_be_bytes();
    }
}

#[derive(Clone, Copy)]
pub struct Vert {
    pub position: Vec4,
    pub normal: Vec3,
    pub texture: Vec2,
}

pub struct SimdAttr<const LANES: usize>
where
    std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
{
    pub normal: Vec<Simd<f32, LANES>, 3>,
    pub uv: Vec<Simd<f32, LANES>, 2>,
}

#[derive(Clone, Copy)]
pub struct VertBuf<'a> {
    pub positions: &'a [Vec4],
    pub normals: &'a [Vec3],
    pub uvs: &'a [Vec2],
}

impl<'a> prim3d::VertexBuf for VertBuf<'a> {
    type Index = Index;
    type SimdAttr<const LANES: usize> = SimdAttr<LANES>
    where
        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount;

    fn position(&self, index: Index) -> Vec3 {
        self.positions[index.position as usize].xyz()
    }

    fn interpolate_simd<const LANES: usize>(
        &self,
        w: vec::Vec<std::simd::Simd<f32, LANES>, 3>,
        v0: Index,
        v1: Index,
        v2: Index,
    ) -> SimdAttr<LANES>
    where
        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount
    {
        let n0 = self.normals[v0.normal as usize].splat();
        let n1 = self.normals[v1.normal as usize].splat();
        let n2 = self.normals[v2.normal as usize].splat();

        let uv0 = self.uvs[v0.uv as usize].splat();
        let uv1 = self.uvs[v1.uv as usize].splat();
        let uv2 = self.uvs[v2.uv as usize].splat();

        SimdAttr {
            normal:   w.x *  n0 + w.y *  n1 + w.z *  n2,
            uv:       w.x * uv0 + w.y * uv1 + w.z * uv2,
        }
    }
}

impl prim3d::Vertex for Vert {
    type Attr = Vec3;

    fn position(&self) -> &Vec3 {
        self.position.slice::<0, 3>()
    }

    fn interpolate(w: Vec3, v0: &Self, v1: &Self, v2: &Self) -> Vec3 {
        w.x * v0.normal + w.y * v1.normal + w.z * v2.normal
    }
}

pub trait SimdAttribute {
    type Simd<const LANES: usize>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn interpolate<const LANES: usize>(
        w: Vec<Simd<f32, LANES>, 3>,
        v0: &Self,
        v1: &Self,
        v2: &Self,
    ) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;
}

#[derive(Clone, Copy)]
pub struct Normal(Vec3);

impl SimdAttribute for Normal {
    type Simd<const LANES: usize> = Vec<Simd<f32, LANES>, 3>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn interpolate<const LANES: usize>(
        w: Vec<Simd<f32, LANES>, 3>,
        Normal(v0): &Self,
        Normal(v1): &Self,
        Normal(v2): &Self,
    ) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        w.x * v0.splat() + w.y * v1.splat() + w.z * v2.splat()
    }
}

#[derive(Clone, Copy)]
pub struct UV(Vec2);

impl SimdAttribute for UV {
    type Simd<const LANES: usize> = Vec<Simd<f32, LANES>, 2>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn interpolate<const LANES: usize>(
        w: Vec<Simd<f32, LANES>, 3>,
        UV(v0): &Self,
        UV(v1): &Self,
        UV(v2): &Self,
    ) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        w.x * v0.splat() + w.y * v1.splat() + w.z * v2.splat()
    }
}

pub trait FragShader<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type SimdAttr;

    fn exec(&self, _mask: Mask<i32, LANES>, attrs: Self::SimdAttr) -> vec::Vec<Simd<f32, LANES>, 4>;
}

pub struct FromFn<F, A, const LANES: usize> {
    f: F,
    _marker: std::marker::PhantomData<A>,
}

impl<F, A, const LANES: usize> FromFn<F, A, LANES>
where
    F: Fn(Mask<i32, LANES>, A) -> vec::Vec<Simd<f32, LANES>, 4>,
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn new(f: F) -> Self {
        FromFn { f, _marker: std::marker::PhantomData }
    }
}


impl<F, A, const LANES: usize> FragShader<LANES> for FromFn<F, A, LANES>
where
    F: Fn(Mask<i32, LANES>, A) -> vec::Vec<Simd<f32, LANES>, 4>,
    LaneCount<LANES>: SupportedLaneCount,
{
    type SimdAttr = A;

    fn exec(&self, mask: Mask<i32, LANES>, attrs: A) -> vec::Vec<Simd<f32, LANES>, 4> {
        (self.f)(mask, attrs)
    }
}

