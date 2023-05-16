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

use std::simd::{Simd, LaneCount, SupportedLaneCount};

use vec::{Vec, Vec2, Vec3, Vec4};

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
    pub texture: Vec<Simd<f32, LANES>, 2>,
}

impl prim3d::Vertex for Vert {
    type Attr = Vec3;
    type SimdAttr<const LANES: usize> = SimdAttr<LANES>
    where
        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount;

    fn position(&self) -> &Vec3 {
        self.position.slice::<0, 3>()
    }

    fn interpolate(w: Vec3, v0: &Self, v1: &Self, v2: &Self) -> Vec3 {
        w.x * v0.normal + w.y * v1.normal + w.z * v2.normal
    }

    fn interpolate_simd<const LANES: usize>(
        w: vec::Vec<std::simd::Simd<f32, LANES>, 3>,
        v0: &Self,
        v1: &Self,
        v2: &Self,
    ) -> SimdAttr<LANES>
    where
        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount
    {
        SimdAttr {
            normal: w.x * v0.normal.splat() + w.y * v1.normal.splat() + w.z * v2.normal.splat(),
            texture: w.x * v0.texture.splat() + w.y * v1.texture.splat() + w.z * v2.texture.splat(),
        }
    }
}

pub fn compute_normals(vert: &mut [Vert], tris: &[[u32; 3]]) {
    for v in &mut *vert {
        v.normal = Vec3::zero();
    }

    for &[p0_ix, p1_ix, p2_ix] in tris {
        let p0 = vert[p0_ix as usize].position.xyz();
        let p1 = vert[p1_ix as usize].position.xyz();
        let p2 = vert[p2_ix as usize].position.xyz();
        let n = (p0 - p1).cross(p2 - p1).normalized();
        vert[p0_ix as usize].normal += n;
        vert[p1_ix as usize].normal += n;
        vert[p2_ix as usize].normal += n;
    }

    for v in vert {
        v.normal.normalize();
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
