#![feature(slice_as_chunks, iter_next_chunk, portable_simd, int_roundings, split_array)]

pub mod buf;
pub mod obj;
pub mod prim3d;
pub mod vec;

pub type ScreenPos = (i32, i32, f32);
pub type Pixel = [u8; 4];

use vec::{Vec4, Vec3};

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
    pub pos: Vec4,
    pub normal: Vec3,
}

impl prim3d::Vertex for Vert {
    type Attr = Vec3;
    type SimdAttr<const LANES: usize> = vec::Vec<std::simd::Simd<f32, LANES>, 3>
    where
        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount;

    fn position(&self) -> &Vec3 {
        self.pos.slice::<0, 3>()
    }

    fn interpolate(w: Vec3, v0: &Self, v1: &Self, v2: &Self) -> Vec3 {
        w.x * v0.normal + w.y * v1.normal + w.z * v2.normal
    }

    fn interpolate_simd<const LANES: usize>(
        w: vec::Vec<std::simd::Simd<f32, LANES>, 3>,
        v0: &Self,
        v1: &Self,
        v2: &Self,
    ) -> Self::SimdAttr<LANES>
    where
        std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount
    {
        w.x * v0.normal.splat() + w.y * v1.normal.splat() + w.z * v2.normal.splat()
    }
}

pub fn compute_normals(vert: &[Vec4], tris: &[[u32; 3]]) -> Vec<Vert> {
    let mut vert = vert.into_iter()
        .copied()
        .map(|pos| Vert {
            pos,
            normal: Vec3::zero(),
        })
        .collect::<Vec<_>>();

    for &[p0_ix, p1_ix, p2_ix] in tris {
        let p0 = vert[p0_ix as usize].pos.xyz();
        let p1 = vert[p1_ix as usize].pos.xyz();
        let p2 = vert[p2_ix as usize].pos.xyz();
        let n = (p0 - p1).cross(p2 - p1).normalized();
        vert[p0_ix as usize].normal += n;
        vert[p1_ix as usize].normal += n;
        vert[p2_ix as usize].normal += n;
    }

    for v in &mut vert {
        v.normal.normalize();
    }

    vert
}
