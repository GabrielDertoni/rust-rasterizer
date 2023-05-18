#![feature(
    slice_as_chunks,
    iter_next_chunk,
    portable_simd,
    int_roundings,
    split_array,
    str_split_whitespace_remainder,
    pointer_is_aligned
)]

pub mod buf;
pub mod obj;
pub mod prim3d;
pub mod utils;
pub mod vec;

// TODO: Move to separate crate
pub mod frag_shaders;
pub mod world;

pub type ScreenPos = (i32, i32, f32);
pub type Pixel = [u8; 4];

use std::simd::{LaneCount, Mask, Simd, SupportedLaneCount};

use obj::Index;
use vec::{Vec, Vec2i, Vec3, Vec4};

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

// #[derive(Clone, Copy)]
// pub struct Vert {
//     pub position: Vec4,
//     pub normal: Vec3,
//     pub texture: Vec2,
// }

pub struct SimdAttr<const LANES: usize>
where
    std::simd::LaneCount<LANES>: std::simd::SupportedLaneCount,
{
    pub normal: Vec<Simd<f32, LANES>, 3>,
    pub uv: Vec<Simd<i32, LANES>, 2>,
}

#[derive(Clone, Copy)]
pub struct VertBuf<'a> {
    pub positions: &'a [Vec4],
    pub normals: &'a [Vec3],
    pub uvs: &'a [Vec2i],
}

#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: Vec4,
    pub normal: Vec3,
    pub uv: Vec2i,
}

#[derive(Clone, Copy)]
pub struct Triangle {
    pub vertices: [Vertex; 3],
    pub area: i32,
}

impl<'a> VertexBuf<4> for VertBuf<'a> {
    type Index = Index;
    type SimdAttr = SimdAttr<4>;
    type Triangle = Triangle;

    #[inline(always)]
    fn position(&self, index: Index) -> Vec3 {
        self.positions[index.position as usize].xyz()
    }

    fn triangle_info(
        &self,
        v0: Self::Index,
        v1: Self::Index,
        v2: Self::Index,
        area: i32,
    ) -> Self::Triangle {
        Triangle {
            vertices: [
                Vertex {
                    position: self.positions[v0.position as usize],
                    normal: self.normals[v0.normal as usize],
                    uv: self.uvs[v0.uv as usize],
                },
                Vertex {
                    position: self.positions[v1.position as usize],
                    normal: self.normals[v1.normal as usize],
                    uv: self.uvs[v1.uv as usize],
                },
                Vertex {
                    position: self.positions[v2.position as usize],
                    normal: self.normals[v2.normal as usize],
                    uv: self.uvs[v2.uv as usize],
                },
            ],
            area,
        }
    }

    #[inline(always)]
    fn interpolate_simd(&self, _w: vec::Vec3x4, _tri: &Self::Triangle) -> SimdAttr<4> {
        panic!("unsupported");
    }

    #[inline(always)]
    fn interpolate_simd_specialized(
        &self,
        wi: Vec<Simd<i32, 4>, 3>,
        w: Vec<Simd<f32, 4>, 3>,
        tri: &Self::Triangle,
    ) -> Self::SimdAttr {
        let n0 = tri.vertices[0].normal.to_array();
        let n1 = tri.vertices[1].normal.to_array();
        let n2 = tri.vertices[2].normal.to_array();

        let n0 = Vec::from(crate::unroll_array!(i = 0, 1, 2 => Simd::splat(n0[i])));
        let n1 = Vec::from(crate::unroll_array!(i = 0, 1, 2 => Simd::splat(n1[i])));
        let n2 = Vec::from(crate::unroll_array!(i = 0, 1, 2 => Simd::splat(n2[i])));

        let uv0 = tri.vertices[0].uv.to_array();
        let uv1 = tri.vertices[1].uv.to_array();
        let uv2 = tri.vertices[2].uv.to_array();

        let uv0 = Vec::from(crate::unroll_array!(i = 0, 1 => Simd::splat(uv0[i])));
        let uv1 = Vec::from(crate::unroll_array!(i = 0, 1 => Simd::splat(uv1[i])));
        let uv2 = Vec::from(crate::unroll_array!(i = 0, 1 => Simd::splat(uv2[i])));

        SimdAttr {
            normal: w.x * n0 + w.y * n1 + w.z * n2,
            uv: (wi.x * uv0 + wi.y * uv1 + wi.z * uv2) / Simd::splat(tri.area),
        }
    }
}

// pub trait SimdAttribute {
//     type Simd<const LANES: usize>
//     where
//         LaneCount<LANES>: SupportedLaneCount;

//     fn interpolate<const LANES: usize>(
//         w: Vec<Simd<f32, LANES>, 3>,
//         v0: &Self,
//         v1: &Self,
//         v2: &Self,
//     ) -> Self::Simd<LANES>
//     where
//         LaneCount<LANES>: SupportedLaneCount;
// }

// #[derive(Clone, Copy)]
// pub struct Normal(Vec3);

// impl SimdAttribute for Normal {
//     type Simd<const LANES: usize> = Vec<Simd<f32, LANES>, 3>
//     where
//         LaneCount<LANES>: SupportedLaneCount;

//     fn interpolate<const LANES: usize>(
//         w: Vec<Simd<f32, LANES>, 3>,
//         Normal(v0): &Self,
//         Normal(v1): &Self,
//         Normal(v2): &Self,
//     ) -> Self::Simd<LANES>
//     where
//         LaneCount<LANES>: SupportedLaneCount,
//     {
//         w.x * v0.splat() + w.y * v1.splat() + w.z * v2.splat()
//     }
// }

// #[derive(Clone, Copy)]
// pub struct UV(Vec2);

// impl SimdAttribute for UV {
//     type Simd<const LANES: usize> = Vec<Simd<f32, LANES>, 2>
//     where
//         LaneCount<LANES>: SupportedLaneCount;

//     fn interpolate<const LANES: usize>(
//         w: Vec<Simd<f32, LANES>, 3>,
//         UV(v0): &Self,
//         UV(v1): &Self,
//         UV(v2): &Self,
//     ) -> Self::Simd<LANES>
//     where
//         LaneCount<LANES>: SupportedLaneCount,
//     {
//         w.x * v0.splat() + w.y * v1.splat() + w.z * v2.splat()
//     }
// }

pub trait FragShader {
    type SimdAttr<const LANES: usize>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        attrs: Self::SimdAttr<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec_specialized(
        &self,
        mask: Mask<i32, 4>,
        attrs: Self::SimdAttr<4>,
        pixels: &mut Simd<u32, 4>,
    ) {
        use std::simd::SimdFloat;

        let colors = Simd::from(
            self.exec(mask, attrs)
                .map_4(|el| {
                    (el.simd_clamp(Simd::splat(0.0), Simd::splat(1.0)) * Simd::splat(255.0))
                        .cast::<u8>()
                })
                .simd_transpose_4() // Convert from SoA to AoS
                .map_4(|el| u32::from_ne_bytes(el.to_array()))
                .to_array(),
        );
        *pixels = mask.select(colors, *pixels);
    }
}

pub trait VertexBuf<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Index: Copy;
    type SimdAttr;
    type Triangle;

    fn position(&self, index: Self::Index) -> Vec3;

    fn triangle_info(
        &self,
        v0: Self::Index,
        v1: Self::Index,
        v2: Self::Index,
        area: i32,
    ) -> Self::Triangle;

    fn interpolate_simd(&self, w: Vec<Simd<f32, LANES>, 3>, tri: &Self::Triangle)
        -> Self::SimdAttr;

    fn interpolate_simd_specialized(
        &self,
        _wi: Vec<Simd<i32, LANES>, 3>,
        w: Vec<Simd<f32, LANES>, 3>,
        tri: &Self::Triangle,
    ) -> Self::SimdAttr {
        self.interpolate_simd(w, tri)
    }
}
