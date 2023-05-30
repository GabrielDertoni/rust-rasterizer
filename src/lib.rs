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
pub mod shaders;

pub type ScreenPos = (i32, i32, f32);
pub type Pixel = [u8; 4];

use std::simd::{LaneCount, Mask, Simd, SupportedLaneCount};

use obj::Index;
use vec::{Mat4x4, Vec, Vec2, Vec3, Vec4, Vec4xN};

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

pub struct SimdAttrs<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub position: Vec<Simd<f32, LANES>, 4>,
    pub normal: Vec<Simd<f32, LANES>, 3>,
    pub uv: Vec<Simd<f32, LANES>, 2>,
}

#[derive(Clone, Copy)]
pub struct VertBuf<'a> {
    pub positions: &'a [Vec4],
    pub normals: &'a [Vec3],
    pub uvs: &'a [Vec2],
}

#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: Vec4,
    pub normal: Vec3,
    pub uv: Vec2,
}

impl<'a> VertexBuf for VertBuf<'a> {
    type Index = Index;
    type Vertex = Vertex;

    #[inline(always)]
    fn index(&self, index: Self::Index) -> Self::Vertex {
        Vertex {
            position: self.positions[index.position as usize],
            normal: self.normals[index.normal as usize],
            uv: self.uvs[index.uv as usize],
        }
    }

    /*
    #[inline(always)]
    fn interpolate_simd_specialized(
        &self,
        _wi: Vec<Simd<i32, 4>, 3>,
        w: Vec<Simd<f32, 4>, 3>,
        _z: Simd<f32, 4>,
        tri: &Self::Triangle,
    ) -> Self::SimdAttr {
        let p0 = tri.vertices[0].position.to_array();
        let p1 = tri.vertices[1].position.to_array();
        let p2 = tri.vertices[2].position.to_array();

        let p0 = Vec::from(crate::unroll_array!(i = 0, 1, 2 => Simd::splat(p0[i])));
        let p1 = Vec::from(crate::unroll_array!(i = 0, 1, 2 => Simd::splat(p1[i])));
        let p2 = Vec::from(crate::unroll_array!(i = 0, 1, 2 => Simd::splat(p2[i])));

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
            position: w.x * p0 + w.y * p1 + w.z * p2,
            normal: w.x * n0 + w.y * n1 + w.z * n2,
            uv: w.x * uv0 + w.y * uv1 + w.z * uv2,
        }
    }
    */
}

pub struct VertShader {
    transform: Mat4x4,
}

impl VertShader {
    pub fn new(transform: Mat4x4) -> Self {
        VertShader { transform }
    }
}

impl VertexShader<Vertex> for VertShader {
    type Output = Vertex;

    fn exec(&self, vertex: Vertex) -> (Self::Output, Vec4) {
        (vertex, self.transform * vertex.position)
    }
}

impl Attributes for Vertex {
    type Simd<const LANES: usize> = SimdAttrs<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    #[rustfmt::skip]
    #[inline(always)]
    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        SimdAttrs {
            position: self.position.splat(),
            normal:   self.normal.splat(),
            uv:       self.uv.splat(),
        }
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn interpolate<const LANES: usize>(
        p0: &Self,
        p1: &Self,
        p2: &Self,
        w: Vec<Simd<f32, LANES>, 3>,
    ) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        SimdAttrs {
            position: w.x * p0.position.splat() + w.y * p1.position.splat() + w.z * p2.position.splat(),
            normal:   w.x * p0.normal.splat()   + w.y * p1.normal.splat()   + w.z * p2.normal.splat(),
            uv:       w.x * p0.uv.splat()       + w.y * p1.uv.splat()       + w.z * p2.uv.splat(),
        }
    }
}

pub trait FragmentShader {
    type SimdAttr<const LANES: usize>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec<const LANES: usize>(
        &self,
        mask: Mask<i32, LANES>,
        pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: Self::SimdAttr<LANES>,
    ) -> Vec4xN<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec_specialized(
        &self,
        mask: Mask<i32, 4>,
        attrs: Self::SimdAttr<4>,
        pixel_coords: Vec<Simd<i32, 4>, 2>,
        pixels: &mut Simd<u32, 4>,
    ) {
        use std::simd::SimdFloat;

        let colors = Simd::from(
            self.exec(mask, pixel_coords, attrs)
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

pub trait VertexShader<Vertex> {
    type Output: Attributes;

    fn exec(&self, vertex: Vertex) -> (Self::Output, Vec4);
}

impl<Vertex, F, Output> VertexShader<Vertex> for F
where
    F: Fn(Vertex) -> (Output, Vec4),
    Output: Attributes,
{
    type Output = Output;

    fn exec(&self, vertex: Vertex) -> (Self::Output, Vec4) {
        (self)(vertex)
    }
}

pub trait Attributes: Sized {
    /// SoA representation of `Self`.
    type Simd<const LANES: usize>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn interpolate<const LANES: usize>(
        p0: &Self,
        p1: &Self,
        p2: &Self,
        w: Vec<Simd<f32, LANES>, 3>,
    ) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;
}

/// The empty attribute set
impl Attributes for () {
    type Simd<const LANES: usize> = ()
    where
        LaneCount<LANES>: SupportedLaneCount;

    #[inline(always)]
    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        ()
    }

    #[inline(always)]
    fn interpolate<const LANES: usize>(
        (): &Self,
        (): &Self,
        (): &Self,
        _: Vec<Simd<f32, LANES>, 3>,
    ) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        ()
    }
}

/// The empty attribute set
impl<const M: usize, const N: usize> Attributes for vec::Mat<f32, M, N> {
    type Simd<const LANES: usize> = vec::Mat<Simd<f32, LANES>, M, N>
    where
        LaneCount<LANES>: SupportedLaneCount;

    #[inline(always)]
    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        self.splat()
    }

    #[inline(always)]
    fn interpolate<const LANES: usize>(
        p0: &Self,
        p1: &Self,
        p2: &Self,
        w: Vec<Simd<f32, LANES>, 3>,
    ) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        w.x * p0.splat() + w.y * p1.splat() + w.z * p2.splat()
    }
}

macro_rules! impl_attributes_tuple {
    () => {};
    ($($ty:ident : $idx:tt),+; $($rest:tt)*) => {
        impl_attributes_tuple!($($ty : $idx),+);
        impl_attributes_tuple!($($rest)*);
    };
    ($($ty:ident : $idx:tt),+) => {
        #[allow(unused_parens)] // No idea why it says those parenthesis are unecessary...
        impl<$($ty),+> Attributes for ($($ty,)+)
        where
            $($ty: Attributes,)+
        {
            type Simd<const LANES: usize> = ($(<$ty as Attributes>::Simd<LANES>),+)
            where
                LaneCount<LANES>: SupportedLaneCount;

            #[inline(always)]
            fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                ($(self.$idx.splat()),+)
            }

            #[inline(always)]
            fn interpolate<const LANES: usize>(
                p0: &Self,
                p1: &Self,
                p2: &Self,
                w: Vec<Simd<f32, LANES>, 3>,
            ) -> Self::Simd<LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                ($(<$ty as Attributes>::interpolate(&p0.$idx, &p1.$idx, &p2.$idx, w)),+)
            }
        }
    };
}

impl_attributes_tuple!(
    T0: 0;
    T0: 0, T1: 1;
    T0: 0, T1: 1, T2: 2;
    T0: 0, T1: 1, T2: 2, T3: 3;
    T0: 0, T1: 1, T2: 2, T3: 3, T4: 4;
);

pub trait VertexBuf {
    type Index: Copy;
    type Vertex;

    fn index(&self, index: Self::Index) -> Self::Vertex;
}

impl<V: Copy> VertexBuf for [V] {
    type Index = usize;
    type Vertex = V;

    fn index(&self, index: Self::Index) -> Self::Vertex {
        self[index]
    }
}
