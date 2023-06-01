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
pub struct Vertex {
    pub position: Vec4,
    pub normal: Vec3,
    pub uv: Vec2,
}

/// INVARIANT: The length of all fields is the same
#[derive(Clone, Default)]
pub struct VertBuf {
    pub positions: std::vec::Vec<Vec4>,
    pub normals: std::vec::Vec<Vec3>,
    pub uvs: std::vec::Vec<Vec2>,
}

impl VertBuf {
    pub fn with_capacity(cap: usize) -> Self {
        VertBuf {
            positions: std::vec::Vec::with_capacity(cap),
            normals: std::vec::Vec::with_capacity(cap),
            uvs: std::vec::Vec::with_capacity(cap),
        }
    }

    pub fn push(&mut self, vertex: Vertex) {
        self.positions.push(vertex.position);
        self.normals.push(vertex.normal);
        self.uvs.push(vertex.uv);
    }
}

impl VertexBuf for VertBuf {
    type Vertex = Vertex;

    fn index(&self, index: usize) -> Vertex {
        Vertex {
            position: self.positions[index],
            normal: self.normals[index],
            uv: self.uvs[index],
        }
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.positions.len(), self.normals.len());
        debug_assert_eq!(self.positions.len(), self.uvs.len());
        self.positions.len()
    }
}

/// INVARIANT: The length of all fields is the same
#[derive(Clone, Copy)]
pub struct VertBufSlice<'a> {
    pub positions: &'a [Vec4],
    pub normals: &'a [Vec3],
    pub uvs: &'a [Vec2],
}

impl<'a> VertexBuf for VertBufSlice<'a> {
    type Vertex = Vertex;

    #[inline(always)]
    fn index(&self, index: usize) -> Vertex {
        Vertex {
            position: self.positions[index],
            normal: self.normals[index],
            uv: self.uvs[index],
        }
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.positions.len(), self.normals.len());
        debug_assert_eq!(self.positions.len(), self.uvs.len());
        self.positions.len()
    }
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

    fn exec(&self, vertex: Vertex) -> Self::Output {
        Vertex {
            position: self.transform * vertex.position,
            normal: vertex.normal,
            uv: vertex.uv,
        }
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

    fn position(&self) -> &Vec4 {
        &self.position
    }

    fn position_mut(&mut self) -> &mut Vec4 {
        &mut self.position
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
        mask: &mut Mask<i32, 4>,
        attrs: Self::SimdAttr<4>,
        pixel_coords: Vec<Simd<i32, 4>, 2>,
        pixels: &mut Simd<u32, 4>,
    ) {
        use std::simd::SimdFloat;

        let colors = Simd::from(
            self.exec(*mask, pixel_coords, attrs)
                .map(|el| {
                    u32::from_ne_bytes(
                        (el.simd_clamp(Simd::splat(0.0), Simd::splat(1.0)) * Simd::splat(255.0))
                            .cast::<u8>()
                            .to_array(),
                    )
                })
                .to_array(),
        );

        // Casting the mask to i8, makes the mask structure have 8x4=32 bits. Since -1 represents true
        // in the mask, and bits are stored in twos-compliment, that is a bitset with only 1s when true
        // If we then convert the mask to u32, we'll have a mask for the pixels. We just broadcast this
        // to every channel and mask the things we want.
        let mask = mask.cast::<i8>();
        let mask = Simd::splat(u32::from_ne_bytes(mask.to_int().cast().to_array()));

        *pixels = (colors & mask) + (*pixels & !mask)
    }
}

pub trait VertexShader<Vertex> {
    type Output: Attributes;

    fn exec(&self, vertex: Vertex) -> Self::Output;
}

impl<Vertex, F, Output> VertexShader<Vertex> for F
where
    F: Fn(Vertex) -> Output,
    Output: Attributes,
{
    type Output = Output;

    fn exec(&self, vertex: Vertex) -> Self::Output {
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

    fn position(&self) -> &Vec4;
    fn position_mut(&mut self) -> &mut Vec4;
}

impl Attributes for Vec4 {
    type Simd<const LANES: usize> = Vec4xN<LANES>
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

    fn position(&self) -> &Vec4 {
        self
    }

    fn position_mut(&mut self) -> &mut Vec4 {
        self
    }
}

/*
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
*/

pub trait VertexBuf {
    type Vertex;

    fn index(&self, index: usize) -> Self::Vertex;
    fn len(&self) -> usize;
}

impl<V: Copy> VertexBuf for [V] {
    type Vertex = V;

    fn index(&self, index: usize) -> Self::Vertex {
        self[index]
    }

    fn len(&self) -> usize {
        self.len()
    }
}