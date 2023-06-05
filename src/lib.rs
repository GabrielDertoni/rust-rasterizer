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
pub mod texture;
pub mod utils;
pub mod vec;

// TODO: Move to separate crate
pub mod frag_shaders;
pub mod shaders;

pub type ScreenPos = (i32, i32, f32);
pub type Pixel = [u8; 4];

use std::simd::{LaneCount, Mask, Simd, SimdElement, SupportedLaneCount};

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

#[derive(Clone, Copy, IntoSimd, Attributes)]
pub struct Vertex {
    #[position]
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

    pub fn from_obj(obj: &obj::Obj) -> (Self, std::vec::Vec<[usize; 3]>) {
        let mut vert_idxs_set = obj
            .tris
            .iter()
            .copied()
            .flatten()
            .collect::<std::vec::Vec<_>>();
        vert_idxs_set.sort_unstable();
        vert_idxs_set.dedup();
        let mut vert_buf = VertBuf::with_capacity(vert_idxs_set.len());

        for idxs in &vert_idxs_set {
            vert_buf.push(Vertex {
                position: obj.verts[idxs.position as usize],
                normal: obj.normals[idxs.normal as usize],
                uv: obj.uvs[idxs.uv as usize],
            });
        }

        let mut index_buf = std::vec::Vec::with_capacity(obj.tris.len());
        for tri in &obj.tris {
            index_buf.push(tri.map(|v| vert_idxs_set.binary_search(&v).unwrap()));
        }

        (vert_buf, index_buf)
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

pub trait FragmentShader<A: Attributes> {
    fn exec<const LANES: usize>(
        &self,
        mask: Mask<i32, LANES>,
        pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: A::Simd<LANES>,
    ) -> Vec4xN<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec_specialized(
        &self,
        mask: &mut Mask<i32, 4>,
        attrs: A::Simd<4>,
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

pub trait VertexShader<Vertex: IntoSimd> {
    type Output: Attributes;

    fn exec(&self, vertex: Vertex) -> Self::Output;

    fn exec_simd<const LANES: usize>(
        &self,
        vertex: Vertex::Simd<LANES>,
    ) -> <Self::Output as IntoSimd>::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        struct ConstFn<'a, T, Vertex, const LANES: usize>
        where
            LaneCount<LANES>: SupportedLaneCount,
            Vertex: IntoSimd,
            T: VertexShader<Vertex> + ?Sized,
        {
            shader: &'a T,
            vertex: Vertex::Simd<LANES>,
        }

        impl<'a, Vertex, T, const LANES: usize> vec::detail::ConstFn for ConstFn<'a, T, Vertex, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
            Vertex: IntoSimd,
            T: VertexShader<Vertex> + ?Sized,
        {
            type Output = T::Output;

            fn call<const I: usize>(&mut self) -> Self::Output {
                self.shader.exec(self.vertex.index(I))
            }

            fn call_runtime(&mut self, i: usize) -> Self::Output {
                self.shader.exec(self.vertex.index(i))
            }
        }

        StructureOfArray::from_array(vec::detail::array_from_fn(ConstFn {
            shader: self,
            vertex,
        }))
    }
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
pub trait Attributes: IntoSimd + Sized {
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
    type Vertex: IntoSimd;

    fn index(&self, index: usize) -> Self::Vertex;

    fn gather<const LANES: usize>(
        &self,
        index: Simd<usize, LANES>,
    ) -> <Self::Vertex as IntoSimd>::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        struct ConstFn<'a, T: ?Sized, const LANES: usize>(&'a T, Simd<usize, LANES>)
        where
            LaneCount<LANES>: SupportedLaneCount;

        impl<'a, T: VertexBuf + ?Sized, const LANES: usize> vec::detail::ConstFn for ConstFn<'a, T, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = T::Vertex;

            #[inline(always)]
            fn call<const I: usize>(&mut self) -> Self::Output {
                self.0.index(self.1[I])
            }

            #[inline(always)]
            fn call_runtime(&mut self, i: usize) -> Self::Output {
                self.0.index(self.1[i])
            }
        }

        StructureOfArray::from_array(vec::detail::array_from_fn(ConstFn(self, index)))
    }

    fn len(&self) -> usize;
}

impl<V: IntoSimd + Copy> VertexBuf for [V] {
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
