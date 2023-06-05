use crate::{
    buf::Texture,
    vec::{Mat4x4, Vec, Vec2, Vec3xN, Vec4},
    Attributes, FragmentShader, IntoSimd, StructureOfArray, Vertex, VertexShader, VertexSimd,
};

use std::simd::{LaneCount, Mask, Simd, SimdPartialEq, SupportedLaneCount};

#[derive(Clone, Copy, Debug, IntoSimd, Attributes)]
pub struct TexturedAttributes {
    #[position]
    pub position_ndc: Vec4,
    pub uv: Vec2,
}

/*
pub struct TexturedAttributesSimd<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub uv: Vec2xN<LANES>,
}

impl IntoSimd for TexturedAttributes {
    type Simd<const LANES: usize> = TexturedAttributesSimd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        TexturedAttributesSimd {
            uv: self.uv.splat(),
        }
    }
}

impl<const LANES: usize> StructureOfArray<LANES> for TexturedAttributesSimd<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Structure = TexturedAttributes;

    fn from_array(array: [Self::Structure; LANES]) -> Self {
        TexturedAttributesSimd {
            uv: Vec2xN::from_array(array.map(|el| el.uv)),
        }
    }

    fn index(&self, i: usize) -> Self::Structure {
        TexturedAttributes {
            position_ndc: todo!(),
            uv: self.uv.index(i),
        }
    }
}

impl Attributes for TexturedAttributes {
    fn interpolate<const LANES: usize>(
        p0: &Self,
        p1: &Self,
        p2: &Self,
        w: Vec3xN<LANES>,
    ) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        TexturedAttributesSimd {
            uv: w.x * p0.uv.splat() + w.y * p1.uv.splat() + w.z * p2.uv.splat(),
        }
    }

    fn position(&self) -> &Vec4 {
        &self.position_ndc
    }

    fn position_mut(&mut self) -> &mut Vec4 {
        &mut self.position_ndc
    }
}
*/

pub struct TexturedVertexShader {
    transform: Mat4x4,
}

impl TexturedVertexShader {
    pub fn new(transform: Mat4x4) -> Self {
        TexturedVertexShader { transform }
    }
}

impl VertexShader<Vertex> for TexturedVertexShader {
    type Output = TexturedAttributes;

    fn exec(&self, vertex: Vertex) -> Self::Output {
        TexturedAttributes {
            position_ndc: self.transform * vertex.position,
            uv: vertex.uv,
        }
    }

    fn exec_simd<const LANES: usize>(
        &self,
        vertex: VertexSimd<LANES>,
    ) -> TexturedAttributesSimd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        TexturedAttributesSimd {
            position_ndc: self.transform.splat() * vertex.position,
            uv: vertex.uv,
        }
    }
}

pub struct TexturedFragmentShader<'a> {
    texture: Texture<'a>,
}

impl<'a> TexturedFragmentShader<'a> {
    pub fn new(texture: Texture<'a>) -> Self {
        TexturedFragmentShader { texture }
    }
}

impl<'a> FragmentShader<TexturedAttributes> for TexturedFragmentShader<'a> {
    #[inline(always)]
    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
        _attrs: TexturedAttributesSimd<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        panic!("unsupported")
    }

    #[inline(always)]
    fn exec_specialized(
        &self,
        mask: &mut Mask<i32, 4>,
        attrs: TexturedAttributesSimd<4>,
        _pixel_coords: Vec<Simd<i32, 4>, 2>,
        pixels: &mut Simd<u32, 4>,
    ) {
        let colors = self.texture.index_uv_repeat(attrs.uv, *mask);
        let colors = Simd::from(
            colors
                .map(|el| u32::from_ne_bytes(el.to_array()))
                .to_array(),
        );

        // Will be 1s where the pixel is visible
        let alpha = colors.as_array()[3];
        let alpha_mask = Simd::from_array(alpha.to_ne_bytes()).simd_eq(Simd::splat(0xff));

        // Update the original mask to not update the depth buffer when we hit a transparent pixel
        *mask &= alpha_mask.cast();

        let mask = mask.cast::<i8>();
        let mask = Simd::splat(u32::from_ne_bytes(mask.to_int().cast().to_array()));

        *pixels = (colors & mask) + (*pixels & !mask);
    }
}
