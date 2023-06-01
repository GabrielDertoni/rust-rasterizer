use crate::{
    vec::{Mat4x4, Vec, Vec2, Vec2xN, Vec3xN, Vec4},
    buf::Texture,
    Vertex, VertexShader, Attributes, FragmentShader,
};

use std::simd::{LaneCount, Mask, Simd, SupportedLaneCount, SimdPartialEq};

pub struct TexturedVertexShader {
    transform: Mat4x4,
}

impl TexturedVertexShader {
    pub fn new(transform: Mat4x4) -> Self {
        TexturedVertexShader {
            transform,
        }
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
}

#[derive(Clone, Copy)]
pub struct TexturedAttributes {
    pub position_ndc: Vec4,
    pub uv: Vec2,
}

pub struct TexturedAttributesSimd<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub uv: Vec2xN<LANES>,
}

impl Attributes for TexturedAttributes {
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
}

pub struct TexturedFragmentShader<'a> {
    texture: Texture<'a>,
}

impl<'a> TexturedFragmentShader<'a> {
    pub fn new(texture: Texture<'a>) -> Self {
        TexturedFragmentShader {
            texture,
        }
    }
}

impl<'a> FragmentShader for TexturedFragmentShader<'a> {
    type SimdAttr<const LANES: usize> = TexturedAttributesSimd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

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
        attrs: Self::SimdAttr<4>,
        _pixel_coords: Vec<Simd<i32, 4>, 2>,
        pixels: &mut Simd<u32, 4>,
    ) {
        let colors = self.texture.index_uv(attrs.uv, *mask);
        let colors = Simd::from(colors.map(|el| u32::from_ne_bytes(el.to_array())).to_array());

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