use crate::{
    texture::{BorrowedTextureRGBA, TextureWrap},
    vec::{Mat4x4, Vec, Vec2, Vec4},
    Attributes, FragmentShader, IntoSimd, StructureOfArray, Vertex, VertexShader, VertexSimd,
};

use std::simd::{LaneCount, Mask, Simd, SimdPartialEq, SupportedLaneCount};

#[derive(Clone, Copy, Debug, IntoSimd, Attributes)]
pub struct TexturedAttributes {
    #[position]
    pub position_ndc: Vec4,
    pub uv: Vec2,
}

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

    /*
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
    */
}

pub struct TexturedFragmentShader<'a> {
    texture: BorrowedTextureRGBA<'a>,
}

impl<'a> TexturedFragmentShader<'a> {
    pub fn new(texture: BorrowedTextureRGBA<'a>) -> Self {
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
        let colors = self.texture.simd_index_uv(attrs.uv, *mask);
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

    fn exec_basic(&self, attrs: TexturedAttributes) -> Vec4 {
        self.texture.index_uv(attrs.uv, TextureWrap::Repeat)
    }
}
