use crate::{
    texture::{BorrowedTextureRGBA, TextureWrap, RowMajorPowerOf2},
    vec::{Mat4x4, Vec, Vec2, Vec4, Vec4x4, Vec4xN},
    Attributes, AttributesSimd, IntoSimd, StructureOfArray, Vertex, VertexShader, FragmentShaderSimd, math_utils::{simd_remap, rgb_hex, simd_mix},
};

use std::simd::{Mask, Simd, LaneCount, SupportedLaneCount};

#[derive(Clone, Copy, Debug, IntoSimd, Attributes, AttributesSimd)]
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
    near: f32,
    far: f32,
    texture: BorrowedTextureRGBA<'a, RowMajorPowerOf2>,
}

impl<'a> TexturedFragmentShader<'a> {
    pub fn new(texture: BorrowedTextureRGBA<'a, RowMajorPowerOf2>) -> Self {
        TexturedFragmentShader {
            near: 0.1,
            far: 50.,
            texture,
        }
    }
}

impl<'a, const LANES: usize> FragmentShaderSimd<TexturedAttributes, LANES> for TexturedFragmentShader<'a>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn exec(
        &self,
        mask: Mask<i32, LANES>,
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: TexturedAttributesSimd<LANES>,
    ) -> Vec4xN<LANES>
    {
        let depth = simd_remap(
            Simd::splat(1.) / attrs.position_ndc.w,
            Simd::splat(self.near)..Simd::splat(self.far),
            Simd::splat(0.)..Simd::splat(1.)
        );
        let fog_color = rgb_hex(0x61b7e8).splat();
        let color = self.texture.simd_index_uv(attrs.uv, mask, TextureWrap::Repeat);
        (simd_mix(fog_color, color.xyz(), depth * depth * depth), color.w).into()
    }

    // #[inline(always)]
    // fn exec_specialized(
    //     &self,
    //     mask: &mut Mask<i32, 4>,
    //     attrs: TexturedAttributesSimd<4>,
    //     _pixel_coords: Vec<Simd<i32, 4>, 2>,
    //     pixels: &mut Simd<u32, 4>,
    // ) {

    //     // Will be 1s where the pixel is visible
    //     let alpha = colors.as_array()[3];
    //     let alpha_mask = Simd::from_array(alpha.to_ne_bytes()).simd_eq(Simd::splat(0xff));

    //     // Update the original mask to not update the depth buffer when we hit a transparent pixel
    //     *mask &= alpha_mask.cast();

    //     let mask = mask.cast::<i8>();
    //     let mask = Simd::splat(u32::from_ne_bytes(mask.to_int().cast().to_array()));

    //     *pixels = (colors & mask) + (*pixels & !mask);
    // }
}
