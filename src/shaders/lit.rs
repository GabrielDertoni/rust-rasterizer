use std::simd::{LaneCount, Mask, Simd, SimdFloat, SimdPartialOrd, SupportedLaneCount};

use crate::{
    buf,
    vec::{Mat4x4, Vec, Vec2, Vec3, Vec4, Vec4xN},
    Attributes, IntoSimd, StructureOfArray, Vertex,
};

#[derive(Clone, Copy, Debug, IntoSimd, Attributes)]
pub struct LitAttributes {
    #[position]
    pub position_ndc: Vec4,
    pub normal: Vec3,
    pub frag_position: Vec4,
    pub uv: Vec2,
    pub shadow_ndc: Vec3,
}

pub struct LitVertexShader {
    transform: Mat4x4,
    light_transform: Mat4x4,
}

impl LitVertexShader {
    pub fn new(transform: Mat4x4, light_transform: Mat4x4) -> Self {
        LitVertexShader {
            transform,
            light_transform,
        }
    }
}

impl crate::VertexShader<Vertex> for LitVertexShader {
    type Output = LitAttributes;

    fn exec(&self, vertex: Vertex) -> Self::Output {
        let shadow_clip = self.light_transform * vertex.position;
        let shadow_ndc = shadow_clip / shadow_clip.w;

        LitAttributes {
            position_ndc: self.transform * vertex.position,
            normal: vertex.normal,
            frag_position: vertex.position,
            uv: vertex.uv,
            shadow_ndc: shadow_ndc.xyz(),
        }
    }
}

pub struct LitFragmentShader<'a> {
    camera_pos: Vec3,
    normal_local_to_world: Mat4x4,
    light_pos: Vec3,
    light_color: Vec3,
    texture: buf::MatrixSlice<'a, [u8; 4]>,
    shadow_map: buf::MatrixSlice<'a, f32>,
}

impl<'a> LitFragmentShader<'a> {
    pub fn new(
        camera_pos: Vec3,
        model: Mat4x4,
        light_pos: Vec3,
        light_color: Vec3,
        texture: buf::MatrixSlice<'a, [u8; 4]>,
        shadow_map: buf::MatrixSlice<'a, f32>,
    ) -> Self {
        LitFragmentShader {
            camera_pos,
            normal_local_to_world: model.inverse().transpose(),
            light_pos,
            light_color,
            texture,
            shadow_map,
        }
    }
}

impl<'a> crate::FragmentShader<LitAttributes> for LitFragmentShader<'a> {
    // source: https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
        _attrs: LitAttributesSimd<LANES>,
    ) -> Vec4xN<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        panic!("unsuported");
    }

    fn exec_specialized(
        &self,
        mask: &mut Mask<i32, 4>,
        attrs: LitAttributesSimd<4>,
        _pixel_coords: Vec<Simd<i32, 4>, 2>,
        pixels: &mut Simd<u32, 4>,
    ) {
        let mask = *mask;

        let normal = (self.normal_local_to_world.splat() * attrs.normal.to_hom()).xyz();

        let bias_unit = 1. / self.shadow_map.width as f32;
        let bias = 5. * bias_unit;

        let lit_mask = {
            let shadow_uv =
                attrs.shadow_ndc.xy() * Simd::splat(0.5) + Vec::from([0.5, 0.5]).splat();
            let shadow_map = self.shadow_map.channel1();
            let light_depth = shadow_map
                .index_uv_or_4(shadow_uv, Mask::from([true; 4]), Simd::splat(f32::MIN))
                .x
                + Simd::splat(bias);
            attrs.shadow_ndc.z.simd_le(light_depth)
        };

        let ambient = self.light_color * 0.15;

        let light_dir = (self.light_pos.splat() - attrs.frag_position.xyz()).normalized();
        let diffuse_intensity = light_dir.dot(normal).simd_max(Simd::splat(0.));
        let diffuse = diffuse_intensity * self.light_color.splat();

        let view_dir = (self.camera_pos.splat() - attrs.frag_position.xyz()).normalized();
        let halfway_dir = (light_dir + view_dir).normalized();
        let specular_intensity = {
            let mut pow = normal.dot(halfway_dir).simd_max(Simd::splat(0.));
            for _ in 0..64_usize.ilog2() {
                pow *= pow;
            }
            pow
        };

        let specular = specular_intensity * self.light_color.splat();
        let texture_color = self.texture.texture_idx_4(attrs.uv, mask);

        let color = (ambient.splat()
            + (diffuse + specular).map_3(|el| lit_mask.select(el, Simd::splat(0.0))))
        .element_mul(texture_color.xyz());

        let out_color = Vec4xN::from([color.x, color.y, color.z, Simd::splat(1.)]);

        let colors = Simd::from(
            out_color
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

pub struct DebugLightIntensity {
    camera_pos: Vec3,
    normal_local_to_world: Mat4x4,
    light_pos: Vec3,
}

impl DebugLightIntensity {
    pub fn new(camera_pos: Vec3, model: Mat4x4, light_pos: Vec3) -> Self {
        DebugLightIntensity {
            camera_pos,
            normal_local_to_world: model.inverse().transpose(),
            light_pos,
        }
    }
}

impl crate::FragmentShader<LitAttributes> for DebugLightIntensity {
    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: LitAttributesSimd<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let normal = (self.normal_local_to_world.splat() * attrs.normal.to_hom()).xyz();

        let light_dir = (self.light_pos.splat() - attrs.frag_position.xyz()).normalized();
        let diffuse_intensity = light_dir.dot(normal).simd_max(Simd::splat(0.));

        /*
        let view_dir = (self.camera_pos.splat() - attrs.frag_position.xyz()).normalized();
        let halfway_dir = (light_dir + view_dir).normalized();
        let specular_intensity = {
            let mut pow = normal.dot(halfway_dir).simd_max(Simd::splat(0.));
            for _ in 0..64_usize.ilog2() {
                pow *= pow;
            }
            pow
        };
        */

        // Vec4xN::from([specular_intensity, specular_intensity, specular_intensity, Simd::splat(1.)])
        Vec4xN::from([
            diffuse_intensity,
            diffuse_intensity,
            diffuse_intensity,
            Simd::splat(1.),
        ])
        // Vec4xN::from([specular_intensity, diffuse_intensity, Simd::splat(0.), Simd::splat(1.)])
    }
}

pub mod gouraud {
    use super::*;

    pub struct VertexShader {
        transform: Mat4x4,
        light_color: Vec3,
        light_transform: Mat4x4,
        light_pos: Vec3,
    }

    impl VertexShader {
        pub fn new(
            transform: Mat4x4,
            light_transform: Mat4x4,
            light_pos: Vec3,
            light_color: Vec3,
        ) -> Self {
            VertexShader {
                transform,
                light_color,
                light_transform,
                light_pos,
            }
        }
    }

    impl crate::VertexShader<Vertex> for VertexShader {
        type Output = TexturedAttributes;

        fn exec(&self, vertex: Vertex) -> Self::Output {
            let shadow_clip = self.light_transform * vertex.position;
            let shadow_ndc = shadow_clip / shadow_clip.w;

            let light_dir = (self.light_pos - vertex.position.xyz()).normalized();

            TexturedAttributes {
                position_ndc: self.transform * vertex.position,
                light: light_dir.dot(vertex.normal) * self.light_color,
                uv: vertex.uv,
                shadow_ndc: shadow_ndc.xyz(),
            }
        }
    }

    #[derive(Clone, Copy, Debug, IntoSimd, Attributes)]
    pub struct TexturedAttributes {
        #[position]
        pub position_ndc: Vec4,
        pub light: Vec3,
        pub uv: Vec2,
        pub shadow_ndc: Vec3,
    }

    pub struct FragmentShader<'a> {
        texture: buf::MatrixSlice<'a, [u8; 4]>,
        shadow_map: buf::MatrixSlice<'a, f32>,
    }

    impl<'a> FragmentShader<'a> {
        pub fn new(
            texture: buf::MatrixSlice<'a, [u8; 4]>,
            shadow_map: buf::MatrixSlice<'a, f32>,
        ) -> Self {
            FragmentShader {
                texture,
                shadow_map,
            }
        }
    }

    impl<'a> crate::FragmentShader<TexturedAttributes> for FragmentShader<'a> {
        // source: https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
        fn exec<const LANES: usize>(
            &self,
            _mask: Mask<i32, LANES>,
            _pixel_coords: Vec<Simd<i32, LANES>, 2>,
            _attrs: TexturedAttributesSimd<LANES>,
        ) -> Vec4xN<LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            panic!("unsuported");
        }

        fn exec_specialized(
            &self,
            mask: &mut Mask<i32, 4>,
            attrs: TexturedAttributesSimd<4>,
            _pixel_coords: Vec<Simd<i32, 4>, 2>,
            pixels: &mut Simd<u32, 4>,
        ) {
            let mask = *mask;

            let bias_unit = 1. / self.shadow_map.width as f32;
            let bias = 5. * bias_unit;

            let lit_mask = {
                /*
                let shadow_uv =
                    attrs.shadow_ndc.xy() * Simd::splat(0.5) + Vec::from([0.5, 0.5]).splat();
                let shadow_map = self.shadow_map.channel1();
                let light_depth = shadow_map
                    .index_uv_or_4(shadow_uv, Mask::from([true; 4]), Simd::splat(f32::MIN))
                    .x
                    + Simd::splat(bias);
                attrs.shadow_ndc.z.simd_le(light_depth)
                */
                Mask::from([true; 4])
            };

            let ambient = Vec3::from([0.15, 0.15, 0.15]);
            // let texture_color = self.texture.texture_idx_4(attrs.uv, mask);
            let texture_color = Vec4::from([0.8, 0.8, 0.8, 1.0]).splat();

            let light_intensity = ambient.splat()
                + attrs
                    .light
                    .map_3(|el| lit_mask.select(el, Simd::splat(0.0)));
            let color = light_intensity.element_mul(texture_color.xyz());

            let out_color = Vec4xN::from([color.x, color.y, color.z, Simd::splat(1.)]);

            let colors = Simd::from(
                out_color
                    .map(|el| {
                        u32::from_ne_bytes(
                            (el.simd_clamp(Simd::splat(0.0), Simd::splat(1.0))
                                * Simd::splat(255.0))
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
}
