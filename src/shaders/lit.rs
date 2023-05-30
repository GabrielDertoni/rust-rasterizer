use std::simd::{
    LaneCount, Mask, Simd, SimdFloat, SimdPartialEq, SimdPartialOrd, SupportedLaneCount,
};

use crate::{
    buf,
    vec::{Mat4x4, Vec, Vec2, Vec3, Vec2xN, Vec3xN, Vec4, Vec4xN},
    Attributes, FragmentShader, Vertex, VertexShader,
};

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

impl VertexShader<Vertex> for LitVertexShader {
    type Output = LitAttributes;

    fn exec(&self, vertex: Vertex) -> (Self::Output, Vec4) {
        let shadow_uv = self.light_transform * vertex.position;
        let shadow_uv = shadow_uv / shadow_uv.w;
        (
            LitAttributes {
                normal: vertex.normal,
                frag_position: vertex.position,
                uv: vertex.uv,
                shadow_uv,
            },
            self.transform * vertex.position,
        )
    }
}

pub struct LitAttributes {
    pub normal: Vec3,
    pub frag_position: Vec4,
    pub uv: Vec2,
    pub shadow_uv: Vec4,
}

pub struct LitAttributesSimd<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub normal: Vec3xN<LANES>,
    pub frag_position: Vec4xN<LANES>,
    pub uv: Vec2xN<LANES>,
    pub shadow_uv: Vec4xN<LANES>,
}

impl Attributes for LitAttributes {
    type Simd<const LANES: usize> = LitAttributesSimd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn splat<const LANES: usize>(self) -> Self::Simd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        LitAttributesSimd {
            normal: self.normal.splat(),
            frag_position: self.frag_position.splat(),
            uv: self.uv.splat(),
            shadow_uv: self.shadow_uv.splat(),
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
        LitAttributesSimd {
            normal: w.x * p0.normal.splat() + w.y * p1.normal.splat() + w.z * p2.normal.splat(),
            frag_position: w.x * p0.frag_position.splat()
                + w.y * p1.frag_position.splat()
                + w.z * p2.frag_position.splat(),
            uv: w.x * p0.uv.splat() + w.y * p1.uv.splat() + w.z * p2.uv.splat(),
            shadow_uv: w.x * p0.shadow_uv.splat()
                + w.y * p1.shadow_uv.splat()
                + w.z * p2.shadow_uv.splat(),
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

impl<'a> FragmentShader for LitFragmentShader<'a> {
    type SimdAttr<const LANES: usize> = LitAttributesSimd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    // source: https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
    fn exec<const LANES: usize>(
        &self,
        mask: Mask<i32, LANES>,
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: LitAttributesSimd<LANES>,
    ) -> Vec4xN<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let normal = (self.normal_local_to_world.splat() * attrs.normal.to_hom()).xyz();
        let shadow_pos = self.shadow_map.ndc_to_screen().splat() * attrs.shadow_uv;
        let shadow_pos_i32 = Vec::from([shadow_pos.x.cast::<i32>(), shadow_pos.y.cast::<i32>()]);

        let in_bounds = shadow_pos_i32.x.simd_ge(Simd::splat(0))
            & shadow_pos_i32.x.simd_le(Simd::splat(self.shadow_map.width as i32))
            & shadow_pos_i32.y.simd_ge(Simd::splat(0))
            & shadow_pos_i32.y.simd_le(Simd::splat(self.shadow_map.height as i32));
        let idx = shadow_pos_i32.y * Simd::splat(self.shadow_map.stride as i32) + shadow_pos_i32.x;
        let shadow_map = self.shadow_map.as_slice();

        let bias_unit = 1. / self.shadow_map.width as f32;
        let bias = 5. * bias_unit;
        let values =
            Simd::gather_select(shadow_map, in_bounds.cast(), idx.cast(), Simd::splat(f32::MIN)) + Simd::splat(bias);
        let lit_mask = shadow_pos.z.simd_le(values);

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

        let [u, v] = attrs.uv.to_array();
        let x = (u * Simd::splat(self.texture.width as f32)).cast::<i32>();
        let y = ((Simd::splat(1.0) - v) * Simd::splat(self.texture.height as f32)).cast::<i32>();
        let texture_idx = (y * Simd::splat(self.texture.width as i32) + x).cast::<usize>();

        let texture_color = {
            Vec3xN::from([
                Simd::from(std::array::from_fn(|lane| {
                    let idx = texture_idx.as_array()[lane];
                    if mask.test(lane) {
                        self.texture.as_slice()[idx][0] as f32 / 255.
                    } else {
                        0.
                    }
                })),
                Simd::from(std::array::from_fn(|lane| {
                    let idx = texture_idx.as_array()[lane];
                    if mask.test(lane) {
                        self.texture.as_slice()[idx][1] as f32 / 255.
                    } else {
                        0.
                    }
                })),
                Simd::from(std::array::from_fn(|lane| {
                    let idx = texture_idx.as_array()[lane];
                    if mask.test(lane) {
                        self.texture.as_slice()[idx][2] as f32 / 255.
                    } else {
                        0.
                    }
                })),
            ])
        };

        let color = (ambient.splat()
            + (diffuse + specular).map_3(|el| lit_mask.select(el, Simd::splat(0.0)))).element_mul(texture_color);
        // let color = texture_color;

        Vec4xN::from([color.x, color.y, color.z, Simd::splat(1.)])
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

impl FragmentShader for DebugLightIntensity {
    type SimdAttr<const LANES: usize> = LitAttributesSimd<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

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

        let view_dir = (self.camera_pos.splat() - attrs.frag_position.xyz()).normalized();
        let halfway_dir = (light_dir + view_dir).normalized();
        let specular_intensity = {
            let mut pow = normal.dot(halfway_dir).simd_max(Simd::splat(0.));
            for _ in 0..64_usize.ilog2() {
                pow *= pow;
            }
            pow
        };

        // Vec4xN::from([specular_intensity, specular_intensity, specular_intensity, Simd::splat(1.)])
        Vec4xN::from([diffuse_intensity, diffuse_intensity, diffuse_intensity, Simd::splat(1.)])
        // Vec4xN::from([specular_intensity, diffuse_intensity, Simd::splat(0.), Simd::splat(1.)])
    }
}