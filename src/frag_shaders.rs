use crate::{
    buf,
    vec::{Mat4x4, Vec, Vec2i, Vec3},
    FragmentShader, SimdAttrs,
};

use std::simd::{
    LaneCount, Mask, Simd, SimdFloat, SimdPartialEq, SimdPartialOrd, SupportedLaneCount,
};

pub struct TextureMappingFragShader<'a> {
    texture_width: i32,
    texture_height: i32,
    texture: &'a [u32],
}

impl<'a> TextureMappingFragShader<'a> {
    pub fn new(texture_img: &'a image::RgbaImage) -> Self {
        let texture_width = texture_img.width();
        let texture_height = texture_img.height();
        let texture: &[u8] = &*texture_img;
        let ptr = texture.as_ptr().cast::<u32>();
        assert!(ptr.is_aligned());
        // SAFETY: Pointer is aligned
        let texture = unsafe { std::slice::from_raw_parts(ptr, texture.len() / 4) };
        TextureMappingFragShader {
            texture_width: texture_width as i32,
            texture_height: texture_height as i32,
            texture,
        }
    }
}

impl<'a> FragmentShader for TextureMappingFragShader<'a> {
    type SimdAttr<const LANES: usize> = SimdAttrs<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    #[inline(always)]
    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        _attrs: SimdAttrs<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        panic!("unsupported")
    }

    #[inline(always)]
    fn exec_specialized(
        &self,
        mask: Mask<i32, 4>,
        attrs: Self::SimdAttr<4>,
        _pixel_coords: Vec<Simd<i32, 4>, 2>,
        pixels: &mut Simd<u32, 4>,
    ) {
        let [u, v] = attrs.uv.to_array();
        let x = (u * Simd::splat(self.texture_width as f32)).cast::<i32>();
        let y = ((Simd::splat(1.0) - v) * Simd::splat(self.texture_height as f32)).cast::<i32>();
        let idx = (y * Simd::splat(self.texture_width) + x).cast();

        unsafe {
            *pixels = Simd::gather_select_unchecked(&self.texture, mask.cast(), idx, *pixels);
        }
    }
}

/*
pub struct LinearFilteringFragShader<'a> {
    texture_width: u32,
    texture_height: u32,
    texture: &'a [u32],
}

impl<'a> LinearFilteringFragShader<'a> {
    pub fn new(texture_img: &'a image::RgbaImage) -> Self {
        let texture_width = texture_img.width();
        let texture_height = texture_img.height();
        let texture: &[u8] = &*texture_img;
        let ptr = texture.as_ptr().cast::<u32>();
        assert!(ptr.is_aligned());
        // SAFETY: Pointer is aligned
        let texture = unsafe { std::slice::from_raw_parts(ptr, texture.len() / 4) };
        LinearFilteringFragShader {
            texture_width,
            texture_height,
            texture,
        }
    }
}

impl<'a> FragShader for LinearFilteringFragShader<'a> {
    type SimdAttr<const LANES: usize> = SimdAttr<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        attrs: SimdAttr<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        use std::simd::SimdFloat;

        let [u, v] = attrs.uv.to_array();
        let x = u * Simd::splat(self.texture_width as f32);
        let y = (Simd::splat(1.0) - v) * Simd::splat(self.texture_height as f32);

        let idx0 = y.cast::<usize>() * Simd::splat(self.texture_width as usize) + x.cast::<usize>();
        let idx1 = idx0 + Simd::splat(1);
        let idx2 = idx0 + Simd::splat(self.texture_width as usize);
        let idx3 = idx0 + Simd::splat(self.texture_width as usize + 1);

        let fx = x.cast::<i32>().cast::<f32>();
        let fy = y.cast::<i32>().cast::<f32>();

        let dx = x - fx;
        let dy = y - fy;

        let dist0 = dx.abs().simd_max(dy.abs());
        let dist1 = Simd::splat(1.0) - dx;
        let dist2 = Simd::splat(1.0) - dy;
        let dist3 = dist1.simd_max(dist2);

        let total = dist0 + dist1 + dist2 + dist3;

        let w0 = dist0 / total;
        let w1 = dist1 / total;
        let w2 = dist2 / total;
        let w3 = dist3 / total;

        let c0 = Vec::from(
            Simd::gather_or_default(&self.texture, idx0)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes())),
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0));

        let c1 = Vec::from(
            Simd::gather_or_default(&self.texture, idx1)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes())),
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0));

        let c2 = Vec::from(
            Simd::gather_or_default(&self.texture, idx2)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes())),
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0));

        let c3 = Vec::from(
            Simd::gather_or_default(&self.texture, idx3)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes())),
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0));

        c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3
    }
}
*/

pub struct FakeLitFragShader {
    light_dir: Vec3,
    local_to_global: Mat4x4,
}

impl FakeLitFragShader {
    pub fn new(light_dir: Vec3, model: Mat4x4) -> Self {
        FakeLitFragShader {
            light_dir,
            local_to_global: model.inverse().transpose(),
        }
    }
}

impl FragmentShader for FakeLitFragShader {
    type SimdAttr<const LANES: usize> = SimdAttrs<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        attrs: SimdAttrs<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let n = (self.local_to_global.splat() * attrs.normal.to_hom())
            .xyz()
            .normalized()
            .dot(self.light_dir.splat());
        Vec::from([n, n, n, Simd::splat(1.0)])
    }
}

pub struct LitFragShader<'a> {
    normal_local_to_world: Mat4x4,
    light_pos: Vec3,
    light_transform: Mat4x4,
    light_color: Vec3,
    shadow_map: buf::MatrixSlice<'a, f32>,
}

impl<'a> LitFragShader<'a> {
    pub fn new(
        model: Mat4x4,
        light_pos: Vec3,
        light_transform: Mat4x4,
        light_color: Vec3,
        shadow_map: buf::MatrixSlice<'a, f32>,
    ) -> Self {
        LitFragShader {
            normal_local_to_world: model.inverse().transpose(),
            light_pos,
            light_transform,
            light_color,
            shadow_map,
        }
    }
}

impl<'a> FragmentShader for LitFragShader<'a> {
    type SimdAttr<const LANES: usize> = SimdAttrs<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        _attrs: SimdAttrs<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        panic!("unsupported")
    }

    // source: https://learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
    fn exec_specialized(
        &self,
        mask: Mask<i32, 4>,
        attrs: Self::SimdAttr<4>,
        _pixel_coords: Vec<Simd<i32, 4>, 2>,
        pixels: &mut Simd<u32, 4>,
    ) {
        let shadow_uv = self.light_transform.splat() * attrs.position;
        let shadow_pos = self.shadow_map.ndc_to_screen().splat() * (shadow_uv / shadow_uv.w);

        let idx = shadow_pos.y.cast::<i32>() * Simd::splat(self.shadow_map.stride as i32)
            + shadow_pos.x.cast::<i32>();
        let shadow_map = self.shadow_map.as_slice();

        let values = Simd::gather_or(shadow_map, idx.cast(), Simd::splat(1.0)) + Simd::splat(0.001);
        let lit_mask = shadow_pos.z.simd_le(values);

        let ambient = self.light_color * 0.15;

        let normal = self.normal_local_to_world.splat() * attrs.normal.to_hom();
        let light_dir = (self.light_pos.splat() - attrs.position.xyz()).normalized();
        let diffuse_intensity = light_dir.dot(normal.xyz()).simd_max(Simd::splat(0.));
        let diffuse = diffuse_intensity * self.light_color.splat();

        // TODO
        let specular = Vec3::zero().splat();

        let lighting = ambient.splat()
            + (diffuse + specular).map_3(|el| lit_mask.select(el, Simd::splat(0.0)));

        let color = lighting.map_3(|el| (el * Simd::splat(128.)).cast::<u8>());

        // Add alpha channel and traspose. Here `color.x` is the red values of each of the pixels.
        // Thus, this is in SoA form, but we need it in AoS form in order to write to `pixels`.
        let colors = Vec::from([color.x, color.y, color.z, Simd::splat(0xff)])
            .simd_transpose_4()
            .map_4(|el| u32::from_ne_bytes(el.to_array()));

        *pixels = mask.select(Simd::from(colors.to_array()), *pixels);
    }
}

pub struct ShowNormalsFragShader;

impl ShowNormalsFragShader {
    pub fn new() -> Self {
        ShowNormalsFragShader
    }
}

impl FragmentShader for ShowNormalsFragShader {
    type SimdAttr<const LANES: usize> = SimdAttrs<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        attrs: SimdAttrs<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        use std::simd::SimdFloat;

        let n = attrs.normal.map(|el| el.abs());
        Vec::from([n.x, n.y, n.z, Simd::splat(1.0)])
    }
}

pub struct ShowGlobalNormalsFragShader {
    local_to_global: Mat4x4,
}

impl ShowGlobalNormalsFragShader {
    pub fn new(model: Mat4x4) -> Self {
        ShowGlobalNormalsFragShader {
            local_to_global: model.inverse().transpose(),
        }
    }
}

impl FragmentShader for ShowGlobalNormalsFragShader {
    type SimdAttr<const LANES: usize> = SimdAttrs<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        attrs: SimdAttrs<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let result = self.local_to_global.splat() * attrs.normal.to_hom();
        Vec::from([result.x, result.y, result.z, Simd::splat(1.0)])
    }
}
