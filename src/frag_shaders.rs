use crate::{
    vec::{Mat4x4, Vec, Vec3},
    FragmentShader, SimdAttrs,
};

use std::simd::{LaneCount, Mask, Simd, SupportedLaneCount};

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
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
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
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
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
        _pixel_coords: Vec<Simd<i32, LANES>, 2>,
        attrs: SimdAttrs<LANES>,
    ) -> Vec<Simd<f32, LANES>, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let result = self.local_to_global.splat() * attrs.normal.to_hom();
        Vec::from([result.x, result.y, result.z, Simd::splat(1.0)])
    }
}
