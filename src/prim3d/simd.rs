use std::simd::{Simd, SimdPartialOrd};

use crate::{
    common::*,
    config::CullingMode,
    math_utils::simd_clamp01,
    pipeline::Metrics,
    simd_config::*,
    texture::BorrowedMutTexture,
    vec::{Vec, Vec2i, Vec3},
    Attributes, AttributesSimd, FragmentShaderSimd, IntoSimd, ScreenPos, math::BBox,
};

pub fn draw_triangles<Attr, S>(
    attr: &[Attr],
    tris: &[[usize; 3]],
    frag_shader: &S,
    mut color_buf: BorrowedMutTexture<u32>,
    mut depth_buf: BorrowedMutTexture<f32>,
    bbox: BBox<i32>,
    culling: CullingMode,
    alpha_clip: Option<f32>,
    metrics: &mut Metrics,
) where
    Attr: Attributes + AttributesSimd<LANES> + Copy,
    S: FragmentShaderSimd<Attr, LANES>,
{
    let alpha_clip = alpha_clip.map(|alpha_clip| Simd::splat(alpha_clip));

    for &[v0, v1, v2] in tris {
        let (v0, v1, v2) = match culling {
            CullingMode::FrontFace => (v0, v1, v2),
            CullingMode::BackFace => (v2, v1, v0),
            CullingMode::Disabled => {
                let sign = orient_2d(
                    attr[v0].position().xy(),
                    attr[v1].position().xy(),
                    attr[v2].position().xy(),
                );
                if sign > 0.0 {
                    (v0, v1, v2)
                } else {
                    (v2, v1, v0)
                }
            }
        };

        let v0 = attr[v0];
        let v1 = attr[v1];
        let v2 = attr[v2];

        let inside_frustum = (-1.0..1.0).contains(&v0.position().z)
            && (-1.0..1.0).contains(&v1.position().z)
            && (-1.0..1.0).contains(&v2.position().z);

        if !inside_frustum {
            metrics.behind_culled += 1;
            continue;
        }

        draw_triangle(
            v0,
            v1,
            v2,
            frag_shader,
            color_buf.borrow_mut(),
            depth_buf.borrow_mut(),
            bbox,
            alpha_clip,
            &mut *metrics,
        );
    }
}

#[inline(always)]
fn draw_triangle<VertOut, S>(
    // Accessing `vi.position()` must return a `Vec4` where xy are in viewport, z is in NDC and w is 1/z.
    v0: VertOut,
    v1: VertOut,
    v2: VertOut,
    frag_shader: &S,
    // `pixels` must be coalesced into bundles of `LANES`
    // TODO: Maybe these could be `MatrixSliceMut<Simd<u32, LANES>>`, since that's actually how they're used
    mut pixels: BorrowedMutTexture<u32>,
    mut depth_buf: BorrowedMutTexture<f32>,
    bbox: BBox<i32>,
    alpha_clip: Option<Simd<f32, LANES>>,
    metrics: &mut Metrics,
) where
    VertOut: Attributes + AttributesSimd<LANES>,
    S: FragmentShaderSimd<VertOut, LANES>,
{
    let stride = pixels.indexer().stride();

    let p0_ndc = *v0.position();
    let p1_ndc = *v1.position();
    let p2_ndc = *v2.position();

    let p0i = p0_ndc.xy().to_i32();
    let p1i = p1_ndc.xy().to_i32();
    let p2i = p2_ndc.xy().to_i32();

    let mut min = p0i.min(p1i).min(p2i);
    let mask = LANES as i32 - 1;
    if min.x & mask != 0 {
        min.x = min.x & !mask;
    }
    min.x = min.x.max(bbox.x);
    min.y = min.y.max(bbox.y);
    let max = p0i.max(p1i).max(p2i).min(Vec2i::from([
        bbox.x + bbox.width - (STEP_X - 1),
        bbox.y + bbox.height - (STEP_Y - 1),
    ]));

    // 2 times the area of the triangle
    let tri_area = orient_2d(p0i, p1i, p2i);

    let nz = {
        let u = p0i - p1i;
        let v = p2i - p1i;
        u.x * v.y - u.y * v.x
    };

    if tri_area <= 0 || nz >= 0 {
        if nz >= 0 {
            metrics.backfaces_culled += 1;
        }
        return;
    }

    let (w0_inc, mut w0_row) = orient_2d_step(p1i, p2i, min);
    let (w1_inc, mut w1_row) = orient_2d_step(p2i, p0i, min);
    let (w2_inc, mut w2_row) = orient_2d_step(p0i, p1i, min);

    let inv_area = Simd::splat(1.0 / tri_area as f32);

    let mut row_start = (min.y - bbox.y) as usize * stride + (min.x - bbox.x) as usize;

    count_cycles! {
        #[counter(metrics.performance_counters.process_pixel, increment = ((max.x - min.x + 1) * (max.y - min.y + 1)) as u64)]

        let mut y = min.y;
        while y < max.y {
            let mut w0 = w0_row;
            let mut w1 = w1_row;
            let mut w2 = w2_row;
            let mut idx = row_start;
            let mut x = min.x;
            while x < max.x {
                let mask = (w0 | w1 | w2).simd_ge(Simd::splat(0));

                if mask.any() {
                    let w = Vec::from([w0, w1, w2]).to_f32() * inv_area;

                    let mut w_persp = w.element_mul(Vec3::from([p0_ndc.w, p1_ndc.w, p2_ndc.w]).splat());
                    w_persp /= w_persp.x + w_persp.y + w_persp.z;

                    let z = w.dot(Vec::from([p0_ndc.z, p1_ndc.z, p2_ndc.z]).splat());
                    let prev_depth = unsafe {
                        let ptr = depth_buf.as_ptr().add(idx);
                        *ptr.cast::<Simd<f32, LANES>>()
                    };
                    let mut mask = mask & z.simd_lt(prev_depth);

                    if mask.any() {
                        // TODO: This is wrong! This will also interpolate the NDC coordinates, which it shouldn't. The w coordinate should be interpolated using
                        // 2D barycentrics instead of the perspective correct version.
                        let interp = AttributesSimd::interpolate(&v0, &v1, &v2, w_persp);
                        let simd_pixels =
                            unsafe { &mut *pixels.as_mut_ptr().add(idx).cast::<SimdColorGamma>() };
                        let pixel_coords = Vec::from([Simd::splat(x), Simd::splat(y)]) + Vec::from([X_OFF, Y_OFF]);

                        let frag_color = frag_shader.exec(mask, pixel_coords, interp);
                        blend_pixels(frag_color, &mut mask, simd_pixels, alpha_clip);

                        let new_depth = mask.select(z, prev_depth);
                        unsafe {
                            let ptr = depth_buf.as_mut_ptr().add(idx);
                            *ptr.cast::<Simd<f32, LANES>>() = new_depth;
                        }
                    }
                }

                w0 += w0_inc.x;
                w1 += w1_inc.x;
                w2 += w2_inc.x;

                idx += STEP_X as usize;
                x += STEP_X;
            }
            w0_row += w0_inc.y;
            w1_row += w1_inc.y;
            w2_row += w2_inc.y;

            row_start += stride * STEP_Y as usize;
            y += STEP_Y;
        }
    }
    // dbg!((tri_area / 2, min, max));
    metrics.sum_areas += tri_area as i64;
    metrics.triangles_drawn += 1;
}

#[inline(always)]
fn blend_pixels(
    frag_output: SimdVec4,
    mask: &mut SimdMask,
    pixels: &mut SimdColorGamma,
    alpha_clip: Option<SimdF32>,
) {
    let alpha = frag_output.w;
    let prev_color = pixels.map(|chan| chan.cast::<f32>() / Simd::splat(255.));
    let blended = frag_output * alpha + prev_color * (Simd::splat(1.) - alpha);
    let blended_srgb = blended.map(|chan| (simd_clamp01(chan) * Simd::splat(255.)).cast::<u8>());

    if let Some(alpha_clip) = alpha_clip {
        *mask = *mask & alpha.simd_gt(alpha_clip);
    }

    // Casting the mask to i8, makes the mask structure have 8x4=32 bits. Since -1 represents true
    // in the mask, and bits are stored in twos-compliment, that is a bitset with only 1s when true
    // If we then convert the mask to u32, we'll have a mask for the pixels. We just broadcast this
    // to every channel and mask the things we want.
    // let mask = mask.cast::<i8>();
    // let mask = Simd::splat(u32::from_ne_bytes(mask.to_int().cast().to_array()));

    // *pixels = (blended_srgb & mask) + (*pixels & !mask);

    *pixels = blended_srgb.zip_with(*pixels, |c, p| mask.cast().select(c, p));
}

// (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
// u.x         * (p.y - a.y) - u.y         * (p.x - a.x)
// u.x * p.y - u.x * a.y - (u.y * p.x - u.y * a.x);
// u.x * p.y - u.x * a.y + u.y * a.x - u.y * p.x;
// u.x * p.y - B         + C         - u.y * p.x;
// u.x * p.y + (-B)      + C         - u.y * p.x;
// u.x * p.y +        (C - B)        - u.y * p.x;
// u.x * p.y +           D           - u.y * p.x;
// u.x * p.y - u.y * p.x + D;
#[inline(always)]
fn orient_2d_step(
    from: Vec2i,
    to: Vec2i,
    p: Vec2i,
) -> (Vec<Simd<i32, LANES>, 2>, Simd<i32, LANES>) {
    let u = to - from;
    let c = u.y * from.x - u.x * from.y;
    let p = p.splat() + Vec::from([X_OFF, Y_OFF]);
    let w = Simd::splat(u.x) * p.y - Simd::splat(u.y) * p.x + Simd::splat(c);
    let inc = Vec2i::from([-u.y * STEP_X, u.x * STEP_Y]).splat();
    (inc, w)
}

#[inline(always)]
fn is_triangle_visible(p0_screen: Vec3, p1_screen: Vec3, p2_screen: Vec3, aabb: BBox<f32>) -> bool {
    let inside_frustum =
        is_inside_frustum_screen(p0_screen, p1_screen, p2_screen, aabb.width, aabb.height);

    // Compute the normal `z` coordinate for backface culling
    let nz = {
        let u = p0_screen - p1_screen;
        let v = p2_screen - p1_screen;
        u.x * v.y - u.y * v.x
    };

    inside_frustum && nz < 0.0
}

#[inline(always)]
#[allow(dead_code)]
fn barycentric_coords(
    (x, y): (f32, f32),
    (x1, y1): (f32, f32),
    (x2, y2): (f32, f32),
    (x3, y3): (f32, f32),
) -> Vec3 {
    // source: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    let det = (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
    let lambda_1 = ((y3 - y2) * (x - x3) + (x3 - x2) * (y - y3)) / det;
    let lambda_2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det;
    let lambda_3 = 1.0 - lambda_1 - lambda_2;
    Vec3::from([lambda_1, lambda_2, lambda_3])
}

#[inline(always)]
#[allow(dead_code)]
fn depth_barycentric(l: Vec3, z0: f32, z1: f32, z2: f32) -> f32 {
    z0 * l.x + z1 * l.y + z2 * l.z
}

#[inline(always)]
#[allow(dead_code)]
fn triangle_depth((x, y): (i32, i32), p0: ScreenPos, p1: ScreenPos, p2: ScreenPos) -> f32 {
    let lambda = barycentric_coords(
        (x as f32, y as f32),
        (p0.0 as f32, p0.1 as f32),
        (p1.0 as f32, p1.1 as f32),
        (p2.0 as f32, p2.1 as f32),
    );
    depth_barycentric(lambda, p0.2, p1.2, p2.2)
}
