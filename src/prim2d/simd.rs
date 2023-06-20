use std::simd::{Mask, Simd, SimdPartialOrd};

use crate::{
    common::{count_cycles, ndc_to_screen, orient_2d},
    math::BBox,
    pipeline::Metrics,
    simd_config::*,
    texture::BorrowedMutTexture,
    vec::{Vec, Vec2i},
    Attributes, AttributesSimd, FragmentShaderSimd, IntoSimd, math_utils::simd_clamp01,
};

pub fn draw_triangles<Attr, S>(
    attr: &[Attr],
    tris: &[[usize; 3]],
    frag_shader: &S,
    mut color_buf: BorrowedMutTexture<u32>,
    metrics: &mut Metrics,
) where
    Attr: Attributes + AttributesSimd<LANES> + Copy,
    S: FragmentShaderSimd<Attr, LANES>,
{
    let bbox = BBox {
        x: 0,
        y: 0,
        width: color_buf.width() as i32,
        height: color_buf.height() as i32,
    };

    for &[v0, v1, v2] in tris {
        let (v0, v1, v2) = {
            let sign = orient_2d(
                attr[v0].position().xy(),
                attr[v1].position().xy(),
                attr[v2].position().xy(),
            );
            if sign > 0.0 {
                (v2, v1, v0)
            } else {
                (v0, v1, v2)
            }
        };

        draw_triangle(
            attr[v0],
            attr[v1],
            attr[v2],
            frag_shader,
            color_buf.borrow_mut(),
            bbox,
            &mut *metrics,
        );
    }
}

#[inline(always)]
fn draw_triangle<VertOut, S>(
    // Accessing `vi.position()` must return a `Vec4` where xyz are in NDC and w is 1/z.
    v0: VertOut,
    v1: VertOut,
    v2: VertOut,
    frag_shader: &S,
    // `pixels` must be coalesced into bundles of `LANES`
    // TODO: Maybe these could be `MatrixSliceMut<Simd<u32, LANES>>`, since that's actually how they're used
    mut pixels: BorrowedMutTexture<u32>,
    aabb: BBox<i32>,
    metrics: &mut Metrics,
) where
    VertOut: Attributes + AttributesSimd<LANES>,
    S: FragmentShaderSimd<VertOut, LANES>,
{
    let stride = pixels.indexer().stride();
    let width = pixels.width() as i32;
    let height = pixels.height() as i32;
    let pixels = pixels.as_slice_mut();

    let p0_ndc = *v0.position();
    let p1_ndc = *v1.position();
    let p2_ndc = *v2.position();

    let p0_screen = ndc_to_screen(p0_ndc.xy(), width as f32, height as f32).to_i32();
    let p1_screen = ndc_to_screen(p1_ndc.xy(), width as f32, height as f32).to_i32();
    let p2_screen = ndc_to_screen(p2_ndc.xy(), width as f32, height as f32).to_i32();

    let mut min = p0_screen.min(p1_screen).min(p2_screen);
    min.x = min.x.next_multiple_of(-(LANES as i32)).max(aabb.x);
    min.y = min.y.max(aabb.y);
    let max = p0_screen.max(p1_screen).max(p2_screen).min(Vec2i::from([
        aabb.x + aabb.width - STEP_X,
        aabb.y + aabb.height - STEP_Y,
    ]));

    // 2 times the area of the triangle
    let tri_area = orient_2d(p0_screen, p1_screen, p2_screen);

    let nz = {
        let u = p0_screen - p1_screen;
        let v = p2_screen - p1_screen;
        u.x * v.y - u.y * v.x
    };

    // let inside_frustum = is_inside_frustum(p0_ndc.xyz(), p1_ndc.xyz(), p2_ndc.xyz());
    let inside_frustum = (p0_ndc.z >= -1. || p1_ndc.z >= -1. || p2_ndc.z >= -1.)
        && (p0_ndc.z <= 1. && p1_ndc.z <= 1. && p2_ndc.z <= 1.);

    if tri_area <= 0 || nz >= 0 || !inside_frustum || min.x == max.x || min.y == max.y {
        if !inside_frustum {
            metrics.behind_culled += 1;
        }
        if nz >= 0 {
            metrics.backfaces_culled += 1;
        }
        return;
    }

    let (w0_inc, mut w0_row) = orient_2d_step(p1_screen, p2_screen, min);
    let (w1_inc, mut w1_row) = orient_2d_step(p2_screen, p0_screen, min);
    let (w2_inc, mut w2_row) = orient_2d_step(p0_screen, p1_screen, min);

    let inv_area = Simd::splat(1.0 / tri_area as f32);

    let mut row_start = min.y as usize * stride + min.x as usize;
    let mut y = min.y;
    while y < max.y {
        let mut w0 = w0_row;
        let mut w1 = w1_row;
        let mut w2 = w2_row;
        let mut idx = row_start;
        let mut x = min.x;
        while x < max.x {
            let mask = (w0 | w1 | w2).simd_ge(Simd::splat(0));

            count_cycles! {
                #[counter(metrics.performance_counters.test_pixel)]

                if mask.any() {
                    count_cycles! {
                        #[counter(metrics.performance_counters.fill_pixel)]

                        let w = Vec::from([w0, w1, w2]).to_f32() * inv_area;

                        let interp = AttributesSimd::interpolate(&v0, &v1, &v2, w);
                        let simd_pixels =
                            unsafe { &mut *(&mut pixels[idx] as *mut u32).cast::<Simd<u32, LANES>>() };
                        let pixel_coords = Vec::from([Simd::splat(x), Simd::splat(y)]) + Vec::from([X_OFF, Y_OFF]);

                        let frag_color = frag_shader.exec(mask, pixel_coords, interp);
                        blend_pixels(frag_color, mask, simd_pixels);
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
    metrics.sum_areas += tri_area as i64;
    metrics.triangles_drawn += 1;
}

#[inline(always)]
fn blend_pixels(frag_output: SimdVec4, mask: Mask<i32, LANES>, pixels: &mut Simd<u32, LANES>) {
    let alpha = frag_output.w;
    let frag_output =
        frag_output.map(|chan| (simd_clamp01(chan) * Simd::splat(255.)).cast::<u32>());
    let colors = frag_output.x
        | (frag_output.y << Simd::splat(8))
        | (frag_output.z << Simd::splat(16))
        | (frag_output.w << Simd::splat(24));
    *pixels = mask.select(colors, *pixels);
}

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
