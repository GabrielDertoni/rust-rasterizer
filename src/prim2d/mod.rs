pub mod simd;

use crate::{
    common::*,
    pipeline::Metrics,
    texture::BorrowedMutTexture,
    vec::{Vec, Vec2i, Vec4},
    Attributes, FragmentShader, math::BBox,
};

pub fn draw_triangles<Attr, S>(
    attributes: &[Attr],
    tris: &[[usize; 3]],
    frag_shader: &S,
    mut pixels: BorrowedMutTexture<[u8; 4]>,
    metrics: &mut Metrics,
) where
    Attr: Attributes + Copy,
    S: FragmentShader<Attr>,
{
    let bbox = BBox {
        x: 0,
        y: 0,
        width: pixels.width() as i32,
        height: pixels.height() as i32,
    };

    for &[v0, v1, v2] in tris {
        let (v0, v1, v2) = {
            let sign = orient_2d(
                attributes[v0].position().xy(),
                attributes[v1].position().xy(),
                attributes[v2].position().xy(),
            );
            if sign > 0.0 {
                (v0, v1, v2)
            } else {
                (v2, v1, v0)
            }
        };

        let v0 = attributes[v0];
        let v1 = attributes[v1];
        let v2 = attributes[v2];

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
            pixels.borrow_mut(),
            bbox,
            &mut *metrics,
        );
    }
}

/// Draws a single triangle with bounding box filling method, but without using SIMD instructions explicitly.
/// - `pixels` is a texture in sRGBA format.
/// - `depth_buf` is optional, so disabling depth_buffer writes is as simple as providing `Option::None`.
#[inline(always)]
fn draw_triangle<Attr, F>(
    // Accessing `vi.position()` must return a `Vec4` where xyz are in NDC and w is 1/z.
    v0: Attr,
    v1: Attr,
    v2: Attr,
    frag_shader: &F,
    mut pixels: BorrowedMutTexture<[u8; 4]>,
    aabb: BBox<i32>,
    metrics: &mut Metrics,
) where
    Attr: Attributes,
    F: FragmentShader<Attr>,
{
    let p0_ndc = *v0.position();
    let p1_ndc = *v1.position();
    let p2_ndc = *v2.position();

    let p0i = p0_ndc.xy().to_i32();
    let p1i = p1_ndc.xy().to_i32();
    let p2i = p2_ndc.xy().to_i32();

    let mut min = p0i.min(p1i).min(p2i);
    min.x = min.x.max(aabb.x);
    min.y = min.y.max(aabb.y);
    let max = p0i.max(p1i).max(p2i).min(Vec2i::from([
        aabb.x + aabb.width - 1,
        aabb.y + aabb.height - 1,
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

    let inv_area = 1.0 / tri_area as f32;

    for y in min.y..=max.y {
        let mut w0 = w0_row;
        let mut w1 = w1_row;
        let mut w2 = w2_row;
        for x in min.x..=max.x {
            count_cycles! {
                #[counter(metrics.performance_counters.test_pixel)]

                if (w0 | w1 | w2) >= 0 {
                    count_cycles! {
                        #[counter(metrics.performance_counters.fill_pixel)]

                        let idx = (x as usize, y as usize);
                        let w = Vec::from([w0, w1, w2]).to_f32() * inv_area;

                        let interp = Attributes::interpolate(&v0, &v1, &v2, w);

                        let prev_color = Vec::from(pixels[idx]).to_f32() / 255.;

                        let prev_color_linear = prev_color.xyz().map(|chan| chan * chan);
                        let color_linear = frag_shader.exec(Vec2i::from([x, y]), interp);
                        let alpha = color_linear.w;
                        let blended_linear = prev_color_linear * (1. - alpha) + color_linear.xyz() * alpha;

                        let blended_srgb = blended_linear.map(|chan| chan.sqrt());

                        let final_color = Vec4::from([blended_srgb.x, blended_srgb.y, blended_srgb.z, 1.]).map(|chan| (chan.clamp(0., 1.) * 255.) as u8);
                        pixels[idx] = final_color.to_array();
                    }
                }
            }

            w0 += w0_inc.x;
            w1 += w1_inc.x;
            w2 += w2_inc.x;
        }
        w0_row += w0_inc.y;
        w1_row += w1_inc.y;
        w2_row += w2_inc.y;
    }
    metrics.sum_areas += tri_area as i64;
    metrics.triangles_drawn += 1;
}
