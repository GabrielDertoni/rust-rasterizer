pub(crate) mod common;
pub(crate) mod simd;
// pub(crate) mod scanline;
// pub(crate) mod specialized;

use common::*;

use crate::{
    Attributes, VertexBuf, VertexShader, FragmentShader,
    vec::{Vec, Vec2i, Vec3, Vec4},
    pipeline::Metrics,
    texture::BorrowedMutTexture,
};

/*
pub fn draw_triangles<B, V, S>(
    vert: &B,
    tris: &[[usize; 3]],
    vert_shader: &V,
    frag_shader: &S,
    mut pixels: BorrowedMutTexture<[u8; 4]>,
    mut depth_buf: BorrowedMutTexture<f32>,
    metrics: &mut Metrics,
)
where
    B: VertexBuf + ?Sized,
    V: VertexShader<B::Vertex>,
    V::Output: Copy,
    S: Fn(V::Output) -> Vec4,
{
    assert_eq!(pixels.width(), depth_buf.width());
    assert_eq!(pixels.height(), depth_buf.height());

    let bbox = BBox {
        x: 0,
        y: 0,
        width: pixels.width() as i32,
        height: pixels.height() as i32,
    };

    for &[v0, v1, v2] in tris {
        let (v2, v1, v0) = (v0, v1, v2);

        let attrib0 = process_vertex(vert.index(v0), vert_shader);
        let attrib1 = process_vertex(vert.index(v1), vert_shader);
        let attrib2 = process_vertex(vert.index(v2), vert_shader);

        draw_triangle(
            attrib0,
            attrib1,
            attrib2,
            frag_shader,
            pixels.borrow_mut(),
            depth_buf.borrow_mut(),
            bbox,
            &mut *metrics,
        );
    }
}
*/

pub fn draw_triangles<Attr, S>(
    attributes: &[Attr],
    tris: &[[usize; 3]],
    frag_shader: &S,
    mut pixels: BorrowedMutTexture<[u8; 4]>,
    mut depth_buf: BorrowedMutTexture<f32>,
    metrics: &mut Metrics,
)
where
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
        let (v2, v1, v0) = (v0, v1, v2);

        draw_triangle(
            attributes[v0],
            attributes[v1],
            attributes[v2],
            frag_shader,
            pixels.borrow_mut(),
            depth_buf.borrow_mut(),
            bbox,
            &mut *metrics,
        );
    }
}

#[inline(always)]
fn draw_triangle<Attr, F>(
    // Accessing `vi.position()` must return a `Vec4` where xyz are in NDC and w is 1/z.
    v0: Attr,
    v1: Attr,
    v2: Attr,
    frag_shader: &F,
    // `pixels` must be coalesced into bundles of `LANES`
    // TODO: Maybe these could be `MatrixSliceMut<Simd<u32, LANES>>`, since that's actually how they're used
    mut pixels: BorrowedMutTexture<[u8; 4]>,
    mut depth_buf: BorrowedMutTexture<f32>,
    aabb: BBox<i32>,
    metrics: &mut Metrics,
) where
    Attr: Attributes,
    F: FragmentShader<Attr>,
{
    let width = pixels.width() as i32;
    let height = pixels.height() as i32;

    let p0_ndc = *v0.position();
    let p1_ndc = *v1.position();
    let p2_ndc = *v2.position();

    let p0_screen = ndc_to_screen(p0_ndc.xy(), width as f32, height as f32).to_i32();
    let p1_screen = ndc_to_screen(p1_ndc.xy(), width as f32, height as f32).to_i32();
    let p2_screen = ndc_to_screen(p2_ndc.xy(), width as f32, height as f32).to_i32();

    let mut min = p0_screen.min(p1_screen).min(p2_screen);
    min.x = min.x.max(aabb.x);
    min.y = min.y.max(aabb.y);
    let max = p0_screen.max(p1_screen).max(p2_screen).min(Vec2i::from([
        aabb.x + aabb.width - 1,
        aabb.y + aabb.height - 1,
    ]));

    // 2 times the area of the triangle
    let tri_area = orient_2d(p0_screen, p1_screen, p2_screen);

    let nz = {
        let u = p0_screen - p1_screen;
        let v = p2_screen - p1_screen;
        u.x * v.y - u.y * v.x
    };

    let inside_frustum = is_inside_frustum(p0_ndc.xyz(), p1_ndc.xyz(), p2_ndc.xyz());

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

    let inv_area = 1.0 / tri_area as f32;

    for y in min.y..=max.y {
        let mut w0 = w0_row;
        let mut w1 = w1_row;
        let mut w2 = w2_row;
        for x in min.x..=max.x {
            if (w0 | w1 | w2) >= 0 {
                let (x, y) = (x as usize, y as usize);
                let w = Vec::from([w0, w1, w2]).to_f32() * inv_area;

                let mut w_persp = w.element_mul(Vec3::from([p0_ndc.w, p1_ndc.w, p2_ndc.w]));
                w_persp /= w_persp.x + w_persp.y + w_persp.z;

                let z = w.dot(Vec::from([p0_ndc.z, p1_ndc.z, p2_ndc.z]));
                let prev_depth = depth_buf[(x, y)];

                if z < prev_depth {
                    let interp = Attributes::interpolate_basic(&v0, &v1, &v2, w_persp);
                    let prev_color = Vec::from(pixels[(x, y)]).to_f32() / 255.;
                    let color = frag_shader.exec_basic(interp);
                    let alpha = color.w;
                    if alpha > 0.1 {
                        let blended = prev_color.xyz() * (1. - alpha) + color.xyz() * alpha;
                        let final_color = Vec4::from([blended.x, blended.y, blended.z, 1.]).map(|chan| (chan.clamp(0., 1.) * 255.) as u8);
                        pixels[(x, y)] = final_color.to_array();
                        depth_buf[(x, y)] = z;
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

#[inline(always)]
fn orient_2d_step(
    from: Vec2i,
    to: Vec2i,
    p: Vec2i,
) -> (Vec2i, i32) {
    let u = to - from;
    let c = u.y * from.x - u.x * from.y;
    let w = u.x * p.y - u.y * p.x + c;
    let inc = Vec2i::from([-u.y, u.x]);
    (inc, w)
}