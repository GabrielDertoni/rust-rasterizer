use std::simd::{Simd, SimdPartialOrd};

use crate::{
    Attributes, FragmentShader, IntoSimd, ScreenPos,
    vec::{Vec, Vec2i, Vec3},
    pipeline::Metrics,
    texture::{BorrowedMutTexture, Dim},
    simd_config::LANES,
};

use super::common::*;

const STEP_X: i32 = 4;
const STEP_Y: i32 = 1;
const X_OFF: Simd<i32, LANES> = Simd::from_array([0, 1, 2, 3]);
const Y_OFF: Simd<i32, LANES> = Simd::from_array([0, 0, 0, 0]);

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
fn draw_triangle<VertOut, S>(
    // Accessing `vi.position()` must return a `Vec4` where xyz are in NDC and w is 1/z.
    v0: VertOut,
    v1: VertOut,
    v2: VertOut,
    frag_shader: &S,
    // `pixels` must be coalesced into bundles of `LANES`
    // TODO: Maybe these could be `MatrixSliceMut<Simd<u32, LANES>>`, since that's actually how they're used
    mut pixels: BorrowedMutTexture<u32>,
    mut depth_buf: BorrowedMutTexture<f32>,
    aabb: BBox<i32>,
    metrics: &mut Metrics,
) where
    VertOut: Attributes,
    S: FragmentShader<VertOut>,
{
    let stride = pixels.indexer().stride().to_usize();
    let width = pixels.width() as i32;
    let height = pixels.height() as i32;
    let pixels = pixels.as_slice_mut();
    let depth_buf = depth_buf.as_slice_mut();

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
    let inside_frustum = (p0_ndc.z >= -1. || p1_ndc.z >= -1. || p2_ndc.z >= -1.) && (p0_ndc.z <= 1. && p1_ndc.z <= 1. && p2_ndc.z <= 1.);

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
    for y in (min.y..=max.y).step_by(STEP_Y as usize) {
        let mut w0 = w0_row;
        let mut w1 = w1_row;
        let mut w2 = w2_row;
        let mut idx = row_start;
        for x in (min.x..=max.x).step_by(STEP_X as usize) {
            let mask = (w0 | w1 | w2).simd_ge(Simd::splat(0));

            if mask.any() {
                let w = Vec::from([w0, w1, w2]).to_f32() * inv_area;

                let mut w_persp = w.element_mul(Vec3::from([p0_ndc.w, p1_ndc.w, p2_ndc.w]).splat());
                w_persp /= w_persp.x + w_persp.y + w_persp.z;

                let z = w.dot(Vec::from([p0_ndc.z, p1_ndc.z, p2_ndc.z]).splat());
                let prev_depth = unsafe {
                    let ptr = &mut depth_buf[idx] as *const f32;
                    *ptr.cast::<Simd<f32, LANES>>()
                };
                let mut mask = mask & z.simd_lt(prev_depth);

                if mask.any() {
                    let interp = Attributes::interpolate(&v0, &v1, &v2, w_persp);
                    let simd_pixels =
                        unsafe { &mut *(&mut pixels[idx] as *mut u32).cast::<Simd<u32, LANES>>() };
                    let pixel_coords = Vec::from([Simd::splat(x), Simd::splat(y)])
                        + Vec::from([Simd::from([0, 1, 2, 3]), Simd::from([0, 0, 0, 0])]);
                    frag_shader.exec_specialized(&mut mask, interp, pixel_coords, simd_pixels);

                    let new_depth = mask.select(z, prev_depth);
                    unsafe {
                        let ptr = &mut depth_buf[idx] as *mut f32;
                        *ptr.cast::<Simd<f32, LANES>>() = new_depth;
                    }
                }
            }

            w0 += w0_inc.x;
            w1 += w1_inc.x;
            w2 += w2_inc.x;

            idx += STEP_X as usize;
        }
        w0_row += w0_inc.y;
        w1_row += w1_inc.y;
        w2_row += w2_inc.y;

        row_start += stride * STEP_Y as usize;
    }
    metrics.sum_areas += tri_area as i64;
    metrics.triangles_drawn += 1;
}

/*
pub fn draw_triangles<B, V, S>(
    vert: &B,
    tris: &[[usize; 3]],
    vert_shader: &V,
    frag_shader: &S,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
    ctx: &mut RenderContext,
) -> Metrics
where
    B: VertexBuf + ?Sized,
    V: VertexShader<B::Vertex>,
    V::Output: Copy,
    S: FragmentShader<V::Output>,
{
    assert_eq!(pixels.width, depth_buf.width);
    assert_eq!(pixels.height, depth_buf.height);

    assert!(pixels
        .as_ptr()
        .is_aligned_to(std::mem::align_of::<Simd<u32, LANES>>()));
    assert!(depth_buf
        .as_ptr()
        .is_aligned_to(std::mem::align_of::<Simd<f32, LANES>>()));

    // Coalesce the pixel layout to be [rrrrggggbbbbaaaa] repeating
    assert_eq!(pixels.width % LANES, 0, "width must be a multiple of LANES");
    assert_eq!(depth_buf.stride % LANES, 0);
    assert!(pixels
        .as_ptr()
        .is_aligned_to(std::mem::align_of::<Vec<Simd<u8, 4>, 4>>()));
    {
        let pixels = unsafe {
            let ptr = pixels.as_mut_ptr().cast::<Vec<Simd<u8, 4>, 4>>();
            std::slice::from_raw_parts_mut(ptr, pixels.size() >> 2)
        };
        for el in pixels {
            *el = el.simd_transpose_4();
        }
    }

    // Run the vertex shader and cache the results
    // let vertex_attrib = vertex_processing(vert, vert_shader, &mut *ctx);
    let mut metrics = Metrics::new();

    let bbox = BBox {
        x: 0,
        y: 0,
        width: pixels.width as i32,
        height: pixels.height as i32,
    };

    let mut pixels_u32: MatrixSliceMut<u32> = unsafe { pixels.borrow().cast() };

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
            pixels_u32.borrow(),
            depth_buf.borrow(),
            bbox,
            &mut metrics,
        );
    }

    // Restore to original pixel layout
    {
        let pixels = unsafe {
            let ptr = pixels.as_mut_ptr().cast::<Vec<Simd<u8, 4>, 4>>();
            std::slice::from_raw_parts_mut(ptr, pixels.size() >> 2)
        };
        for el in pixels {
            *el = el.simd_transpose_4();
        }
    }
    metrics
}
*/

pub fn draw_triangles<Attr, S>(
    attr: &[Attr],
    tris: &[[usize; 3]],
    frag_shader: &S,
    mut color_buf: BorrowedMutTexture<u32>,
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
        width: color_buf.width() as i32,
        height: color_buf.height() as i32,
    };

    for &[v0, v1, v2] in tris {
        let (v2, v1, v0) = (v0, v1, v2);

        draw_triangle(
            attr[v0],
            attr[v1],
            attr[v2],
            frag_shader,
            color_buf.borrow_mut(),
            depth_buf.borrow_mut(),
            bbox,
            &mut *metrics,
        );
    }
}

/*
pub fn draw_triangles_depth<B, V>(
    vert: &B,
    tris: &[[usize; 3]],
    vert_shader: &V,
    mut depth_buf: MatrixSliceMut<f32>,
    ctx: &mut RenderContext,
) where
    B: VertexBuf,
    V: VertexShader<B::Vertex, Output = Vec4>,
{
    // Run the vertex shader and cache the results
    let vertex_attrib = vertex_processing(vert, vert_shader, &mut *ctx);

    let bbox = BBox {
        x: 0,
        y: 0,
        width: depth_buf.width as i32,
        height: depth_buf.height as i32,
    };

    for &[v0, v1, v2] in tris {
        let (v2, v1, v0) = (v0, v1, v2);
        draw_triangle_depth(
            vertex_attrib[v0],
            vertex_attrib[v1],
            vertex_attrib[v2],
            depth_buf.borrow(),
            bbox,
        );
    }
}
*/

#[inline(always)]
fn is_triangle_visible(
    p0_screen: Vec2i,
    p1_screen: Vec2i,
    p2_screen: Vec2i,
    z0: f32,
    z1: f32,
    z2: f32,
    aabb: BBox<i32>,
) -> bool {
    let inside_frustum = is_inside_frustum_screen(
        p0_screen,
        p1_screen,
        p2_screen,
        z0,
        z1,
        z2,
        aabb.width,
        aabb.height,
    );

    // Compute the normal `z` coordinate for backface culling
    let nz = {
        let u = p0_screen - p1_screen;
        let v = p2_screen - p1_screen;
        u.x * v.y - u.y * v.x
    };

    inside_frustum && nz < 0
}


/*
fn triangle_bbox(v0: Vec2i, v1: Vec2i, v2: Vec2i) -> BBox<i32> {
    let min = v0.min(v1).min(v2);
    let max = v0.max(v1).max(v2);
    BBox {
        x: min.x,
        y: min.y,
        width: max.x - min.x,
        height: max.y - min.y,
    }
}

pub fn draw_triangles_threads<B, V, S>(
    vert: &B,
    tris: &[[usize; 3]],
    vert_shader: &V,
    frag_shader: &S,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
    ctx: &mut RenderContext,
) where
    B: VertexBuf + ?Sized,
    V: VertexShader<B::Vertex>,
    V::Output: Copy,
    S: FragmentShader<SimdAttr<4> = <V::Output as Attributes>::Simd<4>>,
{
    assert_eq!(pixels.width, depth_buf.width);
    assert_eq!(pixels.height, depth_buf.height);

    assert!(pixels.as_ptr().is_aligned_to(std::mem::align_of::<Simd<u32, LANES>>()));
    assert!(depth_buf.as_ptr().is_aligned_to(std::mem::align_of::<Simd<f32, LANES>>()));

    // Coalesce the pixel layout to be [rrrrggggbbbbaaaa] repeating
    assert_eq!(pixels.width % 8, 0, "width must be a multiple of 8");
    assert!(pixels
        .as_ptr()
        .is_aligned_to(std::mem::align_of::<Vec<Simd<u8, 4>, 4>>()));
    {
        let pixels = unsafe {
            let ptr = pixels.as_mut_ptr().cast::<Vec<Simd<u8, 4>, 4>>();
            std::slice::from_raw_parts_mut(ptr, pixels.size() >> 2)
        };
        for el in pixels {
            *el = el.simd_transpose_4();
        }
    }

    // Run the vertex shader and cache the results
    ctx.vertex_attrib.reset();
    let mut vertex_attrib = bumpalo::collections::Vec::new_in(&ctx.vertex_attrib);
    vertex_attrib.extend(
        (0..vert.len())
            .map(|i| vert_shader.exec(vert.index(i)))
    );

    let mut metrics = Metrics::new();

    let bbox = BBox {
        x: 0,
        y: 0,
        width: pixels.width as i32,
        height: pixels.height as i32,
    };

    let mut pixels_u32: MatrixSliceMut<u32> = unsafe { pixels.borrow().cast() };

    for &[v0, v1, v2] in tris {
        let (v2, v1, v0) = (v0, v1, v2);

        draw_triangle(
            vertex_attrib[v0],
            vertex_attrib[v1],
            vertex_attrib[v2],
            frag_shader,
            pixels_u32.borrow(),
            depth_buf.borrow(),
            bbox,
            &mut metrics
        );
    }

    // Restore to original pixel layout
    {
        let pixels = unsafe {
            let ptr = pixels.as_mut_ptr().cast::<Vec<Simd<u8, 4>, 4>>();
            std::slice::from_raw_parts_mut(ptr, pixels.size() >> 2)
        };
        for el in pixels {
            *el = el.simd_transpose_4();
        }
    }

    println!("{metrics}");
}
*/

/*
pub struct LineIter {
    x0: i32,
    y0: i32,
    z0: f32,
    inner: LineToIter,
}

impl LineIter {
    pub fn new((x0, y0, z0): ScreenPos, (x1, y1, z1): ScreenPos) -> Self {
        LineIter {
            x0,
            y0,
            z0,
            inner: LineToIter::new(x1 - x0, y1 - y0, z1 - z0),
        }
    }
}

impl Iterator for LineIter {
    type Item = ScreenPos;

    fn next(&mut self) -> Option<ScreenPos> {
        let (x, y, z) = self.inner.next()?;
        Some((x + self.x0, y + self.y0, z + self.z0))
    }
}

/// Line from (0, 0, 0) to (x, y, z)
pub struct LineToIter {
    x: i32,
    dx: i32,
    dy: i32,
    dz: f32,
    // Should swap x and y before yielding
    swap: bool,
}

impl LineToIter {
    pub fn new(x: i32, y: i32, z: f32) -> Self {
        if x.abs() >= y.abs() {
            // x will change on every iteration, but y will only change in some of them.
            LineToIter {
                x: 0,
                dx: x,
                dy: y,
                dz: z,
                swap: false,
            }
        } else {
            LineToIter {
                x: 0,
                dx: y,
                dy: x,
                dz: z,
                swap: true,
            }
        }
    }
}

impl Iterator for LineToIter {
    type Item = ScreenPos;

    fn next(&mut self) -> Option<ScreenPos> {
        let inc = self.dx.signum();
        if inc != 0 && (self.x - self.dx).abs() > 0 {
            let x = self.x;
            self.x += inc;
            let y = x * self.dy / self.dx;
            let z = x as f32 * self.dz / self.dx as f32;
            if self.swap {
                Some((y, x, z))
            } else {
                Some((x, y, z))
            }
        } else {
            None
        }
    }
}
*/

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
