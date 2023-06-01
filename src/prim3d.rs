use std::simd::{Simd, SimdPartialOrd};

use crate::buf::{MatrixSliceMut, PixelBuf};
use crate::vec::{Vec, Vec2i, Vec2, Vec3, Vec4};
use crate::{Attributes, FragmentShader, ScreenPos, VertexBuf, VertexShader};

/*
pub fn draw_line(
    p0: Vec4,
    p1: Vec4,
    color: u32,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    let p0 = to_screen_pos(p0);
    let p1 = to_screen_pos(p1);
    for (x, y, z) in LineIter::new(p0, p1) {
        let idx = (x as usize, y as usize);
        if z > depth_buf[idx] {
            depth_buf[idx] = z;
            pixels[idx] = color.to_be_bytes();
        }
    }
}

pub fn draw_triangle_outline(
    p0: Vec4,
    p1: Vec4,
    p2: Vec4,
    color: u32,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    draw_line(p0, p1, color, pixels.borrow(), depth_buf.borrow());
    draw_line(p1, p2, color, pixels.borrow(), depth_buf.borrow());
    draw_line(p0, p2, color, pixels.borrow(), depth_buf.borrow());
}
*/

/// Returns the oriented area of the paralelogram formed by the points `from`, `to`, `p`, `from + (p - to)`. The sign
/// is positive if the points in the paralelogram wind counterclockwise (according to the order given prior) and
/// negative otherwise. In other words, if you were at `from` looking towards `to`, when `p` is to your left, the
/// value would be positive, and if it is to your right the value is negative.
///
/// ## Relationship with barycentric coordinates
///
/// This function's return value has a neat relationship with barycentric coordinates: for any triangle ABC, the barycentric
/// coordinate of a point P, named W has components:
///
/// - `W.x = orient_2d(A, B, P) / orient_2d(A, B, C)`
/// - `W.y = orient_2d(B, C, P) / orient_2d(A, B, C)`
/// - `W.z = orient_2d(C, A, P) / orient_2d(A, B, C)`
///
/// It's also worth noting that `orient_2d(A, B, C)` is twice the area of the triangle ABC.
pub fn orient_2d_i32(from: Vec2i, to: Vec2i, p: Vec2i) -> i32 {
    let u = to - from;
    let v = p - from;
    u.x * v.y - u.y * v.x
}


const LANES: usize = 4;
const STEP_X: i32 = 4;
const STEP_Y: i32 = 1;

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
    let p = p.splat() + Vec::from([Simd::from([0, 1, 2, 3]), Simd::from([0, 0, 0, 0])]);
    let w = Simd::splat(u.x) * p.y - Simd::splat(u.y) * p.x + Simd::splat(c);
    let inc = Vec2i::from([-u.y * STEP_X, u.x * STEP_Y]).splat();
    (inc, w)
}

fn ndc_to_screen(ndc: Vec2, width: f32, height: f32) -> Vec2 {
    Vec2::from([
        ndc.x * width  / 2. + width  / 2.,
        -ndc.y * height / 2. + height / 2.,
    ])
}

#[derive(Clone, Copy, Debug)]
struct BBox<T> {
    x: T,
    y: T,
    width: T,
    height: T,
}

/*
impl<T: Num> BBox<T> {
    pub fn intersects(&self, other: BBox<T>) -> bool {
        let x_intersection = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
        let y_intersection = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);
        x_intersection > T::zero() && y_intersection > T::zero()
    }
}
*/

// source: https://www.cs.utexas.edu/~fussell/courses/cs354-fall2015/lectures/lecture9.pdf
// This assumes that if all points are outside the frustum, than so is the triangle
fn is_inside_frustum_clip(p0_clip: Vec4, p1_clip: Vec4, p2_clip: Vec4) -> bool {
    let range0 = -p0_clip.w..p0_clip.w;
    let range1 = -p1_clip.w..p1_clip.w;
    let range2 = -p2_clip.w..p2_clip.w;

    (range0.contains(&p0_clip.x) && range0.contains(&p0_clip.y) && range0.contains(&p0_clip.z))
        || (range1.contains(&p1_clip.x) && range1.contains(&p1_clip.y) && range1.contains(&p1_clip.z))
        || (range2.contains(&p2_clip.x) && range2.contains(&p2_clip.y) && range2.contains(&p2_clip.z))
}

fn is_inside_frustum(p0_ndc: Vec3, p1_ndc: Vec3, p2_ndc: Vec3) -> bool {
    ((-1.0..1.0).contains(&p0_ndc.x) && (-1.0..1.0).contains(&p0_ndc.y) && (-1.0..1.0).contains(&p0_ndc.z))
        || ((-1.0..1.0).contains(&p1_ndc.x) && (-1.0..1.0).contains(&p1_ndc.y) && (-1.0..1.0).contains(&p1_ndc.z))
        || ((-1.0..1.0).contains(&p2_ndc.x) && (-1.0..1.0).contains(&p2_ndc.y) && (-1.0..1.0).contains(&p2_ndc.z))
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
    pixels: MatrixSliceMut<u32>,
    depth_buf: MatrixSliceMut<f32>,
    aabb: BBox<i32>,
    metrics: &mut Metrics,
) where
    VertOut: Attributes,
    S: FragmentShader<SimdAttr<4> = <VertOut as Attributes>::Simd<4>>,
{
    let stride = pixels.stride;
    let width = pixels.width as i32;
    let height = pixels.height as i32;
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
    let max = p0_screen
        .max(p1_screen)
        .max(p2_screen)
        .min(Vec2i::from([aabb.x + aabb.width - STEP_X, aabb.y + aabb.height - STEP_Y]));

    // 2 times the area of the triangle
    let tri_area = orient_2d_i32(p0_screen, p1_screen, p2_screen);

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
                let prev_depth =
                    unsafe { *depth_buf.as_ptr().add(idx).cast::<Simd<f32, LANES>>() };
                let mut mask = mask & z.simd_lt(prev_depth);

                if mask.any() {
                    let interp = Attributes::interpolate(&v0, &v1, &v2, w_persp);
                    let simd_pixels = unsafe {
                        &mut *(&mut pixels[idx] as *mut u32).cast::<Simd<u32, LANES>>()
                    };
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
    metrics.triangles_drawn += 1;
}

/// Basically stores preallocated memory that can be reused between draw calls.
pub struct RenderContext {
    vertex_attrib: bumpalo::Bump,
    tiles: bumpalo::Bump,
}

impl RenderContext {
    pub fn alloc(cap: usize) -> Self {
        RenderContext {
            vertex_attrib: bumpalo::Bump::with_capacity(cap),
            tiles: bumpalo::Bump::new(),
        }
    }
}

pub fn draw_triangles<B, V, S>(
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
    assert_eq!(pixels.width % 4, 0, "width must be a multiple of 4");
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
            .map(|i| {
                let mut attrib = vert_shader.exec(vert.index(i));
                let pos = attrib.position_mut();
                let inv_w = 1. / pos.w;
                *pos.xyz_mut() *= inv_w;
                pos.w = inv_w;
                attrib
            })
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

    println!("{metrics}");
}

pub fn draw_triangles_depth<B, V>(
    vert: &B,
    tris: &[[usize; 3]],
    vert_shader: &V,
    depth_buf: MatrixSliceMut<f32>,
    ctx: &mut RenderContext,
) where
    B: VertexBuf,
    V: VertexShader<B::Vertex, Output = Vec4>,
{
    let stride = depth_buf.stride;
    let width = depth_buf.width as i32;
    let height = depth_buf.height as i32;
    let depth_buf = depth_buf.as_slice_mut();

    // Run the vertex shader and cache the results
    ctx.vertex_attrib.reset();
    let mut vertex_attrib = bumpalo::collections::Vec::new_in(&ctx.vertex_attrib);
    vertex_attrib.extend(
        (0..vert.len())
            .map(|i| {
                let mut attrib = vert_shader.exec(vert.index(i));
                let pos = attrib.position_mut();
                let inv_w = 1. / pos.w;
                *pos.xyz_mut() *= inv_w;
                pos.w = inv_w;
                attrib
            })
    );

    for &[v0, v1, v2] in tris {
        let (v2, v1, v0) = (v0, v1, v2);

        let p0_ndc = *vertex_attrib[v0].position();
        let p1_ndc = *vertex_attrib[v1].position();
        let p2_ndc = *vertex_attrib[v2].position();

        let inside_frustum = is_inside_frustum(p0_ndc.xyz(), p1_ndc.xyz(), p2_ndc.xyz());

        let p0_screen = ndc_to_screen(p0_ndc.xy(), width as f32, height as f32).to_i32();
        let p1_screen = ndc_to_screen(p1_ndc.xy(), width as f32, height as f32).to_i32();
        let p2_screen = ndc_to_screen(p2_ndc.xy(), width as f32, height as f32).to_i32();

        let mut min = p0_screen.min(p1_screen).min(p2_screen);
        min.x = min.x.next_multiple_of(-(LANES as i32)).max(0);
        min.y = min.y.max(0);
        let max = p0_screen
            .max(p1_screen)
            .max(p2_screen)
            .min(Vec2i::from([width - STEP_X, height - STEP_Y]));

        // 2 times the area of the triangle
        let tri_area = orient_2d_i32(p0_screen, p1_screen, p2_screen);

        let nz = {
            let u = p0_screen - p1_screen;
            let v = p2_screen - p1_screen;
            u.x * v.y - u.y * v.x
        };

        if tri_area <= 0 || nz >= 0 || !inside_frustum || min.x > max.x || min.y > max.y {
            continue;
        }

        let (w0_inc, mut w0_row) = orient_2d_step(p1_screen, p2_screen, min);
        let (w1_inc, mut w1_row) = orient_2d_step(p2_screen, p0_screen, min);
        let (w2_inc, mut w2_row) = orient_2d_step(p0_screen, p1_screen, min);

        let inv_area = Simd::splat(1.0 / tri_area as f32);

        let bbox_width = (max.x - min.x + 1) as usize;

        let mut row_start = min.y as usize * stride + min.x as usize;
        let end = max.y as usize * stride + min.x as usize;
        while row_start < end {
            let mut w0 = w0_row;
            let mut w1 = w1_row;
            let mut w2 = w2_row;
            let mut idx = row_start;
            let row_end = idx + bbox_width;
            while idx < row_end {
                let mask = (w0 | w1 | w2).simd_ge(Simd::splat(0));
                if mask.any() {
                    let z =
                        w0.cast() * Simd::splat(p0_ndc.z) + w1.cast() * Simd::splat(p1_ndc.z) + w2.cast() * Simd::splat(p2_ndc.z) * inv_area;
                    let prev_depth =
                        unsafe { *depth_buf.as_ptr().add(idx).cast::<Simd<f32, LANES>>() };
                    let mask = mask & z.simd_lt(prev_depth);
                    let new_depth = mask.select(z, prev_depth);
                    unsafe {
                        let ptr = &mut depth_buf[idx] as *mut f32;
                        *ptr.cast::<Simd<f32, LANES>>() = new_depth;
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
    }
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

#[derive(Default, Debug, Clone)]
struct Metrics {
    triangles_drawn: usize,
    backfaces_culled: usize,
    behind_culled: usize,
}

impl Metrics {
    fn new() -> Self {
        Metrics {
            triangles_drawn: 0,
            backfaces_culled: 0,
            behind_culled: 0,
        }
    }
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let Metrics { triangles_drawn, backfaces_culled, behind_culled } = self;
        writeln!(f, "render metrics:")?;
        writeln!(f, "\ttriangles drawn: {triangles_drawn}")?;
        writeln!(f, "\tbackfaces culled: {backfaces_culled}")?;
        writeln!(f, "\tbehind culled: {behind_culled}")?;
        Ok(())
    }
}

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
