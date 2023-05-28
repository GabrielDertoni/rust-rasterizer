use std::simd::Simd;

use crate::buf::{MatrixSliceMut, PixelBuf};
use crate::vec::{Vec, Vec2i, Vec3};
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

type Vec3xX = Vec<Simd<f32, LANES>, 3>;
type IVec2xX = Vec<Simd<i32, LANES>, 2>;

pub fn draw_triangles<B, V, S>(
    vert: &B,
    tris: &[[B::Index; 3]],
    vert_shader: &V,
    frag_shader: &S,
    pixels: PixelBuf,
    depth_buf: MatrixSliceMut<f32>,
) where
    B: VertexBuf,
    V: VertexShader<B::Vertex>,
    S: FragmentShader<SimdAttr<4> = <V::Attrs as Attributes>::Simd<4>>,
{
    use std::simd::SimdPartialOrd;

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
        let p = p.splat() + IVec2xX::from([Simd::from([0, 1, 2, 3]), Simd::from([0, 0, 0, 0])]);
        let w = Simd::splat(u.x) * p.y - Simd::splat(u.y) * p.x + Simd::splat(c);
        let inc = Vec2i::from([-u.y * STEP_X, u.x * STEP_Y]).splat();
        (inc, w)
    }

    let ndc_to_screen = pixels.ndc_to_screen();

    let stride = pixels.stride;
    let width = pixels.width as i32;
    let height = pixels.height as i32;
    let pixels = pixels.as_slice_mut();
    let depth_buf = depth_buf.as_slice_mut();

    for &[v0, v1, v2] in tris {
        let (v2, v1, v0) = (v0, v1, v2);

        let (attrs0, p0) = vert_shader.exec(vert.index(v0));
        let (attrs1, p1) = vert_shader.exec(vert.index(v1));
        let (attrs2, p2) = vert_shader.exec(vert.index(v2));

        let p0 = ndc_to_screen * (p0 / p0.w);
        let p1 = ndc_to_screen * (p1 / p1.w);
        let p2 = ndc_to_screen * (p2 / p2.w);

        let p0i = p0.xy().to_i32();
        let p1i = p1.xy().to_i32();
        let p2i = p2.xy().to_i32();

        let mut min = p0i.min(p1i).min(p2i);
        min.x = min.x.next_multiple_of(-(LANES as i32)).max(0);
        min.y = min.y.max(0);
        let max = p0i
            .max(p1i)
            .max(p2i)
            .min(Vec2i::from([width - STEP_X, height - STEP_Y]));

        // 2 times the area of the triangle
        let tri_area = orient_2d_i32(p0i, p1i, p2i);

        let nz = {
            let u = p0 - p1;
            let v = p2 - p1;
            u.x * v.y - u.y * v.x
        };

        if tri_area <= 0 || nz >= 0.0 {
            continue;
        }

        let (w0_inc, mut w0_row) = orient_2d_step(p1i, p2i, min);
        let (w1_inc, mut w1_row) = orient_2d_step(p2i, p0i, min);
        let (w2_inc, mut w2_row) = orient_2d_step(p0i, p1i, min);

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
                    let w = Vec3xX::from([w0.cast::<f32>(), w1.cast::<f32>(), w2.cast::<f32>()])
                        * inv_area;
                    let z =
                        w.x * Simd::splat(p0.z) + w.y * Simd::splat(p1.z) + w.z * Simd::splat(p2.z);
                    let prev_depth =
                        unsafe { *depth_buf.as_ptr().add(idx).cast::<Simd<f32, LANES>>() };
                    let oldmask = mask;
                    let mask = mask & z.simd_lt(prev_depth) & z.simd_ge(Simd::splat(-1.));
                    let interp = Attributes::interpolate(&attrs0, &attrs1, &attrs2, w);
                    let simd_pixels = unsafe {
                        &mut *(&mut pixels[idx] as *mut [u8; 4]).cast::<Simd<u32, LANES>>()
                    };
                    let pixel_coords = IVec2xX::from([Simd::splat(x), Simd::splat(y)])
                        + IVec2xX::from([Simd::from([0, 1, 2, 3]), Simd::from([0, 0, 0, 0])]);
                    frag_shader.exec_specialized(oldmask, interp, pixel_coords, simd_pixels);

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

pub fn draw_triangles_depth<B, V>(
    vert: &B,
    tris: &[[B::Index; 3]],
    vert_shader: &V,
    depth_buf: MatrixSliceMut<f32>,
) where
    B: VertexBuf,
    V: VertexShader<B::Vertex, Attrs = ()>,
{
    use std::simd::SimdPartialOrd;

    const STEP_X: i32 = 4;
    const STEP_Y: i32 = 1;

    #[inline(always)]
    fn orient_2d_step(
        from: Vec2i,
        to: Vec2i,
        p: Vec2i,
    ) -> (Vec<Simd<i32, LANES>, 2>, Simd<i32, LANES>) {
        let u = to - from;
        let c = u.y * from.x - u.x * from.y;
        let p = p.splat() + IVec2xX::from([Simd::from([0, 1, 2, 3]), Simd::from([0, 0, 0, 0])]);
        let w = Simd::splat(u.x) * p.y - Simd::splat(u.y) * p.x + Simd::splat(c);
        let inc = Vec2i::from([-u.y * STEP_X, u.x * STEP_Y]).splat();
        (inc, w)
    }

    let ndc_to_screen = depth_buf.ndc_to_screen();

    let stride = depth_buf.stride;
    let width = depth_buf.width as i32;
    let height = depth_buf.height as i32;
    let depth_buf = depth_buf.as_slice_mut();

    for &[v0, v1, v2] in tris {
        let (v2, v1, v0) = (v0, v1, v2);

        let (_, p0) = vert_shader.exec(vert.index(v0));
        let (_, p1) = vert_shader.exec(vert.index(v1));
        let (_, p2) = vert_shader.exec(vert.index(v2));

        let p0 = ndc_to_screen * (p0 / p0.w);
        let p1 = ndc_to_screen * (p1 / p1.w);
        let p2 = ndc_to_screen * (p2 / p2.w);

        let p0i = p0.xy().to_i32();
        let p1i = p1.xy().to_i32();
        let p2i = p2.xy().to_i32();

        let mut min = p0i.min(p1i).min(p2i);
        min.x = min.x.next_multiple_of(-(LANES as i32)).max(0);
        min.y = min.y.max(0);
        let max = p0i
            .max(p1i)
            .max(p2i)
            .min(Vec2i::from([width - STEP_X, height - STEP_Y]));

        // 2 times the area of the triangle
        let tri_area = orient_2d_i32(p0i, p1i, p2i);

        let nz = {
            let u = p0 - p1;
            let v = p2 - p1;
            u.x * v.y - u.y * v.x
        };

        if tri_area <= 0 || nz >= 0.0 {
            continue;
        }

        let (w0_inc, mut w0_row) = orient_2d_step(p1i, p2i, min);
        let (w1_inc, mut w1_row) = orient_2d_step(p2i, p0i, min);
        let (w2_inc, mut w2_row) = orient_2d_step(p0i, p1i, min);

        let inv_area = Simd::splat(1.0 / tri_area as f32);

        let mut row_start = min.y as usize * stride + min.x as usize;
        for _y in (min.y..=max.y).step_by(STEP_Y as usize) {
            let mut w0 = w0_row;
            let mut w1 = w1_row;
            let mut w2 = w2_row;
            let mut idx = row_start;
            for _x in (min.x..=max.x).step_by(STEP_X as usize) {
                let mask = (w0 | w1 | w2).simd_ge(Simd::splat(0));
                if mask.any() {
                    let w = Vec3xX::from([w0.cast::<f32>(), w1.cast::<f32>(), w2.cast::<f32>()])
                        * inv_area;
                    let z =
                        w.x * Simd::splat(p0.z) + w.y * Simd::splat(p1.z) + w.z * Simd::splat(p2.z);
                    let prev_depth =
                        unsafe { *depth_buf.as_ptr().add(idx).cast::<Simd<f32, LANES>>() };
                    let mask = mask & z.simd_lt(prev_depth) & z.simd_ge(Simd::splat(-1.));
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
