use crate::buf::{MatrixSliceMut, PixelBuf};
use crate::vec::{Vec, Vec3, Vec4, Vec2i, Vec3i};
use crate::{triangles_iter, ScreenPos};

fn to_screen_pos(v: Vec4) -> ScreenPos {
    (v.x as i32, v.y as i32, v.z)
}

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

pub fn draw_triangle2(
    p0: Vec4,
    p1: Vec4,
    p2: Vec4,
    color: u32,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    let mut p0 = to_screen_pos(p0);
    let mut p1 = to_screen_pos(p1);
    let mut p2 = to_screen_pos(p2);

    if p0 == p1 || p0 == p2 || p1 == p2 {
        return;
    }

    if p1.1 < p0.1 {
        std::mem::swap(&mut p0, &mut p1);
    }

    if p2.1 < p0.1 {
        std::mem::swap(&mut p0, &mut p2);
    }

    if p1.0 > p2.0 {
        std::mem::swap(&mut p1, &mut p2);
    }

    let mut left_dx = p1.0 - p0.0;
    let mut left_dy = p1.1 - p0.1;
    let mut left_coef = if left_dy != 0 {
        p0.0 - p0.1 * left_dx / left_dy
    } else {
        0
    };

    let mut right_dx = p2.0 - p0.0;
    let mut right_dy = p2.1 - p0.1;
    let mut right_coef = if right_dy != 0 {
        p0.0 - p0.1 * right_dx / right_dy
    } else {
        0
    };

    for y in p0.1..p1.1.min(p2.1) {
        let l = left_coef + y * left_dx / left_dy;
        let r = right_coef + y * right_dx / right_dy;
        for x in l..=r {
            if !(x >= 0 && x <= pixels.width as i32 && y >= 0 && y < pixels.height as i32) {
                continue;
            }
            let depth = triangle_depth((x, y), p0, p1, p2);
            let idx = (x as usize, y as usize);
            if depth > depth_buf[idx] {
                depth_buf[idx] = depth;
                pixels[idx] = color.to_be_bytes();
            }
        }
    }

    if p1.1 < p2.1 {
        left_dx = p2.0 - p1.0;
        left_dy = p2.1 - p1.1;
        left_coef = p1.0 - p1.1 * left_dx / left_dy;
    } else if p1.1 > p2.1 {
        right_dx = p1.0 - p2.0;
        right_dy = p1.1 - p2.1;
        right_coef = p2.0 - p2.1 * right_dx / right_dy;
    }

    for y in p1.1.min(p2.1)..p1.1.max(p2.1) {
        let l = left_coef + y * left_dx / left_dy;
        let r = right_coef + y * right_dx / right_dy;
        for x in l..=r {
            if !(x >= 0 && x <= pixels.width as i32 && y >= 0 && y < pixels.height as i32) {
                continue;
            }
            let depth = triangle_depth((x, y), p0, p1, p2);
            let idx = (x as usize, y as usize);
            if depth > depth_buf[idx] {
                depth_buf[idx] = depth;
                pixels[idx] = color.to_be_bytes();
            }
        }
    }
}
*/

pub trait Vertex {
    type Attr;

    fn position(&self) -> &Vec3;
    fn interpolate(w: Vec3, v0: &Self, v1: &Self, v2: &Self) -> Self::Attr;
}

pub fn draw_triangles<V: Vertex>(
    vert: &[V],
    tris: &[[u32; 3]],
    mut frag_shader: impl FnMut(V::Attr) -> Vec4,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    for [p0, p1, p2] in triangles_iter(vert, tris) {
        draw_triangle(
            p0,
            p1,
            p2,
            &mut frag_shader,
            pixels.borrow(),
            depth_buf.borrow(),
        )
    }
}

pub fn draw_triangle<V: Vertex>(
    v0: &V,
    v1: &V,
    v2: &V,
    mut frag_shader: impl FnMut(V::Attr) -> Vec4,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    let p0 = v0.position();
    let p1 = v1.position();
    let p2 = v2.position();

    let p0i = p0.xy().to_i32();
    let p1i = p1.xy().to_i32();
    let p2i = p2.xy().to_i32();

    let min = p0i.min(p1i).min(p2i).max(Vec2i::repeat(0));
    let max = p0i.max(p1i).max(p2i).min(Vec2i::from([pixels.width as i32, pixels.height as i32]));

    // 2 times the area of the triangle
    let tri_area = orient_2d_i32(p0i, p1i, p2i) as f32;

    if tri_area <= 0.0 {
        return;
    }

    for y in min.y..=max.y {
        for x in min.x..=max.x {
            let p = Vec2i::from([x, y]);
            let w = Vec3i::from([
                orient_2d_i32(p1i, p2i, p),
                orient_2d_i32(p2i, p0i, p),
                orient_2d_i32(p0i, p1i, p),
            ]);
            if w.x >= 0 && w.y >= 0 && w.z >= 0 {
                let w = w.to_f32() / tri_area;
                let interp = V::interpolate(w, v0, v1, v2);
                let idx = (x as usize, y as usize);
                let z = w.x * p0.z + w.y * p1.z + w.z * p2.z;
                if z > depth_buf[idx] {
                    let color = frag_shader(interp).map(|el| el.clamp(-1.0, 1.0)) * 255.0;
                    let color = color.to_u8();
                    pixels[idx] = [color.x, color.y, color.z, color.w];
                    depth_buf[idx] = z;
                }
            }
        }
    }
}


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

pub fn draw_triangles_opt<V: Vertex>(
    vert: &[V],
    tris: &[[u32; 3]],
    mut frag_shader: impl FnMut(V::Attr) -> Vec4,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    for &[p0, p1, p2] in tris {
        let v0 = &vert[p2 as usize];
        let v1 = &vert[p1 as usize];
        let v2 = &vert[p0 as usize];

        let p0 = v0.position();
        let p1 = v1.position();
        let p2 = v2.position();

        let p0i = p0.xy().to_i32();
        let p1i = p1.xy().to_i32();
        let p2i = p2.xy().to_i32();

        let min = p0i.min(p1i).min(p2i).max(Vec2i::repeat(0));
        let max = p0i.max(p1i).max(p2i).min(Vec2i::from([pixels.width as i32, pixels.height as i32]));

        // 2 times the area of the triangle
        let tri_area = orient_2d_i32(p0i, p1i, p2i) as f32;

        if tri_area <= 0.0 {
            continue;
        }

        let bbox_width = max.x - min.x + 1;

        // (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
        // u.x         * (p.y - a.y) - u.y         * (p.x - a.x)
        // u.x * p.y - u.x * a.y - (u.y * p.x - u.y * a.x);
        // u.x * p.y - u.x * a.y + u.y * a.x - u.y * p.x;
        // u.x * p.y - B         + C         - u.y * p.x;
        // u.x * p.y + (-B)      + C         - u.y * p.x;
        // u.x * p.y +        (C - B)        - u.y * p.x;
        // u.x * p.y +           D           - u.y * p.x;
        // u.x * p.y - u.y * p.x + D;

        let u_12 = p2i - p1i;
        let c_12 = u_12.y * p1i.x - u_12.x * p1i.y;
        let w0_inc = Vec2i::from([-u_12.y, u_12.x + u_12.y * bbox_width]);
        let mut w0 = u_12.x * min.y - u_12.y * min.x + c_12;

        let u_20 = p0i - p2i;
        let c_20 = u_20.y * p2i.x - u_20.x * p2i.y;
        let w1_inc = Vec2i::from([-u_20.y, u_20.x + u_20.y * bbox_width]);
        let mut w1 = u_20.x * min.y - u_20.y * min.x + c_20;

        let u_01 = p1i - p0i;
        let c_01 = u_01.y * p0i.x - u_01.x * p0i.y;
        let w2_inc = Vec2i::from([-u_01.y, u_01.x + u_01.y * bbox_width]);
        let mut w2 = u_01.x * min.y - u_01.y * min.x + c_01;

        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let p = Vec2i::from([x, y]);

                let w = Vec3i::from([w0, w1, w2]);
                if (w.x | w.y | w.z) >= 0 {
                    let w = w.to_f32() / tri_area;
                    let interp = V::interpolate(w, v0, v1, v2);
                    let idx = (x as usize, y as usize);
                    let z = w.x * p0.z + w.y * p1.z + w.z * p2.z;
                    if z > depth_buf[idx] {
                        let color = frag_shader(interp).map(|el| el.clamp(-1.0, 1.0)) * 255.0;
                        pixels[idx] = color.to_u8().to_array();
                        depth_buf[idx] = z;
                    }
                }
                w0 += w0_inc.x;
                w1 += w1_inc.x;
                w2 += w2_inc.x;
            }
            w0 += w0_inc.y;
            w1 += w1_inc.y;
            w2 += w2_inc.y;
        }
    }
}

/*
pub fn draw_triangle_simd<const LANES: usize>(
    mut p0: ScreenPos,
    mut p1: ScreenPos,
    mut p2: ScreenPos,
    color: u32,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let width = pixels.width as i32;
    let height = pixels.height as i32;

    let min_x = p0.0.min(p1.0).min(p2.0).max(0);
    let min_y = p0.1.min(p1.1).min(p2.1).max(0);
    let max_x = p0.0.max(p1.0).max(p2.0).min(width);
    let max_y = p0.1.max(p1.1).max(p2.1).min(height);

    // let color = Simd::<u8, 4>::from_array(color.to_be_bytes());

    let buf = pixels.as_slice_mut();

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if () {
                let idx = (y * width + x) as usize;
                buf[idx] = color.to_be_bytes();
            }
        }
    }
}

impl<T> Deref for Vec<T, 2>
where
    T: SimdElement,
{
    type Target = XY<T>;

    fn deref(&self) -> &XY<T> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T> DerefMut for Vec<T, 2>
where
    T: SimdElement,
{
    fn deref_mut(&mut self) -> &mut XY<T> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T: Add, const N: usize> Add for Vec<T, N>
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
    }
}

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
fn depth_barycentric(l: Vec3, z0: f32, z1: f32, z2: f32) -> f32 {
    z0 * l.x + z1 * l.y + z2 * l.z
}

#[inline(always)]
fn triangle_depth((x, y): (i32, i32), p0: ScreenPos, p1: ScreenPos, p2: ScreenPos) -> f32 {
    let lambda = barycentric_coords(
        (x as f32, y as f32),
        (p0.0 as f32, p0.1 as f32),
        (p1.0 as f32, p1.1 as f32),
        (p2.0 as f32, p2.1 as f32),
    );
    depth_barycentric(lambda, p0.2, p1.2, p2.2)
}
