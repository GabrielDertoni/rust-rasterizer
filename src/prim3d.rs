use std::simd::{LaneCount, SupportedLaneCount, SimdElement, Simd};
use std::ops::{Deref, DerefMut, Add, Sub};

use crate::{ScreenPos, BarycentricCoords, triangles_iter};
use crate::buf::{PixelBuf, MatrixSliceMut};

pub fn draw_line(
    p0: ScreenPos,
    p1: ScreenPos,
    color: u32,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    for (x, y, z) in LineIter::new(p0, p1) {
        let idx = (x as usize, y as usize);
        if z > depth_buf[idx] {
            depth_buf[idx] = z;
            pixels[idx] = color.to_be_bytes();
        }
    }
}

pub fn draw_triangle_outline(
    p0: ScreenPos,
    p1: ScreenPos,
    p2: ScreenPos,
    color: u32,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    draw_line(p0, p1, color, pixels.borrow(), depth_buf.borrow());
    draw_line(p1, p2, color, pixels.borrow(), depth_buf.borrow());
    draw_line(p0, p2, color, pixels.borrow(), depth_buf.borrow());
}

pub fn draw_triangle(
    mut p0: ScreenPos,
    mut p1: ScreenPos,
    mut p2: ScreenPos,
    color: u32,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
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
    let mut left_coef = if left_dy != 0 { p0.0 - p0.1 * left_dx / left_dy } else { 0 };

    let mut right_dx = p2.0 - p0.0;
    let mut right_dy = p2.1 - p0.1;
    let mut right_coef = if right_dy != 0 { p0.0 - p0.1 * right_dx / right_dy } else { 0 };

    for y in p0.1..p1.1.min(p2.1) {
        let l = left_coef  + y * left_dx  / left_dy;
        let r = right_coef + y * right_dx / right_dy;
        for x in l..=r {
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
        let l = left_coef  + y * left_dx  / left_dy;
        let r = right_coef + y * right_dx / right_dy;
        for x in l..=r {
            let depth = triangle_depth((x, y), p0, p1, p2);
            let idx = (x as usize, y as usize);
            if depth > depth_buf[idx] {
                depth_buf[idx] = depth;
                pixels[idx] = color.to_be_bytes();
            }
        }
    }
}

pub fn draw_triangles(
    vert: &[[f32; 3]],
    tris: &[[u32; 3]],
    color: u32,
    mut pixels: PixelBuf,
    mut depth_buf: MatrixSliceMut<f32>,
) {
    for [p0, p1, p2] in triangles_iter(vert, tris) {
        draw_triangle(
            pixels.ndc_to_screen(p0),
            pixels.ndc_to_screen(p1),
            pixels.ndc_to_screen(p2),
            color,
            pixels.borrow(),
            depth_buf.borrow(),
        )
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


fn orient_2d(a: Vec2i, b: Vec2i, c: Vec2i) -> i32 {
    a - b
}

#[repr(C)]
pub struct XY<T> {
    pub x: T,
    pub y: T,
}

pub type Vec2 = Vec<f32, 2>;
pub type Vec2i = Vec<i32, 2>;

#[repr(transparent)]
pub struct Vec<T, const N: usize>(Simd<T, N>)
where
    T: SimdElement,
    LaneCount<N>: SupportedLaneCount;

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
*/

pub struct LineIter {
    x0: i32,
    y0: i32,
    z0: f32,
    inner: LineToIter,
}

impl LineIter {
    pub fn new(
        (x0, y0, z0): ScreenPos,
        (x1, y1, z1): ScreenPos,
    ) -> Self {
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

#[inline(always)]
fn barycentric_coords(
    (x, y): (f32, f32),
    (x1, y1): (f32, f32),
    (x2, y2): (f32, f32),
    (x3, y3): (f32, f32),
) -> BarycentricCoords {
    // source: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    let det = (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
    let lambda_1 = ((y3 - y2) * (x - x3) + (x3 - x2) * (y - y3)) / det;
    let lambda_2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det;
    let lambda_3 = 1.0 - lambda_1 - lambda_2;
    (lambda_1, lambda_2, lambda_3)
}

#[inline(always)]
fn depth_barycentric((l0, l1, l2): BarycentricCoords, z0: f32, z1: f32, z2: f32) -> f32 {
    z0 * l0 + z1 * l1 + z2 * l2
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
