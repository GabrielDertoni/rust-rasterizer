use crate::vec::Vec;

pub fn draw_triangle(
    p0: Vec2i,
    p1: Vec2i,
    p2: Vec2i,
    z0: f32,
    z1: f32,
    z2: f32,
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