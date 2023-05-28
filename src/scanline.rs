/// This doesn't really implement The Scanline Algorithm™️, but it is an algorithm that goes line by line and only inside each triangle.

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
    let max = p0i
        .max(p1i)
        .max(p2i)
        .min(Vec2i::from([pixels.width as i32, pixels.height as i32]));

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

/*

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
            if !(x >= 0 && x < pixels.width as i32 && y >= 0 && y < pixels.height as i32) {
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
