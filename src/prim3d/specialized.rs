

pub fn draw_triangles_depth_specialized(
    vert_positions: &[Vec4],
    tris: &[[usize; 3]],
    proj_matrix: Mat4x4,
    mut depth_buf: MatrixSliceMut<f32>,
    ctx: &mut RenderContext,
    bias: f32,
) {
    assert_eq!(depth_buf.width % LANES, 0);
    assert_eq!(depth_buf.stride % LANES, 0);
    assert!(depth_buf
        .as_ptr()
        .is_aligned_to(std::mem::align_of::<Simd<f32, LANES>>()));

    let width = depth_buf.width;
    let height = depth_buf.height;

    // Run the vertex shader and cache the results
    ctx.vertex_attrib.reset();
    let mut vertex_attrib = bumpalo::collections::Vec::new_in(&ctx.vertex_attrib);

    let r0 = Simd::from_array(proj_matrix.rows[0]);
    let r1 = Simd::from_array(proj_matrix.rows[1]);
    let r2 = Simd::from_array(proj_matrix.rows[2]);
    let r3 = Simd::from_array(proj_matrix.rows[3]);
    vertex_attrib.extend(vert_positions.iter().map(|v| {
        let v = Simd::from_array(v.to_array());

        let inv_w = 1. / (r3 * v).reduce_sum();
        let x = (r0 * v).reduce_sum() * inv_w;
        let y = (r1 * v).reduce_sum() * inv_w;
        let z = (r2 * v).reduce_sum() * inv_w;

        (
            ndc_to_screen(Vec2::from([x, y]), width as f32, height as f32).to_i32(),
            z,
        )
    }));

    let bbox = BBox {
        x: 0,
        y: 0,
        width: width as i32,
        height: height as i32,
    };

    for &[v0, v1, v2] in tris {
        let (v2, v1, v0) = (v0, v1, v2);

        draw_triangle_depth(
            vertex_attrib[v0].0,
            vertex_attrib[v1].0,
            vertex_attrib[v2].0,
            vertex_attrib[v0].1,
            vertex_attrib[v1].1,
            vertex_attrib[v2].1,
            depth_buf.borrow(),
            bbox,
            bias,
        );
    }
}

#[inline(always)]
fn draw_triangle_depth_alpha(
    p0_screen: Vec2i,
    p1_screen: Vec2i,
    p2_screen: Vec2i,
    z0: f32,
    z1: f32,
    z2: f32,
    depth_buf: MatrixSliceMut<f32>,
    alpha_map: MatrixSlice<u8>,
    aabb: BBox<i32>,
    bias: f32,
) {
    let mut min = p0_screen.min(p1_screen).min(p2_screen);
    // Only works because `STEP_X` is a power of 2.
    min.x &= !(STEP_X - 1) as i32;
    min.x = min.x.max(aabb.x);
    min.y &= !(STEP_Y - 1) as i32;
    min.y = min.y.max(aabb.y);
    let max = p0_screen.max(p1_screen).max(p2_screen).min(Vec2i::from([
        aabb.x + aabb.width - STEP_X,
        aabb.y + aabb.height - STEP_Y,
    ]));

    let is_visible = is_triangle_visible(p0_screen, p1_screen, p2_screen, z0, z1, z2, aabb);

    // 2 times the area of the triangle
    let tri_area = orient_2d(p0_screen, p1_screen, p2_screen);

    if tri_area <= 0 || !is_visible || min.x == max.x || min.y == max.y {
        return;
    }

    let (w0_inc, mut w0_row) = orient_2d_step(p1_screen, p2_screen, min);
    let (w1_inc, mut w1_row) = orient_2d_step(p2_screen, p0_screen, min);
    let (w2_inc, mut w2_row) = orient_2d_step(p0_screen, p1_screen, min);

    let inv_area = Simd::splat(1.0 / tri_area as f32);
    let (z0, z1, z2) = (Simd::splat(z0), Simd::splat(z1), Simd::splat(z2));
    let bias = Simd::splat(bias);

    let stride = depth_buf.stride;
    let depth_buf = depth_buf.as_slice_mut();
    let alpha_map = alpha_map.as_slice();

    let mut y = min.y;
    while y < max.y {
        let mut w0 = w0_row;
        let mut w1 = w1_row;
        let mut w2 = w2_row;
        let mut x = min.x;
        while x < max.x {
            let mask = (w0 | w1 | w2).simd_ge(Simd::splat(0));

            if mask.any() {
                // yyyyxxxx
                // yyyxxxyx
                // let idx = (y / STEP_Y) * stride as i32
                //     + (x / STEP_X) * (STEP_X * STEP_Y)
                //     + (y % STEP_Y) * STEP_X
                //     + x % STEP_X;
                let idx = y * stride as i32 + x;
                let idx = idx as usize;
                let z = (w0.cast() * z0 + w1.cast() * z1 + w2.cast() * z2) * inv_area;
                let prev_depth = unsafe { *depth_buf.as_ptr().add(idx).cast() };
                let mask = mask & z.simd_lt(prev_depth);
                let alpha = unsafe { *alpha_map.as_ptr().add(idx).cast::<Mask<i8, 4>>() };

                let new_depth = (mask & alpha.cast()).select(z + bias, prev_depth);
                unsafe {
                    let ptr = &mut depth_buf[idx] as *mut f32;
                    *ptr.cast() = new_depth;
                }
            }

            w0 += w0_inc.x;
            w1 += w1_inc.x;
            w2 += w2_inc.x;

            x += STEP_X;
        }
        w0_row += w0_inc.y;
        w1_row += w1_inc.y;
        w2_row += w2_inc.y;

        y += STEP_Y;
    }
}