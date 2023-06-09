#![feature(portable_simd, slice_flatten, slice_as_chunks)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rasterization::{
    buf::{self, MatrixSliceMut, PixelBuf},
    obj::Obj,
    prim3d::{self, CullBackFaces},
    shaders,
    utils::{Camera, FpvCamera},
    math::{Mat4x4, Vec2, Vec3, Vec4},
    vertex_shader_simd, IntoSimd, VertBuf, Vertex, VertexBuf,
};

/*
fn triangle_rasterization(c: &mut Criterion) {
    const WIDTH: usize = 720;
    const HEIGHT: usize = 720;

    let mut group = c.benchmark_group("Triangle rasterization");

    let obj::Obj { vert, tris }= obj::load_obj("teapot.obj".as_ref()).unwrap();
    let mut vert = compute_normals(&vert, &tris);

    let mut pixel_storage = vec![[0u8; 4]; WIDTH * HEIGHT];
    let mut pixels = PixelBuf::new(&mut pixel_storage, WIDTH, HEIGHT);
    let mut depth_storage = vec![0.0f32; WIDTH * HEIGHT];
    let mut depth = MatrixSliceMut::new(&mut depth_storage, WIDTH, HEIGHT);

    for v in &mut vert {
        v.pos = pixels.ndc_to_screen(v.pos);
    }

    group.bench_function("draw_triangles", |b| {
        b.iter(|| {
            for d in depth.as_slice_mut() {
                *d = f32::MIN;
            }
            draw_triangles(&vert, &tris, frag_shader, pixels.borrow(), depth.borrow());
            black_box(pixels.borrow());
            black_box(depth.borrow());
        })
    });

    group.bench_function("draw_triangles_opt", |b| {
        b.iter(|| {
            for d in depth.as_slice_mut() {
                *d = f32::MIN;
            }
            draw_triangles_opt(&vert, &tris, frag_shader_simd, pixels.borrow(), depth.borrow());
            black_box(pixels.borrow());
            black_box(depth.borrow());
        })
    });

    let mut f = std::fs::File::create("bench.png").unwrap();
    image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(WIDTH as u32, HEIGHT as u32, pixels.as_slice().flatten())
        .unwrap()
        .write_to(&mut f, image::ImageFormat::Png)
        .unwrap();

    group.finish();
}
*/

fn shadows(c: &mut Criterion) {
    const WIDTH: usize = 720;
    const HEIGHT: usize = 720;

    let obj = Obj::load("models/scene2.obj".as_ref()).unwrap();

    let mut pixel_storage = vec![[0u8; 4]; WIDTH * HEIGHT];
    let mut pixels = PixelBuf::new(&mut pixel_storage, WIDTH, HEIGHT);
    let mut depth_storage = vec![0.0f32; WIDTH * HEIGHT];
    let mut depth_buf = MatrixSliceMut::new(&mut depth_storage, WIDTH, HEIGHT);

    let mut shadow_map = vec![0.0_f32; WIDTH * HEIGHT];

    let ratio = WIDTH as f32 / HEIGHT as f32;

    let camera = FpvCamera {
        position: Vec3::from([0., -10., 4.]),
        up: Vec3::from([0., 0., 1.]),
        pitch: 70.,
        yaw: 0.,
        sensitivity: 0.05,
        speed: 5.,
        fovy: 39.6,
        ratio,
    };

    let (vert_buf, index_buf) = VertBuf::from_obj(&obj);

    let model = Mat4x4::identity();

    let light_pos = Vec3::from([0., -3.77, 6.27]);
    let light_camera =
        Camera::from_blender(light_pos, Vec3::from([33.9, 0., 0.]), 45., 1., 1., 100.);

    let near = 0.1;
    let far = 100.;

    let vert_shader = shaders::lit::LitVertexShader::new(
        camera.transform_matrix(near, far),
        light_camera.transform_matrix(),
    );

    let texture = &obj.materials[0].map_kd;
    let (texture_pixels, _) = texture.as_chunks::<4>();
    let texture = buf::MatrixSlice::new(
        texture_pixels,
        texture.width() as usize,
        texture.height() as usize,
    );

    let core_id = core_affinity::get_core_ids().unwrap()[0];

    let mut render_ctx = prim3d::RenderContext::alloc(vert_buf.len());

    let mut group = c.benchmark_group("shadows");
    group.bench_function("draw_triangles", |b| {
        core_affinity::set_for_current(core_id);
        b.iter(|| {
            let shadow_map = {
                shadow_map.fill(1.0);
                let mut shadow_buf =
                    buf::MatrixSliceMut::new(&mut shadow_map, WIDTH as usize, HEIGHT as usize);

                let light_transform = light_camera.transform_matrix();
                prim3d::draw_triangles_depth_specialized(
                    &vert_buf.positions,
                    &index_buf,
                    light_transform,
                    shadow_buf.borrow(),
                    &mut render_ctx,
                    0.0,
                );
                // Pretty dumb way to make a circular spotlight
                let radius = shadow_buf.width as f32 * 0.5;
                let circle_center =
                    Vec2::from([shadow_buf.width as f32 / 2., shadow_buf.height as f32 / 2.]);
                for y in 0..shadow_buf.height {
                    for x in 0..shadow_buf.width {
                        let p = Vec2::from([x as f32, y as f32]);
                        if (p - circle_center).mag_sq() >= radius * radius {
                            shadow_buf[(x, y)] = -1.0;
                        }
                    }
                }
                buf::MatrixSlice::new(&shadow_map, depth_buf.width, depth_buf.height)
            };

            depth_buf.as_slice_mut().fill(1.0);

            let frag_shader = shaders::lit::LitFragmentShader::new(
                camera.position,
                model,
                light_pos,
                Vec3::from([1., 1., 1.]),
                texture,
                shadow_map.borrow(),
            );

            prim3d::draw_triangles(
                &vert_buf,
                &index_buf,
                &vert_shader,
                &frag_shader,
                pixels.borrow(),
                depth_buf.borrow(),
                &mut render_ctx,
            );
            black_box(pixels.borrow());
            black_box(depth_buf.borrow());
        })
    });

    group.finish();

    let mut f = std::fs::File::create("bench.png").unwrap();
    image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
        WIDTH as u32,
        HEIGHT as u32,
        pixels.as_slice().flatten(),
    )
    .unwrap()
    .write_to(&mut f, image::ImageFormat::Png)
    .unwrap();
}

fn minecraft(c: &mut Criterion) {
    const WIDTH: usize = 1280;
    const HEIGHT: usize = 720;

    let obj = Obj::load("models/lost_empire/lost_empire.obj".as_ref()).unwrap();

    let mut pixel_storage = vec![[0u8; 4]; WIDTH * HEIGHT];
    let mut pixels = PixelBuf::new(&mut pixel_storage, WIDTH, HEIGHT);
    let mut depth_storage = vec![0.0f32; WIDTH * HEIGHT];
    let mut depth_buf = MatrixSliceMut::new(&mut depth_storage, WIDTH, HEIGHT);

    let ratio = WIDTH as f32 / HEIGHT as f32;

    let camera = FpvCamera {
        position: Vec3::from([-7.4, 30.8, 12.7]),
        up: Vec3::from([0., 0., 1.]),
        pitch: 87.4295,
        yaw: -180.382,
        sensitivity: 0.05,
        speed: 5.,
        fovy: 39.6,
        ratio,
    };

    let (vert_buf, index_buf) = VertBuf::from_obj(&obj);

    let near = 0.1;
    let far = 100.;

    let camera_transform = camera.transform_matrix(near, far);
    let vert_shader = shaders::textured::TexturedVertexShader::new(camera_transform);

    let texture = &obj.materials[0].map_kd;
    let (texture_pixels, _) = texture.as_chunks::<4>();
    let texture = buf::MatrixSlice::new(
        texture_pixels,
        texture.width() as usize,
        texture.height() as usize,
    );

    let frag_shader = shaders::textured::TexturedFragmentShader::new(texture);
    let mut render_ctx = prim3d::RenderContext::alloc(vert_buf.len());

    let core_id = core_affinity::get_core_ids().unwrap()[0];

    let mut group = c.benchmark_group("minecraft");
    group.bench_function("draw_triangles", |b| {
        core_affinity::set_for_current(core_id);
        b.iter(|| {
            depth_buf.as_slice_mut().fill(1.0);
            prim3d::draw_triangles(
                &vert_buf,
                &index_buf,
                &vert_shader,
                &frag_shader,
                pixels.borrow(),
                depth_buf.borrow(),
                &mut render_ctx,
            );
            black_box(pixels.borrow());
            black_box(depth_buf.borrow());
        })
    });

    /*
    group.bench_function("depth", |b| {
        core_affinity::set_for_current(core_id);
        b.iter(|| {
            depth_buf.as_slice_mut().fill(1.0);
            prim3d::draw_triangles_depth(
                &vert_buf,
                &index_buf,
                &vertex_shader_simd!([camera_transform: Mat4x4] |vertex: Vertex| -> Vec4 {
                    camera_transform.splat() * vertex.position
                }),
                depth_buf.borrow(),
                &mut render_ctx,
            );
            black_box(depth_buf.borrow());
        })
    });
    */

    group.bench_function("depth_specialized", |b| {
        core_affinity::set_for_current(core_id);
        b.iter(|| {
            depth_buf.as_slice_mut().fill(1.0);
            prim3d::draw_triangles_depth_specialized(
                &vert_buf.positions,
                &index_buf,
                camera_transform,
                depth_buf.borrow(),
                &mut render_ctx,
                0.0,
            );
            black_box(depth_buf.borrow());
        })
    });

    group.finish();

    let mut f = std::fs::File::create("bench.png").unwrap();
    image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
        WIDTH as u32,
        HEIGHT as u32,
        pixels.as_slice().flatten(),
    )
    .unwrap()
    .write_to(&mut f, image::ImageFormat::Png)
    .unwrap();
}

criterion_group!(benches, shadows, minecraft);
criterion_main!(benches);
