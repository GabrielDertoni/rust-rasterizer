#![feature(portable_simd, slice_flatten)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rasterization::{
    buf::{MatrixSliceMut, PixelBuf},
    frag_shaders::TextureMappingFragShader,
    obj::Obj,
    prim3d,
    vec::{Mat4x4, Vec2i, Vec3},
    VertBuf,
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

fn model_with_texture(c: &mut Criterion) {
    const WIDTH: usize = 720;
    const HEIGHT: usize = 720;

    let Obj {
        verts,
        mut tris,
        normals,
        uvs,
        materials,
        ..
    } = Obj::load("Skull/12140_Skull_v3_L2.obj".as_ref()).unwrap();

    let mut pixel_storage = vec![[0u8; 4]; WIDTH * HEIGHT];
    let mut pixels = PixelBuf::new(&mut pixel_storage, WIDTH, HEIGHT);
    let mut depth_storage = vec![0.0f32; WIDTH * HEIGHT];
    let mut depth = MatrixSliceMut::new(&mut depth_storage, WIDTH, HEIGHT);

    let camera_pos = Vec3::from([0.0, 1.0, -1.0]);
    let look_at = Vec3::from([0.0, 1.0, 0.0]);
    let theta = -std::f32::consts::FRAC_PI_2;
    let scale = 15.0;

    let model = Mat4x4::rotation_x(theta) * Vec3::repeat(1.0 / scale).to_scale();

    let eye = camera_pos;
    let up = Vec3::from([0.0, 1.0, 0.0]);

    let look = (look_at - eye).normalized();
    let right = up.cross(look).normalized();
    let up = right.cross(look).normalized();

    let view = Mat4x4::from([
        [right.x, right.y, right.z, -right.dot(eye)],
        [up.x, up.y, up.z, -up.dot(eye)],
        [look.x, look.y, look.z, -look.dot(eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    let transform = pixels.ndc_to_screen() * view * model;

    let positions: Vec<_> = verts.iter().map(|&p| transform * p).collect();

    let uvs = {
        let width = materials[0].map_kd.width();
        let height = materials[0].map_kd.height();
        uvs.iter()
            .map(|uv| {
                Vec2i::from([
                    (uv.x * width as f32) as i32,
                    ((1.0 - uv.y) * height as f32) as i32,
                ])
            })
            .collect::<Vec<_>>()
    };

    let vert_buf = VertBuf {
        positions: &positions,
        normals: &normals,
        uvs: &uvs,
    };

    let light_dir = Vec3::from([-0.5, 0.5, 0.0]).normalized();
    let shader = TextureMappingFragShader::new(light_dir, model, &materials[0].map_kd);

    let mut group = c.benchmark_group("model_with_texture");

    group.bench_function("no vertex cache optimization", |b| {
        b.iter(|| {
            for d in depth.as_slice_mut() {
                *d = f32::MIN;
            }

            prim3d::draw_triangles_opt(&vert_buf, &tris, &shader, pixels.borrow(), depth.borrow());
            black_box(pixels.borrow());
            black_box(depth.borrow());
        })
    });

    /*
    let mut n_used = vec![0; verts.len()];

    for tri in &tris {
        for ix in tri {
            n_used[ix.position as usize] += 1;
        }
    }
    */

    tris.sort_by_key(|idxs| {
        /*
        std::cmp::Reverse(
            n_used[idxs[0].position as usize]
                + n_used[idxs[1].position as usize]
                + n_used[idxs[2].position as usize],
        )
        */
        let v0 = verts[idxs[0].position as usize].xyz();
        let v1 = verts[idxs[1].position as usize].xyz();
        let v2 = verts[idxs[2].position as usize].xyz();
        let mid = (v0 + v1 + v2) / 3.0;
        (mid.y * WIDTH as f32 + mid.x) as i32
    });

    group.bench_function("vertex cache optimized", |b| {
        b.iter(|| {
            for d in depth.as_slice_mut() {
                *d = f32::MIN;
            }

            prim3d::draw_triangles_opt(&vert_buf, &tris, &shader, pixels.borrow(), depth.borrow());
            black_box(pixels.borrow());
            black_box(depth.borrow());
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

/*
#[inline(always)]
fn frag_shader(normal: Vec3) -> Vec4 {
    Vec4::from([normal.x, normal.y, normal.z, 1.0])
}

const LANES: usize = 4;

#[inline(always)]
fn frag_shader_simd(
    mask: Mask<i32, LANES>,
    n: vec::Vec<Simd<f32, LANES>, 3>,
) -> vec::Vec<Simd<f32, LANES>, 4> {
    vec::Vec::from([n.x, n.y, n.z, Simd::splat(1.0)])
}
*/

criterion_group!(benches, model_with_texture);
criterion_main!(benches);
