#![feature(slice_flatten)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rasterization::{
    Vert,
    compute_normals,
    buf::{MatrixSliceMut, PixelBuf},
    obj,
    vec::{Vec3, Vec4},
    prim3d::{self, draw_triangles, draw_triangles_opt},
};


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
            draw_triangles_opt(&vert, &tris, frag_shader, pixels.borrow(), depth.borrow());
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

#[inline(always)]
fn frag_shader(normal: Vec3) -> Vec4 {
    Vec4::from([normal.x, normal.y, normal.z, 1.0])
}

criterion_group!(benches, triangle_rasterization);
criterion_main!(benches);
