use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rasterization::{
    buf::{MatrixSliceMut, PixelBuf},
    obj,
    vec::Vec4,
    prim3d::draw_triangles,
};

fn triangle_rasterization(c: &mut Criterion) {
    const WIDTH: usize = 720;
    const HEIGHT: usize = 720;

    let obj = obj::load_obj("teapot.obj".as_ref()).unwrap();
    let mut pixel_storage = vec![[0u8; 4]; WIDTH * HEIGHT];
    let mut pixels = PixelBuf::new(&mut pixel_storage, WIDTH, HEIGHT);
    let mut depth_storage = vec![0.0f32; WIDTH * HEIGHT];
    let mut depth = MatrixSliceMut::new(&mut depth_storage, WIDTH, HEIGHT);

    c.bench_function("draw_triangles", move |b| {
        b.iter(|| {
            for d in depth.as_slice_mut() {
                *d = f32::MIN;
            }
            draw_triangles(&obj.vert, &obj.tris, |_| Vec4::from([1.0, 1.0, 1.0, 1.0]), pixels.borrow(), depth.borrow());
            black_box(pixels.borrow());
            black_box(depth.borrow());
        })
    });
}

criterion_group!(benches, triangle_rasterization);
criterion_main!(benches);
