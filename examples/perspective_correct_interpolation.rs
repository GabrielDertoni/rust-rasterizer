#![feature(portable_simd, slice_as_chunks)]

use image::{ImageBuffer, ImageFormat, Rgba};
use rasterization::{
    buf, clear_color, prim3d,
    vec::{Vec, Vec4, Vec4xN},
};
use std::simd::{
    LaneCount, Mask, Simd, SimdPartialEq, SimdPartialOrd, StdFloat, SupportedLaneCount,
};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 480;

#[derive(Clone, Copy)]
struct Vert {
    pos: Vec4,
    texcoord: Vec4,
    color: Vec4,
}

struct Frag;

impl rasterization::FragmentShader for Frag {
    type SimdAttr<const LANES: usize> = (Vec4xN<LANES>, Vec4xN<LANES>, Vec4xN<LANES>)
    where
        LaneCount<LANES>: SupportedLaneCount;

    fn exec<const LANES: usize>(
        &self,
        _mask: Mask<i32, LANES>,
        (pos, texcoord, color): Self::SimdAttr<LANES>,
    ) -> Vec4xN<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let wrapped = texcoord.xy().map(|el| el.fract());
        // (wrapped.x < 0.5) != (wrapped.y < 0.5)
        let brighter = wrapped
            .x
            .simd_lt(Simd::splat(0.5))
            .simd_ne(wrapped.y.simd_le(Simd::splat(0.5)));
        let bright = color.xyz() * Simd::splat(0.5);
        let bright = Vec::from([bright.x, bright.y, bright.z, color.w]);
        bright.zip_with_4(color, |bright, color| (!brighter).select(bright, color))
    }
}

// source: https://stackoverflow.com/questions/24441631/how-exactly-does-opengl-do-perspectively-correct-linear-interpolation
fn main() {
    println!("Allocating buffers");
    let mut im_buf = ImageBuffer::<Rgba<u8>, _>::new(WIDTH, HEIGHT);
    let (pixels, _) = im_buf.as_chunks_mut::<4>();
    let mut pixels = buf::PixelBuf::new(pixels, WIDTH as usize, HEIGHT as usize);

    let mut depth_buf = vec![1.0; (WIDTH * HEIGHT) as usize];
    let mut depth_buf = buf::MatrixSliceMut::new(&mut depth_buf, WIDTH as usize, HEIGHT as usize);

    clear_color(pixels.borrow(), 0xff_ff_ff_ff);

    let vert = [
        Vert {
            pos: [-1., -1., -2., 1.].into(),
            texcoord: [0., 0., 0., 1.].into(),
            color: [0., 0., 1., 1.].into(),
        },
        Vert {
            pos: [1., -1., -1., 1.].into(),
            texcoord: [10., 0., 0., 1.].into(),
            color: [1., 0., 0., 1.].into(),
        },
        Vert {
            pos: [0., 1., -1., 1.].into(),
            texcoord: [0., 10., 0., 1.].into(),
            color: [0., 1., 0., 1.].into(),
        },
    ];

    let tris = [[0, 1, 2]];

    prim3d::draw_triangles(
        &vert[..],
        &tris,
        &|Vert {
              pos,
              texcoord,
              color,
          }: Vert| {
            (
                (pos, texcoord, color),
                Vec4::from([pos.x, pos.y, -2. * pos.z - 2. * pos.w, -pos.z]),
            )
        },
        &Frag,
        pixels,
        depth_buf,
    );

    println!("Writing image");
    let mut f = std::fs::File::create("perspective_correct_interpolation.png").unwrap();
    im_buf.write_to(&mut f, ImageFormat::Png).unwrap();
}
