#![feature(slice_as_chunks, iter_next_chunk, portable_simd)]

pub mod buf;
pub mod obj;
pub mod prim3d;
pub mod vec;

pub type ScreenPos = (i32, i32, f32);
pub type Pixel = [u8; 4];

use vec::Vec4;

pub fn triangles_iter<'a>(
    vert: &'a [Vec4],
    tris: &'a [[u32; 3]],
) -> impl Iterator<Item = [Vec4; 3]> + 'a {
    tris.iter()
        .map(|&[p0, p1, p2]| [vert[p0 as usize], vert[p1 as usize], vert[p2 as usize]])
}

pub fn clear_color(pixels: buf::PixelBuf, color: u32) {
    for pixel in pixels.as_slice_mut() {
        *pixel = color.to_be_bytes();
    }
}
