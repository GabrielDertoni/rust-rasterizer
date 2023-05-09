#![feature(slice_as_chunks, iter_next_chunk, portable_simd)]

use std::time::{Instant, Duration};
use std::ops::RangeInclusive;

use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState},
    event_loop::EventLoop,
    window::WindowBuilder,
    dpi::LogicalSize,
};
use pixels::{SurfaceTexture, Pixels};

mod prim3d;
mod buf;

const WIDTH: u32 = 720;
const HEIGHT: u32 = 720;

pub type ScreenPos = (i32, i32, f32);
pub type BarycentricCoords = (f32, f32, f32);
pub type Pixel = [u8; 4];


fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("rasterizer")
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop).unwrap();

    let mut pixels = {
        let size = window.inner_size();
        let surface_texture = SurfaceTexture::new(size.width, size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture).unwrap()
    };

    let mut world = World::new();

    let mut start = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        // control_flow.set_poll();
        control_flow.set_wait();

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => control_flow.set_exit(),
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput {
                    input: KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Space),
                        ..
                    },
                    is_synthetic: false,
                    ..
                },
                ..
            } => world.toggle_pause(),
            Event::MainEventsCleared => {
                let dt = start.elapsed();
                start = Instant::now();
                world.render(pixels.frame_mut(), dt);
                window.request_redraw()
            }
            Event::RedrawRequested(_) => pixels.render().unwrap(),
            _ => (),
        }
    });
}

struct World {
    vert: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    depth_buf: Vec<f32>,
    acc: f32,
    is_paused: bool,
}

impl World {
    fn new() -> Self {
        let obj = std::fs::read_to_string("teapot.obj").unwrap();
        let mut vert = Vec::new();
        let mut tris = Vec::new();
        for line in obj.lines() {
            let mut it = line.split_ascii_whitespace();
            match it.next() {
                Some("v") => {
                    let mut it = it.map(|el| el.parse::<f32>().unwrap() / 4.0);
                    vert.push(it.next_chunk().unwrap());
                }
                Some("f") => {
                    let mut it = it.map(|el| el.parse::<u32>().unwrap() - 1);
                    tris.push(it.next_chunk().unwrap());
                }
                _ => continue,
            }
        }

        let maxval = vert.iter()
            .flat_map(|v| v)
            .max_by(|l, r| l.total_cmp(r));
        dbg!(vert.len());
        dbg!(tris.len());

        World {
            tris,
            vert,
            depth_buf: vec![0.0; (WIDTH * HEIGHT) as usize],
            acc: 0.0,
            is_paused: true,
        }
    }

    fn render(&mut self, buf: &mut [u8], dt: Duration) {
        let start = Instant::now();
        let (pixels, _) = buf.as_chunks_mut::<4>();
        let mut pixels = buf::PixelBuf::new(pixels, WIDTH as usize, HEIGHT as usize);

        if !self.is_paused {
            self.acc += dt.as_secs_f32() * std::f32::consts::PI / 4.0;
        }

        draw_bg(pixels.borrow(), 0x00_00_00_ff);

        let mut depth_buf = buf::MatrixSliceMut::new(
            &mut self.depth_buf,
            WIDTH as usize,
            HEIGHT as usize,
        );

        for d in depth_buf.as_slice_mut() {
            *d = f32::MIN;
        }

        let vert: Vec<_> = self.vert.iter()
            .copied()
            .map(|v| [
                v[0],
                v[1] * self.acc.cos() - v[2] * self.acc.sin(),
                v[1] * self.acc.sin() + v[2] * self.acc.cos(),
            ])
            .collect();

        for [p0, p1, p2] in triangles_iter(&vert, &self.tris) {
            // The cross product will be the normal vector to the face
            // source: https://en.wikipedia.org/wiki/Cross_product
            let a1 = p0[0] - p1[0];
            let a2 = p0[1] - p1[1];
            let a3 = p0[2] - p1[2];

            let b1 = p2[0] - p1[0];
            let b2 = p2[1] - p1[1];
            let b3 = p2[2] - p1[2];

            let cross = [
                a2 * b3 - a3 * b2,
                a1 * b3 - a3 * b1,
                a1 * b2 - a2 * b1,
            ];
            let mag = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            let cross = [cross[0] / mag, cross[1] / mag, cross[2] / mag];
            let color = u32::from_be_bytes([
                (256.0 * (cross[0] + 1.0) / 2.0) as u8,
                (256.0 * (cross[1] + 1.0) / 2.0) as u8,
                (256.0 * (cross[2] + 1.0) / 2.0) as u8,
                0xff,
            ]);

            prim3d::draw_triangle(
                pixels.ndc_to_screen(p0),
                pixels.ndc_to_screen(p1),
                pixels.ndc_to_screen(p2),
                color,
                pixels.borrow(),
                depth_buf.borrow(),
            )
        }

        /*
        prim3d::draw_triangle(
            (50, 50, 0.0),
            (100, 48, 0.0),
            (50, 100, 0.0),
            0xff_ff_ff_ff,
            pixels.borrow(),
            depth_buf.borrow(),
        );
        */

        println!("render time: {:?}", start.elapsed());
    }

    fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
    }
}

pub fn triangle_iter(
    mut p0: ScreenPos,
    mut p1: ScreenPos,
    mut p2: ScreenPos,
) -> impl Iterator<Item = (i32, RangeInclusive<i32>)> {
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

    (p0.1..p1.1.min(p2.1))
        .map(move |y| {
            let l = left_coef  + y * left_dx  / left_dy;
            let r = right_coef + y * right_dx / right_dy;
            (y, l..=r)
        })
        .chain({
            if p1.1 < p2.1 {
                left_dx = p2.0 - p1.0;
                left_dy = p2.1 - p1.1;
                left_coef = p1.0 - p1.1 * left_dx / left_dy;
            } else if p1.1 > p2.1 {
                right_dx = p1.0 - p2.0;
                right_dy = p1.1 - p2.1;
                right_coef = p2.0 - p2.1 * right_dx / right_dy;
            }
            (p1.1.min(p2.1)..p1.1.max(p2.1)) 
                .map(move |y| {
                    let l = left_coef  + y * left_dx  / left_dy;
                    let r = right_coef + y * right_dx / right_dy;
                    (y, l..=r)
                })
        })
}

pub fn fill(
    iter: impl IntoIterator<Item = (i32, impl IntoIterator<Item = i32>)>,
    color: u32,
    pixels: &mut [Pixel],
) {
    for (y, x_range) in iter {
        for x in x_range {
            let idx = y as usize * WIDTH as usize + x as usize;
            pixels[idx] = color.to_be_bytes();
        }
    }
}

pub fn triangles_iter<'a>(
    vert: &'a [[f32; 3]],
    tris: &'a [[u32; 3]],
) -> impl Iterator<Item = [[f32; 3]; 3]> + 'a {
    tris.iter()
        .map(|&[p0, p1, p2]| [
             vert[p0 as usize],
             vert[p1 as usize],
             vert[p2 as usize],
        ])
}

fn draw_bg(pixels: buf::PixelBuf, color: u32) {
    for pixel in pixels.as_slice_mut() {
        *pixel = color.to_be_bytes();
    }
}
