#![feature(slice_as_chunks, iter_next_chunk, portable_simd)]

use std::time::{Duration, Instant};

use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

pub mod buf;
pub mod prim3d;
pub mod vec;

use vec::{Mat, Vec3, Vec4};

use crate::vec::Mat4x4;

const WIDTH: u32 = 720;
const HEIGHT: u32 = 720;

pub type ScreenPos = (i32, i32, f32);
pub type Pixel = [u8; 4];

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("rasterizer")
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

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
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
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
    vert: Vec<Vec4>,
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
                    let [x, y, z] = it.next_chunk().unwrap();
                    vert.push(Vec4::from([x, y, z, 1.0]));
                }
                Some("f") => {
                    let mut it = it.map(|el| el.parse::<u32>().unwrap() - 1);
                    tris.push(it.next_chunk().unwrap());
                }
                _ => continue,
            }
        }

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

        clear_color(pixels.borrow(), 0x00_00_00_ff);

        let mut depth_buf =
            buf::MatrixSliceMut::new(&mut self.depth_buf, WIDTH as usize, HEIGHT as usize);

        for d in depth_buf.as_slice_mut() {
            *d = f32::MIN;
        }

        let model = Vec3::from([-0.5, -0.5, 0.0]).to_translation()
            * Mat4x4::rotation_x(self.acc + std::f32::consts::PI / 4.0)
            * Vec3::from([0.3, 1.0, 1.0]).to_scale();

        let vert: Vec<_> = self.vert.iter().copied().map(|v| model * v).collect();

        for [p0, p1, p2] in triangles_iter(&vert, &self.tris) {
            let cross = (p0 - p1).xyz().cross((p2 - p1).xyz()).normalize();
            let color = 256.0 * (cross + Mat::one()) / 2.0;
            let color = u32::from_be_bytes([color.x as u8, color.y as u8, color.z as u8, 0xff]);

            prim3d::draw_triangle(
                pixels.ndc_to_screen(p0),
                pixels.ndc_to_screen(p1),
                pixels.ndc_to_screen(p2),
                color,
                pixels.borrow(),
                depth_buf.borrow(),
            )
        }
        println!("render time: {:?}", start.elapsed());
    }

    fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
    }
}

pub fn triangles_iter<'a>(
    vert: &'a [Vec4],
    tris: &'a [[u32; 3]],
) -> impl Iterator<Item = [Vec4; 3]> + 'a {
    tris.iter()
        .map(|&[p0, p1, p2]| [vert[p0 as usize], vert[p1 as usize], vert[p2 as usize]])
}

fn clear_color(pixels: buf::PixelBuf, color: u32) {
    for pixel in pixels.as_slice_mut() {
        *pixel = color.to_be_bytes();
    }
}
