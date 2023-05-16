#![feature(slice_as_chunks, iter_next_chunk, portable_simd)]

use std::time::{Duration, Instant};
use std::simd::Simd;


use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use rasterization::{
    clear_color,
    compute_normals,
    Vert,
    obj::Obj,
    buf,
    prim3d,
    vec::{self, Vec2, Vec3, Mat4x4},
};

const WIDTH: u32 = 720;
const HEIGHT: u32 = 720;

fn main() {
    // debug();
    // return;
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
        // control_flow.set_wait();
        control_flow.set_poll();

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
                                state,
                                virtual_keycode: Some(virtual_keycode),
                                ..
                            },
                        is_synthetic: false,
                        ..
                    },
                ..
            } => match (state, virtual_keycode) {
                (ElementState::Pressed, VirtualKeyCode::Space) => world.toggle_pause(),
                (ElementState::Pressed, VirtualKeyCode::V) => world.axis.z = 1.0,
                (ElementState::Pressed, VirtualKeyCode::C) => world.axis.z = -1.0,
                (ElementState::Released, VirtualKeyCode::V | VirtualKeyCode::C) => world.axis.z = 0.0,
                (ElementState::Pressed, VirtualKeyCode::W) => world.axis.y = 1.0,
                (ElementState::Pressed, VirtualKeyCode::S) => world.axis.y = -1.0,
                (ElementState::Released, VirtualKeyCode::W | VirtualKeyCode::S) => world.axis.y = 0.0,
                (ElementState::Pressed, VirtualKeyCode::A) => world.axis.x = 1.0,
                (ElementState::Pressed, VirtualKeyCode::D) => world.axis.x = -1.0,
                (ElementState::Released, VirtualKeyCode::A | VirtualKeyCode::D) => world.axis.x = 0.0,
                _ => (),
            },
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
    vert: Vec<Vert>,
    tris: Vec<[u32; 3]>,
    texture: image::RgbaImage,
    depth_buf: Vec<f32>,
    camera_pos: Vec3,
    look_at: Vec3,
    last_render_times: [Duration; 10],
    theta: f32,
    is_paused: bool,
    axis: Vec3,
}

impl World {
    fn new() -> Self {
        // let mut obj = Obj::load("teapot.obj".as_ref()).unwrap();
        let mut obj = Obj::load("Skull/12140_Skull_v3_L2.obj".as_ref()).unwrap();
        /*
        let vert = if obj.has_normals() && obj.has_texture() {
            obj.iter_vertices()
                .map(|vert| Vert {
                    position: vert.position,
                    normal: vert.normal,
                    texture: vert.texture,
                })
                .collect::<Vec<_>>()
        } else {
            let mut vert = obj.vert.iter()
                .map(|&position| Vert {
                    position,
                    normal: Vec3::zero(),
                    texture: Vec2::zero(),
                })
                .collect::<Vec<_>>();
            compute_normals(&mut vert, &obj.tris);
            vert
        };
        */
        let mut vert = obj.vert.iter()
            .map(|&position| Vert {
                position,
                normal: Vec3::zero(),
                texture: Vec2::zero(),
            })
            .collect::<Vec<_>>();
        compute_normals(&mut vert, &obj.tris);
        // assert_eq!(obj.materials.len(), 1);
        World {
            vert,
            tris: obj.tris,
            texture: if obj.materials.len() == 0 {
                Default::default()
            } else {
                std::mem::take(&mut obj.materials[0].map_kd)
            },
            depth_buf: vec![0.0; (WIDTH * HEIGHT) as usize],
            camera_pos: Vec3::from([0.0, 1.0, -1.0]),
            look_at: Vec3::from([0.0, 1.0, 0.0]),
            last_render_times: [Duration::ZERO; 10],
            theta: 0.0,
            is_paused: true,
            axis: Vec3::zero(),
        }
    }

    fn render(&mut self, buf: &mut [u8], dt: Duration) {
        let start = Instant::now();
        let (pixels, _) = buf.as_chunks_mut::<4>();
        let mut pixels = buf::PixelBuf::new(pixels, WIDTH as usize, HEIGHT as usize);

        if !self.is_paused {
            self.theta -= dt.as_secs_f32() * std::f32::consts::PI / 4.0;
        }

        clear_color(pixels.borrow(), 0x00_00_00_ff);

        let mut depth_buf =
            buf::MatrixSliceMut::new(&mut self.depth_buf, WIDTH as usize, HEIGHT as usize);

        for d in depth_buf.as_slice_mut() {
            *d = f32::MIN;
        }

        let model = Mat4x4::rotation_x(self.theta)
            * Vec3::repeat(1.0/15.0).to_scale();

        self.camera_pos += self.axis * dt.as_secs_f32();
        // self.look_at.x += self.axis.x * dt.as_secs_f32();
        // self.look_at.y += self.axis.y * dt.as_secs_f32();

        /*
        let rho = self.camera_pos.z;
        let camera_pos = rho * Vec3::from([
            self.camera_pos.x.sin() * self.camera_pos.y.cos(),
            self.camera_pos.x.sin() * self.camera_pos.y.sin(),
            self.camera_pos.x.cos(),
        ]);
        */

        let eye = self.camera_pos;

        let up = Vec3::from([0.0, 1.0, 0.0]);

        let zaxis = (self.look_at - eye).normalized();
        let xaxis = up.cross(zaxis).normalized();
        let yaxis = xaxis.cross(zaxis).normalized();

        let view = Mat4x4::from([
            [xaxis.x, xaxis.y, xaxis.z, -eye.x],
            [yaxis.x, yaxis.y, yaxis.z, -eye.y],
            [zaxis.x, zaxis.y, zaxis.z, -eye.z],
            [   0.0,      0.0,     0.0,    1.0],
        ]);

        let transform = pixels.ndc_to_screen() * view * model;

        let vert: Vec<_> = self
            .vert
            .iter()
            .map(|v| Vert {
                position: transform * v.position,
                normal: v.normal,
                texture: v.texture,
            })
            .collect();

        // let texture = &self.texture;

        prim3d::draw_triangles_opt(
            &vert,
            &self.tris,
            |_mask, attrs| {
                let n = attrs.normal;
                vec::Vec::from([n.x, n.y, n.z, Simd::splat(1.0)])
                /*
                let [u, v] = attrs.texture.to_array();
                let x = (u * Simd::splat(texture.width()  as f32)).cast::<usize>();
                let y = (v * Simd::splat(texture.height() as f32)).cast::<usize>();
                let idx = y * Simd::splat(texture.width() as usize) + x;
                Simd::gather_or_default(&texture, idx)
                */
            },
            pixels.borrow(),
            depth_buf.borrow(),
        );
        self.last_render_times.rotate_left(1);
        self.last_render_times[self.last_render_times.len() - 1] = start.elapsed();
        let render_time = self.last_render_times
            .iter()
            .sum::<Duration>() / self.last_render_times.len() as u32;
        println!("render time: {render_time:?}");
    }

    fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
    }
}

/*
#[allow(dead_code)]
fn debug() {
    use image::{ImageBuffer, ImageFormat, Rgba, Luma};

    println!("Allocating buffers");
    let mut im_buf = ImageBuffer::<Rgba<u8>, _>::new(WIDTH, HEIGHT);
    let (pixels, _)= im_buf.as_chunks_mut::<4>();
    let mut pixels = buf::PixelBuf::new(pixels, WIDTH as usize, HEIGHT as usize);

    let mut depth_buf = vec![0.0; (WIDTH * HEIGHT) as usize];
    let mut depth_buf =
        buf::MatrixSliceMut::new(&mut depth_buf, WIDTH as usize, HEIGHT as usize);

    for d in depth_buf.as_slice_mut() {
        *d = f32::MIN;
    }

    clear_color(pixels.borrow(), 0x00_00_00_ff);

    println!("Reading .obj");
    let obj = Obj::load("teapot.obj".as_ref()).unwrap();
    let mut vert = compute_normals(&obj.vert, &obj.tris);

    for v in &mut vert {
        v.pos = pixels.ndc_to_screen() * v.pos;
    }

    let start = std::time::Instant::now();

    println!("Rasterizing");
    prim3d::draw_triangles_opt(
        &vert,
        &obj.tris,
        |_mask, n| rasterization::vec::Vec::from([n.x, n.y, n.z, Simd::splat(1.0)]),
        pixels.borrow(),
        depth_buf.borrow(),
    );

    println!("rendered in {:?}", start.elapsed());

    println!("Writing image");
    let mut f = std::fs::File::create("frag.png").unwrap();
    im_buf.write_to(&mut f, ImageFormat::Png).unwrap();

    let mut f = std::fs::File::create("depth.png").unwrap();
    let depth_img = depth_buf.as_slice()
        .iter()
        .copied()
        .map(|v| (255.0 * (v.clamp(-1.0, 1.0) + 1.0) / 2.0) as u8)
        .collect::<Vec<_>>();
    ImageBuffer::<Luma<u8>, _>::from_raw(WIDTH, HEIGHT, depth_img.as_slice())
        .unwrap()
        .write_to(&mut f, ImageFormat::Png)
        .unwrap();
}
*/
