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
    obj,
    buf,
    prim3d,
    vec::{self, Vec3, Vec4, Mat4x4},
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
                (ElementState::Pressed, VirtualKeyCode::W) => world.axis.y = 1.0,
                (ElementState::Released, VirtualKeyCode::W) => world.axis.y = 0.0,
                (ElementState::Pressed, VirtualKeyCode::S) => world.axis.y = -1.0,
                (ElementState::Released, VirtualKeyCode::S) => world.axis.y = 0.0,
                (ElementState::Pressed, VirtualKeyCode::A) => world.axis.x = 1.0,
                (ElementState::Released, VirtualKeyCode::A) => world.axis.x = 0.0,
                (ElementState::Pressed, VirtualKeyCode::D) => world.axis.x = -1.0,
                (ElementState::Released, VirtualKeyCode::D) => world.axis.x = 0.0,
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
    depth_buf: Vec<f32>,
    camera_pos: Vec3,
    theta: f32,
    is_paused: bool,
    axis: Vec2,
}

impl World {
    fn new() -> Self {
        let obj::Obj { vert, tris } = obj::load_obj("teapot.obj".as_ref()).unwrap();
        let vert = compute_normals(&vert, &tris);
        World {
            tris,
            vert,
            depth_buf: vec![0.0; (WIDTH * HEIGHT) as usize],
            camera_pos: Vec3::from([0.0, 0.0, -1.0]),
            theta: 0.0,
>>>>>>> origin/main
            is_paused: true,
            axis: Vec2::zero(),
        }
    }

    fn render(&mut self, buf: &mut [u8], dt: Duration) {
        let start = Instant::now();
        let (pixels, _) = buf.as_chunks_mut::<4>();
        let mut pixels = buf::PixelBuf::new(pixels, WIDTH as usize, HEIGHT as usize);

        if !self.is_paused {
            self.theta += dt.as_secs_f32() * std::f32::consts::PI / 4.0;
        }

        clear_color(pixels.borrow(), 0x00_00_00_ff);

        let mut depth_buf =
            buf::MatrixSliceMut::new(&mut self.depth_buf, WIDTH as usize, HEIGHT as usize);

        for d in depth_buf.as_slice_mut() {
            *d = f32::MIN;
        }

        let model = Mat4x4::rotation_x(self.theta);

        self.camera_pos.x += self.axis.x * dt.as_secs_f32();
        self.camera_pos.y += self.axis.y * dt.as_secs_f32();

        let look = (self.camera_pos - look_at).normalize();
        let right = look.cross(camera_up);
        let up = right.cross(look);

        let view = Mat4x4::from([
            [right.x, right.y, right.z, -self.camera_pos.x],
            [  up.x,     up.y,    up.z, -self.camera_pos.y],
            [look.x,  look.y,   look.z, -self.camera_pos.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        let vert: Vec<_> = self
            .vert
            .iter()
            .map(|v| Vert {
                pos: pixels.ndc_to_screen(view * model * v.pos),
                normal: v.normal,
            })
            .collect();

        prim3d::draw_triangles_opt(
            &vert,
            &self.tris,
            |_mask, n| vec::Vec::from([n.x, n.y, n.z, Simd::splat(1.0)]),
            pixels.borrow(),
            depth_buf.borrow(),
        );
        println!("render time: {:?}", start.elapsed());
    }

    fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
    }
}

fn debug() {
    use image::{ImageBuffer, ImageFormat, Rgba, Luma};

    use rasterization::vec::Vec4x4;

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
    let obj = obj::load_obj("teapot.obj".as_ref()).unwrap();
    let mut vert = compute_normals(&obj.vert, &obj.tris);

    for v in &mut vert {
        v.pos = pixels.ndc_to_screen(v.pos);
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
