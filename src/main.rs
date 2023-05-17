#![feature(slice_as_chunks, iter_next_chunk, portable_simd, pointer_is_aligned)]

use std::time::{Duration, Instant};
use std::simd::{Simd, Mask};

use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use rasterization::{
    FragShader,
    SimdAttr,
    VertBuf,
    clear_color,
    obj::Obj,
    buf,
    prim3d,
    vec::{self, Vec3, Mat4x4},
};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 480;

fn main() {
    // debug();
    // return;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model = args.get(0).map(|s| s.as_str()).unwrap_or("teapot.obj");
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

    let mut world = World::new(model);

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
                (ElementState::Pressed, VirtualKeyCode::F) => world.toggle_use_filter(),
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
    obj: Obj,
    depth_buf: Vec<f32>,
    camera_pos: Vec3,
    look_at: Vec3,
    last_render_times: [Duration; 10],
    theta: f32,
    scale: f32,
    is_paused: bool,
    use_filter: bool,
    axis: Vec3,
}

impl World {
    fn new(model: &str) -> Self {
        let mut obj = Obj::load(model.as_ref()).unwrap();
        if !obj.has_normals() {
            obj.compute_normals();
        }
        World {
            obj,
            depth_buf: vec![0.0; (WIDTH * HEIGHT) as usize],
            camera_pos: Vec3::from([0.0, 1.0, -1.0]),
            look_at: Vec3::from([0.0, 1.0, 0.0]),
            last_render_times: [Duration::ZERO; 10],
            theta: 0.0,
            scale: 15.0,
            is_paused: true,
            use_filter: false,
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

        let inc = self.axis * dt.as_secs_f32();

        self.scale += inc.z;
        self.camera_pos.x += inc.x;
        self.camera_pos.y += inc.y;
        self.look_at.x += inc.x;
        self.look_at.y += inc.y;

        let model = Mat4x4::rotation_x(self.theta) * Vec3::repeat(1.0/self.scale).to_scale();

        let eye = self.camera_pos;
        let up = Vec3::from([0.0, 1.0, 0.0]);

        let look = (self.look_at - eye).normalized();
        let right = up.cross(look).normalized();
        let up = right.cross(look).normalized();

        let view = Mat4x4::from([
            [right.x, right.y, right.z, -right.dot(eye)],
            [   up.x,    up.y,    up.z,    -up.dot(eye)],
            [ look.x,  look.y,  look.z,  -look.dot(eye)],
            [    0.0,     0.0,     0.0,             1.0],
        ]);

        let transform = pixels.ndc_to_screen() * view * model;

        let positions: Vec<_> = self
            .obj
            .verts
            .iter()
            .map(|&p| transform * p)
            .collect();

        let vert_buf = VertBuf {
            positions: &positions,
            normals: &self.obj.normals,
            uvs: &self.obj.uvs,
        };

        if self.use_filter {
            let shader = LinearFilteringFragShader::new(&self.obj.materials[0].map_kd);

            prim3d::draw_triangles_opt(
                &vert_buf,
                &self.obj.tris,
                &shader,
                pixels.borrow(),
                depth_buf.borrow(),
            );
        } else {
            let shader = TextureMappingFragShader::new(&self.obj.materials[0].map_kd);

            prim3d::draw_triangles_opt(
                &vert_buf,
                &self.obj.tris,
                &shader,
                pixels.borrow(),
                depth_buf.borrow(),
            );
        }

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

    fn toggle_use_filter(&mut self) {
        self.use_filter = !self.use_filter;
    }
}

pub struct TextureMappingFragShader<'a> {
    texture_width: u32,
    texture_height: u32,
    texture: &'a [u32],
}

impl<'a> TextureMappingFragShader<'a> {
    pub fn new(texture_img: &'a image::RgbaImage) -> Self {
        let texture_width = texture_img.width();
        let texture_height = texture_img.height();
        let texture: &[u8] = &*texture_img;
        let ptr = texture.as_ptr().cast::<u32>();
        assert!(ptr.is_aligned());
        // SAFETY: Pointer is aligned
        let texture = unsafe {
            std::slice::from_raw_parts(ptr, texture.len() / 4)
        };
        TextureMappingFragShader {
            texture_width,
            texture_height,
            texture,
        }
    }
}

impl<'a> FragShader<4> for TextureMappingFragShader<'a> {
    type SimdAttr = SimdAttr<4>;

    fn exec(&self, _mask: Mask<i32, 4>, attrs: SimdAttr<4>) -> vec::Vec<Simd<f32, 4>, 4> {
        let [u, v] = attrs.uv.to_array();
        let x = u * Simd::splat(self.texture_width  as f32);
        let y = (Simd::splat(1.0) - v) * Simd::splat(self.texture_height as f32);

        let idx = y.cast::<usize>() * Simd::splat(self.texture_width as usize) + x.cast::<usize>();

        vec::Vec::from(
            Simd::gather_or_default(&self.texture, idx)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes()))
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0))
    }
}

pub struct LinearFilteringFragShader<'a> {
    texture_width: u32,
    texture_height: u32,
    texture: &'a [u32],
}

impl<'a> LinearFilteringFragShader<'a> {
    pub fn new(texture_img: &'a image::RgbaImage) -> Self {
        let texture_width = texture_img.width();
        let texture_height = texture_img.height();
        let texture: &[u8] = &*texture_img;
        let ptr = texture.as_ptr().cast::<u32>();
        assert!(ptr.is_aligned());
        // SAFETY: Pointer is aligned
        let texture = unsafe {
            std::slice::from_raw_parts(ptr, texture.len() / 4)
        };
        LinearFilteringFragShader {
            texture_width,
            texture_height,
            texture,
        }
    }
}

impl<'a> FragShader<4> for LinearFilteringFragShader<'a> {
    type SimdAttr = SimdAttr<4>;

    fn exec(&self, _mask: Mask<i32, 4>, attrs: SimdAttr<4>) -> vec::Vec<Simd<f32, 4>, 4> {
        use std::simd::SimdFloat;

        let [u, v] = attrs.uv.to_array();
        let x = u * Simd::splat(self.texture_width  as f32);
        let y = (Simd::splat(1.0) - v) * Simd::splat(self.texture_height as f32);

        let idx0 = y.cast::<usize>() * Simd::splat(self.texture_width as usize) + x.cast::<usize>();
        let idx1 = idx0 + Simd::splat(1);
        let idx2 = idx0 + Simd::splat(self.texture_width as usize);
        let idx3 = idx0 + Simd::splat(self.texture_width as usize + 1);

        let fx = x.cast::<i32>().cast::<f32>();
        let fy = y.cast::<i32>().cast::<f32>();

        let dx = x - fx;
        let dy = y - fy;

        let dist0 = dx.abs().simd_max(dy.abs());
        let dist1 = Simd::splat(1.0) - dx;
        let dist2 = Simd::splat(1.0) - dy;
        let dist3 = dist1.simd_max(dist2);

        let total = dist0 + dist1 + dist2 + dist3;

        let w0 = dist0 / total;
        let w1 = dist1 / total;
        let w2 = dist2 / total;
        let w3 = dist3 / total;

        let c0 = vec::Vec::from(
            Simd::gather_or_default(&self.texture, idx0)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes()))
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0));

        let c1 = vec::Vec::from(
            Simd::gather_or_default(&self.texture, idx1)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes()))
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0));

        let c2 = vec::Vec::from(
            Simd::gather_or_default(&self.texture, idx2)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes()))
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0));

        let c3 = vec::Vec::from(
            Simd::gather_or_default(&self.texture, idx3)
                .to_array()
                .map(|el| Simd::from(el.to_ne_bytes()))
        )
        .simd_transpose_4()
        .map_4(|el| el.cast::<f32>() / Simd::splat(255.0));

        c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3
    }
}

pub struct ShowNormalsFragShader;

impl FragShader<4> for ShowNormalsFragShader {
    type SimdAttr = SimdAttr<4>;

    fn exec(&self, _mask: Mask<i32, 4>, attrs: SimdAttr<4>) -> vec::Vec<Simd<f32, 4>, 4> {
        let n = attrs.normal;
        vec::Vec::from([n.x, n.y, n.z, Simd::splat(1.0)])
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
