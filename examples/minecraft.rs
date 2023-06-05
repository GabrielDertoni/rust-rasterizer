#![feature(portable_simd, slice_as_chunks)]

use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::{LogicalSize, PhysicalPosition},
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::{CursorGrabMode, WindowBuilder},
};

use std::time::{Duration, Instant};
use std::ops::Range;

use rasterization::{
    texture::{BorrowedMutTexture, BorrowedTexture}, clear_color,
    obj::{Material, Obj},
    prim3d, shaders,
    utils::{Camera, FpvCamera},
    vec::{Mat4x4, Vec2, Vec3},
    vertex_shader_simd, IntoSimd, VertBuf, Vertex, VertexBuf,
};

struct TexturedSubset {
    material_idx: usize,
    range: Range<usize>,
}

pub struct World {
    vert_buf: VertBuf,
    index_buf: Vec<[usize; 3]>,
    materials: Vec<Material>,
    texture_subsets: Vec<TexturedSubset>,
    alpha_map: Vec<i8>,
    depth_buf: Vec<f32>,
    shadow_map: Vec<f32>,
    camera: FpvCamera,
    last_render_times: [Duration; 10],
    is_paused: bool,
    use_filter: bool,
    pub axis: Vec3,
    pub enable_logging: bool,
    enable_shadows: bool,
    model: Mat4x4,
    width: u32,
    height: u32,
}

impl World {
    pub fn new(width: u32, height: u32, model: &str) -> Self {
        let mut obj = Obj::load(model.as_ref()).unwrap();
        if !obj.has_normals() {
            obj.compute_normals();
        }

        let (vert_buf, index_buf) = VertBuf::from_obj(&obj);

        println!("Initially had {} positions, {} normals and {} uvs. Now converted into {} distict vertices", obj.verts.len(), obj.normals.len(), obj.uvs.len(), vert_buf.len());

        let n_vertices = vert_buf.len();
        let ratio = width as f32 / height as f32;
        let texture = &obj.materials[0].map_kd;
        let mut alpha_map = vec![0_i8; (texture.width() * texture.height()) as usize];
        for y in 0..texture.height() {
            for x in 0..texture.width() {
                alpha_map[(y * texture.width() + x) as usize] = texture.get_pixel(x, y).0[3] as i8;
            }
        }

        let mut texture_subsets: Vec<TexturedSubset> = obj.use_material.into_iter()
            .map(|usemtl| {
                let Some(material_idx) = obj.materials.iter()
                    .position(|mtl| mtl.name == usemtl.name) else {
                        let names_found = obj.materials.iter().map(|mtl| mtl.name.as_str()).collect::<Vec<_>>();
                        panic!("could not find material {} in .obj, found {:?}", usemtl.name, names_found);
                    };
                TexturedSubset {
                    material_idx,
                    range: usemtl.range,
                }
            })
            .collect();
        if texture_subsets.len() == 0 {
            texture_subsets.push(TexturedSubset {
                material_idx: 0,
                range: 0..index_buf.len(),
            });
        }
        World {
            vert_buf,
            index_buf,
            materials: obj.materials,
            texture_subsets,
            alpha_map,
            depth_buf: vec![0.0; (width * height) as usize],
            shadow_map: vec![0.0; (width * height) as usize],
            // camera: FpvCamera {
            //     position: Vec3::from([-7.4, 30.8, 12.7]),
            //     up: Vec3::from([0., 0., 1.]),
            //     pitch: 87.4295,
            //     yaw: -180.382,
            //     sensitivity: 0.05,
            //     speed: 5.,
            //     fovy: 39.6,
            //     ratio,
            // },
            camera: FpvCamera {
                position: Vec3::from([5., 7., -4.]),
                up: Vec3::from([0., 0., 1.]),
                pitch: 92.,
                yaw: 80.,
                sensitivity: 0.05,
                speed: 5.,
                fovy: 39.6,
                ratio,
            },
            last_render_times: [Duration::ZERO; 10],
            is_paused: true,
            use_filter: false,
            axis: Vec3::zero(),
            enable_logging: true,
            enable_shadows: false,
            model: Mat4x4::identity(),
            width,
            height,
        }
    }

    pub fn render(&mut self, buf: &mut [u8], dt: Duration) {
        let start = Instant::now();
        let (pixels, _) = buf.as_chunks_mut::<4>();
        let mut pixels = BorrowedMutTexture::from_mut_slice(self.width as usize, self.height as usize, pixels);

        self.camera.move_delta(self.axis * dt.as_secs_f32());

        let fog_color = 0x61b7e8ff;
        clear_color(pixels.borrow_mut(), 0x61b7e8ff);

        let draw_timer = Instant::now();

        if self.enable_shadows {
            // self.render_with_shadows(pixels);
        } else {
            self.render_without_shadows(pixels);
        }

        if self.enable_logging {
            let draw_time = draw_timer.elapsed();
            println!("draw time: {draw_time:?}");

            self.last_render_times.rotate_left(1);
            self.last_render_times[self.last_render_times.len() - 1] = start.elapsed();
            let render_time = self.last_render_times.iter().sum::<Duration>()
                / self.last_render_times.len() as u32;
            println!("render time: {render_time:?}");
            println!("FPS: {}", 1. / render_time.as_secs_f32());
        }
    }

    fn render_without_shadows(&mut self, pixels: BorrowedMutTexture<[u8; 4]>) {
        use rasterization::pipeline::Pipeline;
        use rasterization::vec::Vec;

        let fog_color = 0x61b7e8ff_u32;
        let fog_color = Vec::from(fog_color.to_be_bytes()).map(|chan| chan as f32 / 255.);
        let mut depth_buf = BorrowedMutTexture::from_mut_slice(
            self.width as usize,
            self.height as usize,
            &mut self.depth_buf,
        );

        depth_buf.as_slice_mut().fill(1.0);

        let near = 0.1;
        let far = 50.;

        let camera_transform = self.camera.transform_matrix(near, far);

        let vert_shader = shaders::textured::TexturedVertexShader::new(camera_transform);
        let mut pipeline = Pipeline::new(&self.vert_buf, &self.index_buf);

        pipeline.set_color_buffer(pixels);
        pipeline.set_depth_buffer(depth_buf);
        let mut pipeline = pipeline.process_vertices(&vert_shader);

        for subset in &self.texture_subsets {
            let texture = &self.materials[subset.material_idx].map_kd;
            let (texture_pixels, _) = texture.as_chunks::<4>();
            let texture = BorrowedTexture::from_slice(
                texture.width() as usize,
                texture.height() as usize,
                texture_pixels,
            );
            let frag_shader = shaders::textured::TexturedFragmentShader::new(texture);

            // let depth_start = Instant::now();
            // prim3d::draw_triangles_depth_specialized(
            //     &self.vert_buf.positions,
            //     &self.index_buf,
            //     camera_transform,
            //     depth_buf.borrow(),
            //     &mut self.render_ctx,
            //     1e-4,
            // );
            // println!("Rendering depth took {:?}", depth_start.elapsed());

            pipeline.draw(subset.range.clone(), &frag_shader);
    
            // metrics.combine(prim3d::draw_triangles_basic(
            //     &self.vert_buf,
            //     &self.index_buf[subset.range.clone()],
            //     &vert_shader,
            //     &move |attr: shaders::textured::TexturedAttributes| {
            //         let [u, v] = attr.uv.to_array();
            //         let u = u % 1.;
            //         let u = if u < 0. { 1. + u } else { u };
            //         let v = v % 1.;
            //         let v = if v < 0. { 1. + v } else { v };
            //         let x = (u * texture.width as f32) as usize;
            //         let y = ((1. - v) * texture.height as f32) as usize;
            //         let x = x.clamp(0, texture.width-1);
            //         let y = y.clamp(0, texture.height-1);
            //         let z = (1. / attr.position_ndc.w - near) / (far - near);
            //         let z = z.powi(4);
            //         let tex = Vec::from(texture[(x, y)]).map(|chan| chan as f32 / 255.);
            //         let color = tex * (1. - z) + fog_color * z;
            //         Vec4::from([color.x, color.y, color.z, tex.w])
            //     },
            //     pixels.borrow(),
            //     depth_buf.borrow(),
            //     &mut self.render_ctx,
            // ));
    
            // for y in 0..pixels.height {
            //     for x in 0..pixels.width {
            //         let depth = depth_buf[(x, y)].clamp(-1., 1.);
            //         let z =
            //             1. / ((depth - (far + near) / (far - near)) * (far - near) / (2. * far * near));
            //         let color = (z + near) / (near - far);
            //         let color = color * 255.;
            //         let color = color as u8;
            //         pixels[(x, y)] = [color, color, color, 0xff];
            //     }
            // }
        }
        pipeline.finalize();
        let metrics = pipeline.get_metrics();
        println!("{metrics}");
    }

    pub fn toggle_pause(&mut self) {
        self.is_paused = !self.is_paused;
    }

    pub fn toggle_use_filter(&mut self) {
        self.use_filter = !self.use_filter;
    }

    pub fn update_cursor(&mut self, dx: f32, dy: f32) {
        self.camera.rotate_delta(dy, dx);
    }
}

// const WIDTH: u32 = 480;
// const HEIGHT: u32 = 480;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let model = args.get(0).map(|s| s.as_str()).unwrap_or("teapot.obj");
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("rasterizer")
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let wsize = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(wsize.width, wsize.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture).unwrap()
    };

    let mut world = World::new(WIDTH, HEIGHT, model);

    let core_id = core_affinity::get_core_ids().unwrap()[0];
    core_affinity::set_for_current(core_id);

    let mut start = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        if window.has_focus() {
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .unwrap();
            window.set_cursor_visible(false);
            window
                .set_cursor_position(PhysicalPosition::new(wsize.width / 2, wsize.height / 2))
                .unwrap();
        } else {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
            window.set_cursor_visible(true);
        }

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
                (ElementState::Pressed, VirtualKeyCode::Escape) => control_flow.set_exit(),
                (ElementState::Pressed, VirtualKeyCode::Space) => world.toggle_pause(),
                (ElementState::Pressed, VirtualKeyCode::F) => world.toggle_use_filter(),
                (ElementState::Pressed, VirtualKeyCode::V) => world.axis.z = 1.0,
                (ElementState::Pressed, VirtualKeyCode::C) => world.axis.z = -1.0,
                (ElementState::Released, VirtualKeyCode::V | VirtualKeyCode::C) => {
                    world.axis.z = 0.0
                }
                (ElementState::Pressed, VirtualKeyCode::W) => world.axis.y = 1.0,
                (ElementState::Pressed, VirtualKeyCode::S) => world.axis.y = -1.0,
                (ElementState::Released, VirtualKeyCode::W | VirtualKeyCode::S) => {
                    world.axis.y = 0.0
                }
                (ElementState::Pressed, VirtualKeyCode::A) => world.axis.x = -1.0,
                (ElementState::Pressed, VirtualKeyCode::D) => world.axis.x = 1.0,
                (ElementState::Released, VirtualKeyCode::A | VirtualKeyCode::D) => {
                    world.axis.x = 0.0
                }
                (ElementState::Pressed, VirtualKeyCode::Right) => world.update_cursor(20., 0.),
                (ElementState::Pressed, VirtualKeyCode::Left) => world.update_cursor(-20., 0.),
                (ElementState::Pressed, VirtualKeyCode::Up) => world.update_cursor(0., 20.),
                (ElementState::Pressed, VirtualKeyCode::Down) => world.update_cursor(0., -20.),
                (ElementState::Pressed, VirtualKeyCode::H) => {
                    world.enable_shadows = !world.enable_shadows
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                let dt = start.elapsed();
                start = Instant::now();
                world.render(pixels.frame_mut(), dt);
                // std::thread::sleep(std::time::Duration::from_millis(200));
                window.request_redraw()
            }
            Event::RedrawRequested(_) => pixels.render().unwrap(),
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } => {
                world.update_cursor(dx as f32, dy as f32);
            }
            _ => (),
        }
    });
}
