#![feature(portable_simd, slice_as_chunks)]

use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::{LogicalSize, PhysicalPosition},
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::{CursorGrabMode, WindowBuilder},
};

use std::time::{Duration, Instant};

use rasterization::{
    buf, clear_color,
    obj::{Obj, Material},
    prim3d, shaders,
    utils::{Camera, FpvCamera},
    vec::{Mat4x4, Vec2, Vec3},
    VertexBuf, VertBuf, Vertex,
};

pub struct World {
    vert_buf: VertBuf,
    index_buf: Vec<[usize; 3]>,
    materials: Vec<Material>,
    depth_buf: Vec<f32>,
    shadow_map: Vec<f32>,
    camera: FpvCamera,
    last_render_times: [Duration; 10],
    theta: f32,
    scale: f32,
    is_paused: bool,
    use_filter: bool,
    pub axis: Vec3,
    pub enable_logging: bool,
    render_ctx: prim3d::RenderContext,
    width: u32,
    height: u32,
}

impl World {
    pub fn new(width: u32, height: u32, model: &str) -> Self {
        let mut obj = Obj::load(model.as_ref()).unwrap();
        if !obj.has_normals() {
            obj.compute_normals();
        }

        let mut vert_idxs_set = obj.tris.iter().copied().flatten().collect::<Vec<_>>();
        vert_idxs_set.sort_unstable();
        vert_idxs_set.dedup();
        let mut vert_buf = VertBuf::with_capacity(vert_idxs_set.len());

        for idxs in &vert_idxs_set {
            vert_buf.push(Vertex {
                position: obj.verts[idxs.position as usize],
                normal: obj.normals[idxs.normal as usize],
                uv: obj.uvs[idxs.uv as usize],
            });
        }

        let mut index_buf = Vec::with_capacity(obj.tris.len());
        for tri in &obj.tris {
            index_buf.push(tri.map(|v| vert_idxs_set.binary_search(&v).unwrap()));
        }

        println!("Initially had {} positions, {} normals and {} uvs. Now converted into {} distict vertices", obj.verts.len(), obj.normals.len(), obj.uvs.len(), vert_buf.len());

        let n_vertices = vert_buf.len();
        let ratio = width as f32 / height as f32;
        World {
            vert_buf,
            index_buf,
            materials: obj.materials,
            depth_buf: vec![0.0; (width * height) as usize],
            shadow_map: vec![0.0; (width * height) as usize],
            camera: FpvCamera {
                position: Vec3::from([0., -10., 4.]),
                up: Vec3::from([0., 0., 1.]),
                pitch: 70.,
                yaw: 0.,
                sensitivity: 0.05,
                speed: 5.,
                fovy: 39.6,
                ratio,
            },
            last_render_times: [Duration::ZERO; 10],
            theta: 0.0,
            scale: 1.0,
            is_paused: true,
            use_filter: false,
            axis: Vec3::zero(),
            enable_logging: true,
            render_ctx: prim3d::RenderContext::alloc(n_vertices),
            width,
            height,
        }
    }

    pub fn render(&mut self, buf: &mut [u8], dt: Duration) {
        let start = Instant::now();
        let (pixels, _) = buf.as_chunks_mut::<4>();
        let mut pixels = buf::PixelBuf::new(pixels, self.width as usize, self.height as usize);

        if !self.is_paused {
            self.theta -= dt.as_secs_f32() * std::f32::consts::PI / 4.0;
        }

        self.camera.move_delta(self.axis * dt.as_secs_f32());

        clear_color(pixels.borrow(), 0x00_00_00_ff);

        let mut depth_buf = buf::MatrixSliceMut::new(
            &mut self.depth_buf,
            self.width as usize,
            self.height as usize,
        );

        let draw_timer = Instant::now();

        let light_pos = Vec3::from([0., -3.77, 6.27]);
        let light_camera =
            Camera::from_blender(light_pos, Vec3::from([33.9, 0., 0.]), 45., 1., 1., 100.);

        let shadow_map = {
            self.shadow_map.fill(1.0);
            let mut shadow_buf = buf::MatrixSliceMut::new(
                &mut self.shadow_map,
                self.width as usize,
                self.height as usize,
            );

            let light_transform = light_camera.transform_matrix();
            prim3d::draw_triangles_depth(
                &self.vert_buf,
                &self.index_buf,
                &|vertex: Vertex| light_transform * vertex.position,
                shadow_buf.borrow(),
            );
            // Pretty dumb way to make a circular spotlight
            let radius = shadow_buf.width as f32 * 0.5;
            let circle_center =
                Vec2::from([shadow_buf.width as f32 / 2., shadow_buf.height as f32 / 2.]);
            for y in 0..shadow_buf.height {
                for x in 0..shadow_buf.width {
                    let p = Vec2::from([x as f32, y as f32]);
                    if (p - circle_center).mag_sq() >= radius * radius {
                        shadow_buf[(x, y)] = -1.0;
                    }
                }
            }
            buf::MatrixSlice::new(&self.shadow_map, depth_buf.width, depth_buf.height)
        };

        depth_buf.as_slice_mut().fill(1.0);

        let near = 0.1;
        let far = 100.;

        let vert_shader = shaders::lit::gouraud::VertexShader::new(
            self.camera.transform_matrix(near, far),
            light_camera.transform_matrix(),
            light_pos,
            Vec3::from([1., 1., 1.]),
        );

        /*
        let vert_shader = shaders::lit::LitVertexShader::new(
            self.camera.transform_matrix(near, far),
            light_camera.transform_matrix(),
        );
        */

        let texture = &self.materials[0].map_kd;
        let (texture_pixels, _) = texture.as_chunks::<4>();
        let texture = buf::MatrixSlice::new(
            texture_pixels,
            texture.width() as usize,
            texture.height() as usize,
        );

        let frag_shader = shaders::lit::gouraud::FragmentShader::new(
            texture,
            shadow_map.borrow(),
        );
        /*
        let frag_shader = shaders::lit::LitFragmentShader::new(
            self.camera.position,
            model,
            light_pos,
            Vec3::from([1., 1., 1.]),
            texture,
            shadow_map.borrow(),
        );
        */

        prim3d::draw_triangles(
            &self.vert_buf,
            &self.index_buf,
            &vert_shader,
            &frag_shader,
            pixels.borrow(),
            depth_buf.borrow(),
            &mut self.render_ctx,
        );

        if self.enable_logging {
            let mid_depth = depth_buf[(depth_buf.width / 2, depth_buf.height / 2)];
            println!("mid depth: {mid_depth}");

            let draw_time = draw_timer.elapsed();
            println!("draw time: {draw_time:?}");

            self.last_render_times.rotate_left(1);
            self.last_render_times[self.last_render_times.len() - 1] = start.elapsed();
            let render_time = self.last_render_times.iter().sum::<Duration>()
                / self.last_render_times.len() as u32;
            println!("render time: {render_time:?}");
        }
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

const WIDTH: u32 = 720;
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
            Event::WindowEvent {
                event:
                    WindowEvent::CursorMoved {
                        position,
                        ..
                    },
                ..
            } => {
                let position = Vec2::from([position.x as f32, position.y as f32]);
                let screen_center = Vec2::from([wsize.width as f32, wsize.height as f32]) / 2.;
                let delta = position - screen_center;
                world.update_cursor(delta.x, delta.y);
            }
            _ => (),
        }
    });
}
