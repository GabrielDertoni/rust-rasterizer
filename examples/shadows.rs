#![feature(portable_simd, slice_as_chunks)]

use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::{LogicalSize, PhysicalPosition},
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent, DeviceEvent},
    event_loop::EventLoop,
    window::{CursorGrabMode, WindowBuilder},
};

use std::time::{Duration, Instant};

use rasterization::{
    buf,
    clear_color,
    obj::Obj,
    prim3d, shaders,
    vec::{Mat3x3, Mat4x4, Vec2, Vec3},
    VertBuf, Vertex,
};

pub struct World {
    obj: Obj,
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
    width: u32,
    height: u32,
}

impl World {
    pub fn new(width: u32, height: u32, model: &str) -> Self {
        let mut obj = Obj::load(model.as_ref()).unwrap();
        if !obj.has_normals() {
            obj.compute_normals();
        }

        let ratio = width as f32 / height as f32;
        World {
            obj,
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

        let model = Mat4x4::rotation_y(self.theta)
            * Mat4x4::rotation_x(self.theta)
            * Vec3::repeat(1.0 / self.scale).to_scale();

        let positions: Vec<_> = self.obj.verts.iter().map(|&p| model * p).collect();

        let vert_buf = VertBuf {
            positions: &positions,
            normals: &self.obj.normals,
            uvs: &self.obj.uvs,
        };

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
                &vert_buf,
                &self.obj.tris,
                &|vertex: Vertex| ((), light_transform * vertex.position),
                shadow_buf.borrow(),
            );
            // Pretty dumb way to make a circular spotlight
            let radius = shadow_buf.width as f32 * 0.5;
            let circle_center = Vec2::from([shadow_buf.width as f32 / 2., shadow_buf.height as f32 / 2.]);
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

        let vert_shader = shaders::lit::LitVertexShader::new(
            self.camera.transform_matrix(near, far),
            light_camera.transform_matrix(),
        );
        let texture = &self.obj.materials[0].map_kd;
        let (texture_pixels, _) = texture.as_chunks::<4>();
        let texture = buf::MatrixSlice::new(texture_pixels, texture.width() as usize, texture.height() as usize);
        let frag_shader = shaders::lit::LitFragmentShader::new(
            self.camera.position,
            model,
            light_pos,
            Vec3::from([1., 1., 1.]),
            texture,
            shadow_map.borrow(),
        );

        prim3d::draw_triangles(
            &vert_buf,
            &self.obj.tris,
            &vert_shader,
            &frag_shader,
            pixels.borrow(),
            depth_buf.borrow(),
        );

        if self.enable_logging {
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

pub struct FpvCamera {
    pub position: Vec3,
    pub up: Vec3,
    /// Pitch measured in degrees
    pub pitch: f32,
    /// Yaw measured in degrees
    pub yaw: f32,
    pub sensitivity: f32,
    pub speed: f32,
    pub fovy: f32,
    pub ratio: f32,
}

impl FpvCamera {
    /// Rotate the camera by some delta in pitch and yaw, measured in degrees.
    pub fn rotate_delta(&mut self, delta_pitch: f32, delta_yaw: f32) {
        self.pitch -= delta_pitch * self.sensitivity;
        self.pitch = self.pitch.clamp(-90., 90.);
        self.yaw -= delta_yaw * self.sensitivity;
    }

    /// Move the camera by `axis`. The vector `axis` has coordinates `x` for sideways motion
    /// (positive goes to the right), `y` for going forward and backwards (positive goes forward)
    /// and `z` for going up and down (positive goes up).
    pub fn move_delta(&mut self, axis: Vec3) {
        self.position += self.change_of_basis() * (self.speed * axis);
    }

    pub fn right(&self) -> Vec3 {
        self.front().cross(self.up).normalized()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.front())
    }

    pub fn front(&self) -> Vec3 {
        /*
        let yaw = self.yaw.to_radians();
        let pitch = -self.pitch.to_radians();
        Vec3::from([
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        ])
        */
        let yaw = -self.yaw.to_radians();
        let pitch = self.pitch.to_radians();
        Vec3::from([
            pitch.sin() * yaw.sin(),
            pitch.sin() * yaw.cos(),
            -pitch.cos(),
        ])
    }

    pub fn view_matrix(&self) -> Mat4x4 {
        Mat4x4::look_at(self.position, self.position + self.front(), self.up)
    }

    pub fn projection_matrix(&self, near: f32, far: f32) -> Mat4x4 {
        Mat4x4::perspective(self.ratio, self.fovy, near, far)
    }

    pub fn transform_matrix(&self, near: f32, far: f32) -> Mat4x4 {
        self.projection_matrix(near, far) * self.view_matrix()
    }

    // Change of basis matrix that allows changing from a vector in "camera" space, to world space.
    // Camera space has x coordinates going to the right, y coordinates going forward and z coordinates
    // going up.
    fn change_of_basis(&self) -> Mat3x3 {
        let front = self.front();
        let right = front.cross(self.up).normalized();
        let up = right.cross(front);
        Mat3x3::from([
            [right.x, front.x, up.x],
            [right.y, front.y, up.y],
            [right.z, front.z, up.z],
        ])
    }
}

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fovy: f32,
    pub ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn from_blender(
        position: Vec3,
        rotation_deg: Vec3,
        fovy: f32,
        ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let up = Vec3::from([0., 0., 1.]);
        let rotation = rotation_deg.map(|el| el.to_radians());
        Camera {
            position,
            target: position + (rotation.to_rotation() * (-up).to_hom()).xyz(),
            up,
            fovy,
            ratio,
            near,
            far,
        }
    }

    pub fn view_matrix(&self) -> Mat4x4 {
        Mat4x4::look_at(self.position, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> Mat4x4 {
        Mat4x4::perspective(self.ratio, self.fovy, self.near, self.far)
    }

    pub fn transform_matrix(&self) -> Mat4x4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn right(&self) -> Vec3 {
        self.front().cross(self.up).normalized()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.front())
    }

    pub fn front(&self) -> Vec3 {
        (self.target - self.position).normalized()
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

    let mut start = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        if window.has_focus() {
            window
                .set_cursor_grab(CursorGrabMode::Confined)
                .unwrap();
            window.set_cursor_visible(false);
            window.set_cursor_position(PhysicalPosition::new(wsize.width / 2, wsize.height / 2)).unwrap();
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
