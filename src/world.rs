use std::time::{Duration, Instant};

use crate::{
    buf, clear_color,
    frag_shaders::*,
    obj::Obj,
    prim3d,
    vec::{Mat4x4, Vec2, Vec3},
    VertBuf,
};

pub struct World {
    obj: Obj,
    depth_buf: Vec<f32>,
    camera: Camera,
    last_render_times: [Duration; 10],
    theta: f32,
    scale: f32,
    is_paused: bool,
    use_filter: bool,
    pub axis: Vec3,
    pub enable_logging: bool,
    cursor: Vec2,
    width: u32,
    height: u32,
}

impl World {
    pub fn new(width: u32, height: u32, model: &str) -> Self {
        let mut obj = Obj::load(model.as_ref()).unwrap();
        if !obj.has_normals() {
            obj.compute_normals();
        }
        World {
            obj,
            depth_buf: vec![0.0; (width * height) as usize],
            // camera: Camera::from_blender(Vec3::from([0., -10., 4.]), Vec3::from([70., 0., 0.])),
            camera: Camera {
                position: Vec3::from([0., 0., -10.]),
                target: Vec3::zero(),
                up: Vec3::from([0., 1., 1.]),
            },
            last_render_times: [Duration::ZERO; 10],
            // theta: -std::f32::consts::FRAC_PI_2,
            theta: 0.0,
            scale: 1.0,
            is_paused: true,
            use_filter: false,
            axis: Vec3::zero(),
            enable_logging: true,
            cursor: Vec2::zero(),
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

        clear_color(pixels.borrow(), 0x00_00_00_ff);

        let mut depth_buf = buf::MatrixSliceMut::new(
            &mut self.depth_buf,
            self.width as usize,
            self.height as usize,
        );

        for d in depth_buf.as_slice_mut() {
            *d = f32::MIN;
        }

        let inc = 5. * self.axis * dt.as_secs_f32();

        self.camera.position += inc;
        self.camera.target   += inc;

        let model = Mat4x4::rotation_y(self.theta)
            * Mat4x4::rotation_x(self.theta)
            * Vec3::repeat(1.0 / self.scale).to_scale();

        let view = self.camera.view_matrix();

        let ratio = self.width as f32 / self.height as f32;
        let proj = Mat4x4::perspective(ratio, 90.0, 1.0, 100.0);
        let transform = proj * view * model;

        let positions: Vec<_> = self.obj.verts.iter()
            .map(|&p| {
                let persp = transform * p;
                let proj = persp / persp.w;
                pixels.ndc_to_screen() * proj
            })
            .collect();

        let vert_buf = VertBuf {
            positions: &positions,
            normals: &self.obj.normals,
            uvs: &self.obj.uvs,
        };

        let cursor = Vec3::from([self.cursor.x, 1.0 - self.cursor.y, 0.0]);
        let mut light_dir = cursor - Vec3::from([0.5, 0.5, 0.0]);
        if light_dir.mag_sq() != 0.0 {
            light_dir.normalize();
        }

        let draw_timer = Instant::now();

        if self.obj.has_uvs() {
            if self.use_filter {
                /*
                let shader = LinearFilteringFragShader::new(&self.obj.materials[0].map_kd);

                prim3d::draw_triangles_opt(
                    &vert_buf,
                    &self.obj.tris,
                    &shader,
                    pixels.borrow(),
                    depth_buf.borrow(),
                );
                */
            } else {
                let shader =
                    TextureMappingFragShader::new(light_dir, model, &self.obj.materials[0].map_kd);

                prim3d::draw_triangles_opt(
                    &vert_buf,
                    &self.obj.tris,
                    &shader,
                    pixels.borrow(),
                    depth_buf.borrow(),
                );
            }
        } else {
            let shader = ShowNormalsFragShader::new();
            // let shader = FakeLitFragShader::new(light_dir, model);

            prim3d::draw_triangles_opt(
                &vert_buf,
                &self.obj.tris,
                &shader,
                pixels.borrow(),
                depth_buf.borrow(),
            );
        }

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

    pub fn update_cursor(&mut self, x: f32, y: f32) {
        self.cursor.x = x;
        self.cursor.y = y;
    }
}

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
}

impl Camera {
    pub fn from_blender(position: Vec3, rotation_deg: Vec3) -> Self {
        let up = Vec3::from([0., 0., 1.]);
        let rotation = rotation_deg.map(|el| el.to_radians());
        Camera {
            position,
            target: (rotation.to_rotation() * (position - up).to_hom()).xyz(),
            up,
        }
    }

    pub fn view_matrix(&self) -> Mat4x4 {
        Mat4x4::look_at(self.position, self.target, self.up)
    }
}
