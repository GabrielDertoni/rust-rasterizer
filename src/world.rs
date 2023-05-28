use std::time::{Duration, Instant};

use crate::{
    buf::{self, MatrixSlice},
    clear_color,
    frag_shaders::*,
    obj::Obj,
    prim3d,
    vec::{Mat4x4, Vec2, Vec3},
    VertBuf, VertShader,
};

pub struct World {
    obj: Obj,
    depth_buf: Vec<f32>,
    shadow_map: Vec<f32>,
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

        let ratio = width as f32 / height as f32;
        World {
            obj,
            depth_buf: vec![0.0; (width * height) as usize],
            shadow_map: vec![0.0; (width * height) as usize],
            camera: Camera::from_blender(
                Vec3::from([0., -10., 4.]),
                Vec3::from([70., 0., 0.]),
                39.6,
                ratio,
                0.1,
                100.,
            ),
            last_render_times: [Duration::ZERO; 10],
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

        let inc = 5. * self.axis * dt.as_secs_f32();

        self.camera.position += inc;
        self.camera.target += inc;

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
            Camera::from_blender(light_pos, Vec3::from([33.9, 0., 0.]), 45., 1., 1., 10.);

        let shadow_map = {
            self.shadow_map.fill(1.0);
            let shadow_buf = buf::MatrixSliceMut::new(
                &mut self.shadow_map,
                self.width as usize,
                self.height as usize,
            );

            struct VertexShader {
                transform: Mat4x4,
            }

            impl VertexShader {
                fn new(transform: Mat4x4) -> Self {
                    VertexShader { transform }
                }
            }

            impl crate::VertexShader<crate::Vertex> for VertexShader {
                type Attrs = ();

                fn exec(&self, vertex: crate::Vertex) -> (Self::Attrs, crate::vec::Vec4) {
                    ((), self.transform * vertex.position)
                }
            }

            prim3d::draw_triangles_depth(
                &vert_buf,
                &self.obj.tris,
                &VertexShader::new(light_camera.transform_matrix()),
                shadow_buf,
            );
            MatrixSlice::new(&self.shadow_map, depth_buf.width, depth_buf.height)
        };

        depth_buf.as_slice_mut().fill(1.0);
        /*
        if self.obj.has_uvs() {
            let shader = TextureMappingFragShader::new(&self.obj.materials[0].map_kd);

            prim3d::draw_triangles_opt(
                &vert_buf,
                &self.obj.tris,
                &shader,
                pixels.borrow(),
                depth_buf.borrow(),
            );
        } else {
            // let shader = ShowNormalsFragShader::new();
            // let shader = FakeLitFragShader::new(light_dir, model);

            prim3d::draw_triangles_opt(
                &vert_buf,
                &self.obj.tris,
                &shader,
                pixels.borrow(),
                depth_buf.borrow(),
            );
        }
        */

        // let vert_shader = VertShader::new(light_camera.transform_matrix());

        let cursor_pos =
            depth_buf.ndc_to_screen() * Vec3::from([self.cursor.x, self.cursor.y, 0.]).to_hom();
        let cursor_pos = Vec2::from([cursor_pos.x, cursor_pos.y]).to_i32();

        let vert_shader = VertShader::new(self.camera.transform_matrix());
        let frag_shader = LitFragShader::new(
            model,
            light_pos,
            light_camera.transform_matrix(),
            Vec3::from([1., 1., 1.]),
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

        // for y in 0..pixels.height {
        //     for x in 0..pixels.width {
        //         let value = shadow_map[(x, y)];
        //         assert!((-1.0..=1.).contains(&value));
        //         let value = ((1. - value) * 1_000.) as u8;
        //         pixels[(x, y)] = [value, value, value, 0xff];
        //     }
        // }

        if self.enable_logging {
            println!(
                "depth at cursor ({}, {}): {}",
                cursor_pos.x,
                cursor_pos.y,
                depth_buf[(cursor_pos.x as usize, cursor_pos.y as usize)]
            );

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
}
