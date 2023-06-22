#![feature(portable_simd, slice_as_chunks, vec_into_raw_parts)]
#![windows_subsystem = "windows"]

mod shaders;

use notify::{
    event::{Event as NotifyEvent, EventKind as NotifyEventKind},
    Watcher,
};
use pixels::{Pixels, SurfaceTexture};
use raster_egui::{Cmd, Response};
use winit::{
    dpi::{LogicalSize, PhysicalPosition},
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
    },
    event_loop::EventLoopBuilder,
    window::{CursorGrabMode, Window, WindowBuilder},
};

use std::ops::Range;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use rasterization::{
    clear_color,
    config::{
        CullingMode, Light, RasterizerConfig, RasterizerImplementation, RenderingConfig, Scene,
    },
    math::Size,
    math_utils::{color_to_u32, remap, srgb_to_linear_f32},
    obj::{Material, Obj},
    pipeline::PipelineMode,
    simd_config::LANES,
    texture::{BorrowedMutTexture, BorrowedTexture},
    utils::{FpsCounter, FpvCamera},
    vec::{Mat4x4, Vec3},
    vert_buf::VertBuf,
    FragmentShaderSimd, VertexBuf,
};

struct Model {
    pub name: String,
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub textured_subsets: Vec<TexturedSubset>,
    pub vertex_range: Range<usize>,
}

impl Model {
    pub fn model_matrix(&self) -> Mat4x4 {
        Mat4x4::identity()
            .scale(self.scale)
            .rotate(self.rotation.map(|el| el.to_radians()))
            .translate(self.position)
    }
}

struct TexturedSubset {
    material_idx: usize,
    range: Range<usize>,
}

pub struct LightInfo {
    light: Light,
    // shadow_map: OwnedTexture<f32>,
}

impl LightInfo {
    pub fn model_matrix(&self) -> Mat4x4 {
        Mat4x4::identity().translate(self.light.position)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selected {
    Model(usize),
    Light(usize),
}

pub struct World {
    vert_buf: Vec<rasterization::Vertex>,
    index_buf: Vec<[usize; 3]>,
    materials: Vec<Material>,
    models: Vec<Model>,
    /// Index into `models`
    selected: Option<Selected>,
    lights: Vec<LightInfo>,
    ambient_light: Vec3,
    depth_buf: Vec<f32>,
    camera: FpvCamera,
    fps_counter: FpsCounter,
    axis: Vec3,
    enable_logging: bool,
    enable_shadows: bool,
    rendering_cfg: RenderingConfig,
    rasterizer_cfg: RasterizerConfig,
    // HACK: This option in just there to allow us to `.take()` it
    gui_ctx: Option<raster_egui::Context>,
    in_play_mode: bool,
    selection_mode: egui_gizmo::GizmoMode,
    t: f32,
}

impl World {
    pub fn new(scene: Scene, root_path: PathBuf, window: &Window) -> anyhow::Result<Self> {
        let mut vert_buf = VertBuf::new();
        let mut index_buf = Vec::new();
        let mut models = Vec::new();
        let mut materials = Vec::new();
        for model in &scene.models {
            let path = root_path.join(&model.path);
            let mut obj = Obj::load(path.as_ref()).unwrap();
            if !obj.has_normals() {
                obj.compute_normals();
            }

            let (vert, idx) = VertBuf::from_obj(&obj);
            let n_vert = vert.len();
            let n_faces = idx.len();

            println!(
                "Model {:?} initially had {} positions, {} normals and {} uvs. Now converted into {} distict vertices",
                model.path, obj.verts.len(), obj.normals.len(), obj.uvs.len(), n_vert
            );
            println!("Model has {} faces", obj.tris.len());

            let offset = index_buf.len();
            let material_idx_offset = materials.len();
            let mut textured_subsets: Vec<TexturedSubset> = obj.use_material.into_iter()
                .map(|usemtl| {
                    let Some(material_idx) = obj.materials.iter()
                        .position(|mtl| mtl.name == usemtl.name) else {
                            let names_found = obj.materials.iter().map(|mtl| mtl.name.as_str()).collect::<Vec<_>>();
                            panic!("could not find material {} in .obj, found {:?}", usemtl.name, names_found);
                        };
                    TexturedSubset {
                        material_idx: material_idx_offset + material_idx,
                        range: usemtl.range.start + offset .. usemtl.range.end + offset,
                    }
                })
                .collect();

            if textured_subsets.len() == 0 {
                textured_subsets.push(TexturedSubset {
                    material_idx: material_idx_offset,
                    range: offset..offset + n_faces,
                });
            }

            let off = vert_buf.len();
            vert_buf.extend(vert);
            // Need to shift over all indicies to point to where those vertices ended up in the combined vertex buffer.
            index_buf.extend(idx.into_iter().map(|tri| tri.map(|i| i + off)));
            models.push(Model {
                name: model.name.clone(),
                position: model.position,
                rotation: model.rotation,
                scale: model.scale,
                textured_subsets,
                vertex_range: off..off + n_vert,
            });
            materials.extend(obj.materials);
        }

        let lights = scene
            .lights
            .into_iter()
            .map(|light| {
                LightInfo {
                    light,
                }
            })
            .collect();

        let enable_shadows = match scene.config.get("enable-shadows") {
            Some(&toml::Value::Boolean(value)) => value,
            Some(_) => {
                return Err(anyhow::anyhow!(
                    "expected value of `config.enable-shadows` to be a boolean"
                ))
            }
            None => false,
        };

        let width = scene.rendering.width;
        let height = scene.rendering.height;
        let aspect_ratio = width as f32 / height as f32;

        let vert_buf = (0..vert_buf.len()).map(|i| vert_buf.index(i)).collect();

        let egui_ctx = egui::Context::default();
        egui_ctx.tessellation_options_mut(|options| {
            // options.feathering = false;
            options.feathering = true;
        });

        Ok(World {
            vert_buf,
            index_buf,
            materials,
            models,
            selected: None,
            lights,
            ambient_light: scene.ambient_light,
            depth_buf: vec![0.0; (width * height) as usize],
            camera: scene.camera.into_fpv(aspect_ratio),
            fps_counter: FpsCounter::new(),
            axis: Vec3::zero(),
            enable_logging: true,
            enable_shadows,
            rendering_cfg: scene.rendering,
            rasterizer_cfg: scene.rasterizer,
            gui_ctx: Some(raster_egui::Context::new(egui_ctx, window)),
            in_play_mode: true,
            selection_mode: egui_gizmo::GizmoMode::Rotate,
            t: 0.0,
        })
    }

    pub fn update(&mut self, dt: Duration) {
        use std::f32::consts::PI;

        self.t += dt.as_secs_f32();
        let x_pos = remap((self.t * PI / 5.0).cos(), 1.0..-1.0, -33.0..28.0);
        for model in self.models.iter_mut() {
            match model.name.as_str() {
                "steve" => {
                    model.position.x = x_pos;
                }
                "minecart" => {
                    model.position.x = x_pos;
                }
                "bee" => {
                    let center = Vec3::from([-21.0, 30.0, 36.0]);
                    let radius = 5.0;
                    model.position.x =
                        remap((self.t * PI / 5.0).cos(), -1.0..1.0, -radius..radius) + center.x;
                    model.position.y =
                        remap((self.t * PI / 5.0).sin(), -1.0..1.0, -radius..radius) + center.y;
                    let x_dir = remap(
                        (self.t * PI / 5.0).sin() * PI / 5.0,
                        -1.0..1.0,
                        -radius..radius,
                    );
                    let y_dir = remap(
                        -(self.t * PI / 5.0).cos() * PI / 5.0,
                        -1.0..1.0,
                        -radius..radius,
                    );
                    let z_ang = y_dir.atan2(x_dir);
                    model.rotation.z = z_ang.to_degrees() - 180.0;
                }
                "heart" => {
                    let z_pos = 12.5;
                    model.position.z = z_pos + 0.5 * (self.t * PI * 0.5).cos();
                    model.rotation.z = (self.t * 0.5).to_degrees();
                }
                _ => (),
            }
        }

        let colors = [
            (Vec3::from([196., 235., 255.]) / 255., Vec3::from([254., 190., 75.]) / 255.),
            (Vec3::from([228., 198., 159.]) / 255., Vec3::from([228., 198., 159.]) / 255.),
            (Vec3::from([41., 45., 84.]) / 255., Vec3::from([41., 45., 84.]) / 255.),
        ];

        if let Some(light) = self
            .lights
            .iter_mut()
            .find(|light| light.light.name == "sun")
        {
            let time_of_day = self.t / 15.0;
            let curr = (time_of_day as usize) % colors.len();
            let next = (curr + 1) % colors.len();
            let blending_factor = (self.t / 15.0).fract();
            let fog_color =
                colors[curr].0 * (1.0 - blending_factor) + colors[next].0 * blending_factor;
            let light_color =
                colors[curr].1 * (1.0 - blending_factor) + colors[next].1 * blending_factor;
            light.light.color = light_color;
            self.rendering_cfg.fog_color = (fog_color, 1.0).into();
        }

        self.camera.move_delta(self.axis * dt.as_secs_f32());
        self.camera.position.x = self.camera.position.x.clamp(-37.0, 38.0);
        self.camera.position.y = self.camera.position.y.clamp(-83.0, 82.0);
        self.camera.position.z = self.camera.position.z.clamp(1.0, 60.0);
    }

    pub fn render(&mut self, buf: &mut [u8], window: &Window) {
        let (pixels, _) = buf.as_chunks_mut::<4>();
        let mut pixels = BorrowedMutTexture::from_mut_slice(
            self.rendering_cfg.width as usize,
            self.rendering_cfg.height as usize,
            pixels,
        );

        clear_color(
            pixels.borrow_mut(),
            color_to_u32(self.rendering_cfg.fog_color),
        );

        let start = Instant::now();
        if self.enable_shadows {
            todo!();
            // self.render_with_shadows(pixels);
        } else {
            self.render_without_shadows(pixels.borrow_mut());
        }
        self.draw_gui(pixels, window);
        self.fps_counter.record_measurement(start.elapsed());

        if self.enable_logging {
            println!("render time: {:?}", self.fps_counter.mean_time());
        }
    }

    fn render_without_shadows(&mut self, pixels: BorrowedMutTexture<[u8; 4]>) {
        use rasterization::pipeline::Pipeline;

        let mut depth_buf = BorrowedMutTexture::from_mut_slice(
            self.rendering_cfg.width as usize,
            self.rendering_cfg.height as usize,
            &mut self.depth_buf,
        );

        depth_buf.as_slice_mut().fill(1.0);

        let view_projection = self
            .camera
            .transform_matrix(self.rendering_cfg.near, self.rendering_cfg.far);

        let start = std::time::Instant::now();

        let mut pipeline = Pipeline::new(
            &self.vert_buf,
            &self.index_buf,
            Size::new(self.rendering_cfg.width, self.rendering_cfg.height),
            PipelineMode::Simd3D,
        );

        pipeline
            .with_alpha_clip(self.rendering_cfg.alpha_clip)
            .with_culling(self.rendering_cfg.culling_mode)
            .with_mode(match self.rasterizer_cfg.implementation {
                RasterizerImplementation::BBoxSimd => PipelineMode::Simd3D,
                RasterizerImplementation::BBox => PipelineMode::Basic3D,
                RasterizerImplementation::Scanline => todo!(),
            });

        pipeline.set_color_buffer(pixels);
        pipeline.set_depth_buffer(depth_buf);

        for model in &self.models {
            let vert_shader =
                shaders::TexturedVertexShader::new(model.model_matrix(), view_projection);
            let mut pipeline =
                pipeline.process_vertices_par(&vert_shader, Some(model.vertex_range.clone()));

            for subset in &model.textured_subsets {
                let texture = &self.materials[subset.material_idx].map_kd;
                let (texture_pixels, _) = texture.as_chunks::<4>();
                let texture = BorrowedTexture::from_slice(
                    texture.width() as usize,
                    texture.height() as usize,
                    texture_pixels,
                );
                let frag_shader = shaders::TexturedFragmentShader::new(
                    self.rendering_cfg.near,
                    self.rendering_cfg.far,
                    srgb_to_linear_f32(self.rendering_cfg.fog_color.xyz()),
                    texture,
                    self.camera.position,
                    self.ambient_light,
                    &self.lights,
                );

                pipeline.draw_par(
                    subset.range.clone(),
                    FragmentShaderSimd::<_, LANES>::simd_impl(&frag_shader),
                );
            }
        }
        pipeline.finalize();
        println!("time to render models: {:?}", start.elapsed());
        let metrics = pipeline.get_metrics();
        println!("{metrics}");
    }

    pub fn update_cursor(&mut self, dx: f32, dy: f32) {
        self.camera.rotate_delta(dy, dx);
    }

    pub fn handle_event<T>(&mut self, event: Event<T>) -> Response {
        let mut is_mouse_click = false;
        let response = match &event {
            &Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(key),
                                ..
                            },
                        is_synthetic: false,
                        ..
                    },
                ..
            } => self.handle_key(state, key),
            &Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } if self.is_in_play_mode() => {
                self.update_cursor(dx as f32, dy as f32);
                Response::consumed()
            }
            &Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    },
                ..
            } => {
                is_mouse_click = true;
                Response::empty()
            }
            _ => Response::empty(),
        };

        if !response.consumed {
            let response = self.gui_ctx.as_mut().unwrap().handle_event(event);
            if !response.consumed && is_mouse_click {
                // self.has_focus = true;
            }
            // Suppress the redraw requests comming from the gui
            if let Some(Cmd::Redraw) = &response.cmd {
                return Response {
                    consumed: response.consumed,
                    cmd: None,
                };
            }
            response
        } else {
            response
        }
    }

    fn handle_key(&mut self, state: ElementState, key: VirtualKeyCode) -> Response {
        if !self.in_play_mode {
            if let (ElementState::Pressed, VirtualKeyCode::P) = (state, key) {
                self.in_play_mode = true;
            }
            return Response::empty();
        }
        match (state, key) {
            (ElementState::Pressed, VirtualKeyCode::Escape) => self.in_play_mode = false,
            (ElementState::Pressed, VirtualKeyCode::V) => self.axis.z = 1.0,
            (ElementState::Pressed, VirtualKeyCode::C) => self.axis.z = -1.0,
            (ElementState::Released, VirtualKeyCode::V | VirtualKeyCode::C) => self.axis.z = 0.0,
            (ElementState::Pressed, VirtualKeyCode::W) => self.axis.y = 1.0,
            (ElementState::Pressed, VirtualKeyCode::S) => self.axis.y = -1.0,
            (ElementState::Released, VirtualKeyCode::W | VirtualKeyCode::S) => self.axis.y = 0.0,
            (ElementState::Pressed, VirtualKeyCode::A) => self.axis.x = -1.0,
            (ElementState::Pressed, VirtualKeyCode::D) => self.axis.x = 1.0,
            (ElementState::Released, VirtualKeyCode::A | VirtualKeyCode::D) => self.axis.x = 0.0,
            (ElementState::Pressed, VirtualKeyCode::Right) => self.update_cursor(20., 0.),
            (ElementState::Pressed, VirtualKeyCode::Left) => self.update_cursor(-20., 0.),
            (ElementState::Pressed, VirtualKeyCode::Up) => self.update_cursor(0., 20.),
            (ElementState::Pressed, VirtualKeyCode::Down) => self.update_cursor(0., -20.),
            (ElementState::Pressed, VirtualKeyCode::H) => {
                self.enable_shadows = !self.enable_shadows
            }
            _ => return Response::empty(),
        };
        Response {
            consumed: true,
            cmd: None,
        }
    }

    pub fn draw_gui(&mut self, pixels: BorrowedMutTexture<[u8; 4]>, window: &Window) {
        let mut gui_ctx = self.gui_ctx.take().unwrap();
        gui_ctx.redraw(self, pixels, window);
        self.gui_ctx.replace(gui_ctx);
    }

    pub fn is_in_play_mode(&self) -> bool {
        self.in_play_mode
    }
}

impl raster_egui::App for World {
    fn update(&mut self, ctx: &egui::Context) {
        if let Some(selected) = self.selected {
            let model_matrix = match selected {
                Selected::Model(i) => self.models[i].model_matrix().cols(),
                Selected::Light(i) => self.lights[i].model_matrix().cols(),
            };
            let gizmo = egui_gizmo::Gizmo::new("selection")
                .view_matrix(self.camera.view_matrix().cols())
                .projection_matrix(
                    self.camera
                        .projection_matrix(self.rendering_cfg.near, self.rendering_cfg.far)
                        .cols(),
                )
                .model_matrix(model_matrix)
                .mode(self.selection_mode)
                .visuals(egui_gizmo::GizmoVisuals {
                    inactive_alpha: 0.8,
                    highlight_alpha: 1.0,
                    gizmo_size: 100.0,
                    ..Default::default()
                });

            let mut ui = egui::Ui::new(
                ctx.clone(),
                egui::LayerId::background(),
                egui::Id::new("gizmo"),
                egui::Rect::EVERYTHING,
                ctx.available_rect(),
            );
            if !self.is_in_play_mode() {
                if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::W)) {
                    self.selection_mode = egui_gizmo::GizmoMode::Translate;
                }
                if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::R)) {
                    self.selection_mode = egui_gizmo::GizmoMode::Rotate;
                }
                if ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::S)) {
                    self.selection_mode = egui_gizmo::GizmoMode::Scale;
                }
            }
            if let Some(response) = gizmo.interact(&mut ui) {
                let (x, y, z) = response.rotation.to_euler(glam::EulerRot::XYZ);
                match selected {
                    Selected::Model(i) => {
                        self.models[i].rotation = Vec3::from([x, y, z]);
                        self.models[i].position = Vec3::from(response.translation.to_array());
                        self.models[i].scale = Vec3::from(response.scale.to_array());
                    }
                    Selected::Light(i) => {
                        self.lights[i].light.position = Vec3::from(response.translation.to_array());
                    }
                }
            }
        }
        if self.is_in_play_mode() {
            return;
        }
        egui::Window::new("Info")
            .resizable(true)
            .constrain(false)
            .frame(
                egui::Frame::window(&ctx.style())
                    .shadow(egui::epaint::Shadow::NONE)
                    .rounding(egui::epaint::Rounding::none()),
            )
            .drag_bounds(egui::Rect::EVERYTHING)
            .show(ctx, |ui| {
                ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);

                ui.label("Camera");
                let pos = self.camera.position;
                ui.label(format!(
                    "position {{ x: {:.1}, y: {:.1}, z: {:.1} }}",
                    pos.x, pos.y, pos.z
                ));
                ui.label(format!("pitch {:.1}", self.camera.pitch));
                ui.label(format!("yaw {:.1}", self.camera.yaw));
                ui.separator();

                ui.label("Rendering stats");
                ui.label(format!("{:6.1} FPS", self.fps_counter.mean_fps()));
                ui.label(format!("{:6.1?} per frame", self.fps_counter.mean_time()));

                ui.collapsing("Options", |ui| {
                    egui::Grid::new("rendering options").show(ui, |ui| {
                        ui.label("Render distance");
                        ui.add(egui::Slider::new(&mut self.rendering_cfg.far, 1.0..=100.0));
                        ui.end_row();

                        ui.label("Fog color");
                        ui.color_edit_button_rgb(
                            self.rendering_cfg.fog_color.xyz_mut().as_mut_array(),
                        );
                        ui.end_row();

                        ui.label("Culling");
                        egui::ComboBox::from_id_source("culling")
                            .selected_text(self.rendering_cfg.culling_mode.to_string())
                            .show_ui(ui, |ui| {
                                for mode in CullingMode::enumerate() {
                                    ui.selectable_value(
                                        &mut self.rendering_cfg.culling_mode,
                                        mode,
                                        mode.to_string(),
                                    );
                                }
                            });
                        ui.end_row();

                        ui.label("Alpha clip");
                        let mut checked = self.rendering_cfg.alpha_clip.is_some();
                        ui.checkbox(&mut checked, "enable");
                        if checked {
                            if self.rendering_cfg.alpha_clip.is_none() {
                                self.rendering_cfg.alpha_clip.replace(0.0);
                            }
                            let alpha_clip = self.rendering_cfg.alpha_clip.as_mut().unwrap();
                            ui.add(egui::Slider::new(alpha_clip, 0.0..=1.0));
                        } else {
                            if self.rendering_cfg.alpha_clip.is_some() {
                                self.rendering_cfg.alpha_clip = None;
                            }
                        }
                    });
                });

                ui.collapsing("Scene", |ui| {
                    ui.label("Models");
                    for (i, model) in self.models.iter_mut().enumerate() {
                        ui.radio_value(&mut self.selected, Some(Selected::Model(i)), &model.name);
                    }

                    ui.label("Lights");
                    for (i, light) in self.lights.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.radio_value(
                                &mut self.selected,
                                Some(Selected::Light(i)),
                                &light.light.name,
                            );
                            ui.collapsing("more", |ui| {
                                ui.label("Color");
                                ui.color_edit_button_rgb(&mut light.light.color.as_mut_array());
                            });
                        });
                    }
                    if ui.button("Deselect").clicked() {
                        self.selected = None;
                    }
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("Ambient light");
                        ui.color_edit_button_rgb(&mut self.ambient_light.as_mut_array());
                    });
                });

                ui.collapsing("Scene statistics", |ui| {
                    ui.label(format!("{:6} vertices", self.vert_buf.len()));
                    ui.label(format!("{:6} faces", self.index_buf.len()));
                });
            });
    }
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let Some(scene_path) = args.get(0) else {
        return Err(anyhow::anyhow!("expected a command line argument indicating the scene to load, but got none"));
    };
    let scene_path = scene_path.parse::<PathBuf>()?;
    let scene = Scene::load_toml(&scene_path)?;
    let event_loop = EventLoopBuilder::<NotifyEvent>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_title("rasterizer")
        .with_inner_size(LogicalSize::new(
            scene.rendering.width as u32,
            scene.rendering.height as u32,
        ))
        .build(&event_loop)
        .unwrap();

    let wsize = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(wsize.width, wsize.height, &window);
        Pixels::new(
            scene.rendering.width as u32,
            scene.rendering.height as u32,
            surface_texture,
        )
        .unwrap()
    };

    let fps = scene.rendering.fps;
    let mut world = World::new(scene, scene_path.parent().unwrap().to_owned(), &window)?;

    let event_loop_sender = event_loop.create_proxy();
    let mut watcher = notify::recommended_watcher(move |res| match res {
        Ok(event) => _ = event_loop_sender.send_event(event),
        Err(e) => eprintln!("watch error: {e}"),
    })?;
    watcher.watch(scene_path.as_ref(), notify::RecursiveMode::NonRecursive)?;

    let _guard = play_sound();

    let target_render_time = fps.map(|fps| Duration::from_secs_f32(1. / fps));

    let mut next_frame = Instant::now();

    let mut start = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        if target_render_time.is_some() {
            if next_frame < Instant::now() {
                window.request_redraw();
            } else {
                control_flow.set_wait_until(next_frame);
            }
        } else {
            control_flow.set_poll();
        }

        if window.has_focus() && world.is_in_play_mode() {
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
            Event::MainEventsCleared => {
                if target_render_time.is_none() {
                    window.request_redraw()
                }
            }
            Event::RedrawRequested(_) => {
                let dt = start.elapsed();
                start = Instant::now();

                world.update(dt);
                world.render(pixels.frame_mut(), &window);
                pixels.render().unwrap();

                println!("FPS: {:.1}", 1. / dt.as_secs_f32());

                if let Some(target_render_time) = target_render_time {
                    next_frame = start + target_render_time;
                    control_flow.set_wait_until(next_frame);
                }
            }
            Event::UserEvent(NotifyEvent {
                kind: NotifyEventKind::Create(_) | NotifyEventKind::Modify(_),
                ..
            }) => {
                let scene = match Scene::load_toml(&scene_path) {
                    Ok(scene) => scene,
                    Err(e) => {
                        eprintln!("[ERROR] while trying to hot reload scene: {e}");
                        return;
                    }
                };
                world = match World::new(scene, scene_path.parent().unwrap().to_owned(), &window) {
                    Ok(world) => world,
                    Err(e) => {
                        eprintln!("[ERROR] while trying to create world and hot reload scene: {e}");
                        return;
                    }
                };
            }
            event => {
                let response = world.handle_event(event);
                if let Some(cmd) = response.cmd {
                    match cmd {
                        Cmd::Resize(wsize) => {
                            pixels.resize_buffer(wsize.width, wsize.height).unwrap();
                            pixels.resize_surface(wsize.width, wsize.height).unwrap();
                        }
                        Cmd::Redraw => window.request_redraw(),
                        Cmd::Exit => control_flow.set_exit(),
                    }
                }
            }
        }
    });
}

use rodio::{Decoder, OutputStream, source::Source};

fn play_sound() -> OutputStream {
    use std::io::Cursor;

    let (stream, stream_handle) = OutputStream::try_default().unwrap();

    let bytes = include_bytes!("..\\assets\\C418_Minecraft_Sweden_1.mp3");
    // Decode that sound file into a source
    let source = Decoder::new(Cursor::new(bytes)).unwrap();
    // Play the sound directly on the device
    stream_handle.play_raw(source.convert_samples()).unwrap();

    stream
}