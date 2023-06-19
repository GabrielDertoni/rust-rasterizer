#![feature(portable_simd, slice_as_chunks)]

use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::{LogicalSize, PhysicalPosition},
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoopBuilder,
    window::{CursorGrabMode, WindowBuilder},
};
use notify::{event::{Event as NotifyEvent, EventKind as NotifyEventKind}, Watcher};

use std::time::{Duration, Instant};
use std::ops::Range;
use std::path::PathBuf;

use rasterization::{
    texture::{BorrowedMutTexture, BorrowedTexture}, clear_color,
    obj::{Material, Obj},
    shaders,
    utils::{FpvCamera, FpsCounter},
    vec::Vec3,
    texture::OwnedTexture,
    vert_buf::VertBuf, VertexBuf,
    config::{Light, Scene, RenderingConfig, RasterizerConfig, RasterizerImplementation},
    pipeline::PipelineMode,
    FragmentShaderSimd,
};

struct Model {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
    textured_subsets: Vec<TexturedSubset>,
    face_range: Range<usize>,
}

struct TexturedSubset {
    material_idx: usize,
    range: Range<usize>,
}

struct LightInfo {
    light: Light,
    shadow_map: OwnedTexture<f32>,
}

pub struct World {
    vert_buf: Vec<rasterization::Vertex>,
    index_buf: Vec<[usize; 3]>,
    materials: Vec<Material>,
    models: Vec<Model>,
    lights: Vec<LightInfo>,
    depth_buf: Vec<f32>,
    camera: FpvCamera,
    fps_counter: FpsCounter,
    axis: Vec3,
    enable_logging: bool,
    enable_shadows: bool,
    rendering_cfg: RenderingConfig,
    rasterizer_cfg: RasterizerConfig,
}

impl World {
    pub fn new(scene: Scene, root_path: PathBuf) -> anyhow::Result<Self> {
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
            let mut textured_subsets: Vec<TexturedSubset> = obj.use_material.into_iter()
                .map(|usemtl| {
                    let Some(material_idx) = obj.materials.iter()
                        .position(|mtl| mtl.name == usemtl.name) else {
                            let names_found = obj.materials.iter().map(|mtl| mtl.name.as_str()).collect::<Vec<_>>();
                            panic!("could not find material {} in .obj, found {:?}", usemtl.name, names_found);
                        };
                    TexturedSubset {
                        material_idx,
                        range: usemtl.range.start + offset .. usemtl.range.end + offset,
                    }
                })
                .collect();

            if textured_subsets.len() == 0 {
                textured_subsets.push(TexturedSubset {
                    material_idx: 0,
                    range: offset..offset + n_faces,
                });
            }

            let off = vert_buf.len();
            vert_buf.extend(vert);
            // Need to shift over all indicies to point to where those vertices ended up in the combined vertex buffer.
            index_buf.extend(idx.into_iter().map(|tri| tri.map(|i| i + off)));
            models.push(Model {
                position: model.position,
                rotation: model.rotation,
                scale: model.scale,
                textured_subsets,
                face_range: offset..offset + n_faces,
            });
            materials.extend(
                obj.materials.into_iter()
                    .map(|mut mat| {
                        let (orig_width, orig_height) = (mat.map_kd.width() as usize, mat.map_kd.height() as usize);
                        let (mut width, mut height) = (orig_width as usize, orig_height as usize);
                        if !width.is_power_of_two() {
                            width = width.next_power_of_two();
                        }
                        if !height.is_power_of_two() {
                            height = height.next_power_of_two();
                        }
                        if width != orig_width || height != orig_height {
                            let mut buf = vec![0u8; width * height * 4];
                            let slice = &mat.map_kd as &[u8];
                            for row in 0..orig_height {
                                let start_row = row * width * 4;
                                let start_row_orig = row * orig_width * 4;
                                buf[start_row..start_row + orig_width * 4].copy_from_slice(&slice[start_row_orig..start_row_orig + orig_width * 4]);
                            }
                            mat.map_kd = image::ImageBuffer::<image::Rgba<u8>, _>::from_vec(width as u32, height as u32, buf).unwrap();
                        }
                        mat
                    })
            );
        }

        let lights = scene.lights.into_iter().map(|light| {
            let size = light.shadow_map_size.expect("TODO: have default value for shadow map config");
            LightInfo {
                shadow_map: OwnedTexture::from_vec(size, size, vec![0.0; size * size]),
                light,
            }
        })
        .collect();

        let enable_shadows = match scene.config.get("enable-shadows") {
            Some(&toml::Value::Boolean(value)) => value,
            Some(_) => return Err(anyhow::anyhow!("expected value of `config.enable-shadows` to be a boolean")),
            None => false,
        };

        let width = scene.rendering.width;
        let height = scene.rendering.height;
        let aspect_ratio = width as f32 / height as f32;

        let vert_buf = (0..vert_buf.len()).map(|i| vert_buf.index(i)).collect();

        Ok(World {
            vert_buf,
            index_buf,
            materials,
            models,
            lights,
            depth_buf: vec![0.0; (width * height) as usize],
            camera: scene.camera.into_fpv(aspect_ratio),
            fps_counter: FpsCounter::new(),
            axis: Vec3::zero(),
            enable_logging: true,
            enable_shadows,
            rendering_cfg: scene.rendering,
            rasterizer_cfg: scene.rasterizer,
        })
    }

    pub fn render(&mut self, buf: &mut [u8], dt: Duration) {
        let (pixels, _) = buf.as_chunks_mut::<4>();
        let mut pixels = BorrowedMutTexture::from_mut_slice(
            self.rendering_cfg.width as usize,
            self.rendering_cfg.height as usize,
            pixels,
        );

        self.camera.move_delta(self.axis * dt.as_secs_f32());

        let fog_color = 0x61b7e8ff;
        clear_color(pixels.borrow_mut(), fog_color);

        {
            let start = Instant::now();
            if self.enable_shadows {
                todo!();
                // self.render_with_shadows(pixels);
            } else {
                self.render_without_shadows(pixels);
            }
            self.fps_counter.record_measurement(start.elapsed());
        }

        if self.enable_logging {
            println!("render time: {:?}", self.fps_counter.mean_time());
            println!("FPS: {}", self.fps_counter.mean_fps());
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

        let camera_transform = self.camera.transform_matrix(self.rendering_cfg.near, self.rendering_cfg.far);

        let vert_shader = shaders::textured::TexturedVertexShader::new(camera_transform);
        let mut pipeline = Pipeline::new(&self.vert_buf, &self.index_buf, PipelineMode::Simd3D);

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
        let mut pipeline = pipeline.process_vertices(&vert_shader);

        let start = std::time::Instant::now();
        for model in &self.models {
            for subset in &model.textured_subsets {
                let texture = &self.materials[subset.material_idx].map_kd;
                let (texture_pixels, _) = texture.as_chunks::<4>();
                let texture = BorrowedTexture::from_slice(
                    texture.width() as usize,
                    texture.height() as usize,
                    texture_pixels,
                );
                let frag_shader = shaders::textured::TexturedFragmentShader::new(texture);
    
                pipeline.draw_threaded(subset.range.clone(), FragmentShaderSimd::<_, 8>::simd_impl(&frag_shader));
            }
        }
        println!("time to render models: {:?}", start.elapsed());
        pipeline.finalize();
        let metrics = pipeline.get_metrics();
        println!("{metrics}");
    }

    pub fn update_cursor(&mut self, dx: f32, dy: f32) {
        self.camera.rotate_delta(dy, dx);
    }
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let Some(scene_path) = args.get(0).map(PathBuf::from) else {
        return Err(anyhow::anyhow!("expected a command line argument indicating the scene to load, but got none"));
    };
    let scene = Scene::load_toml(&scene_path)?;
    let event_loop = EventLoopBuilder::<NotifyEvent>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_title("rasterizer")
        .with_inner_size(LogicalSize::new(scene.rendering.width as u32, scene.rendering.height as u32))
        .build(&event_loop)
        .unwrap();

    let wsize = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(wsize.width, wsize.height, &window);
        Pixels::new(scene.rendering.width as u32, scene.rendering.height as u32, surface_texture).unwrap()
    };

    let mut world = World::new(scene, scene_path.parent().unwrap().to_owned())?;

    let event_loop_sender = event_loop.create_proxy();
    let mut watcher = notify::recommended_watcher(move |res| {
        match res {
            Ok(event) => _ = event_loop_sender.send_event(event),
            Err(e) => eprintln!("watch error: {e}"),
        }
    })?;
    watcher.watch(scene_path.as_ref(), notify::RecursiveMode::NonRecursive)?;

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
                world = match World::new(scene, scene_path.parent().unwrap().to_owned()) {
                    Ok(world) => world,
                    Err(e) => {
                        eprintln!("[ERROR] while trying to create world and hot reload scene: {e}");
                        return;
                    }
                };
            }
            _ => (),
        }
    })
}
