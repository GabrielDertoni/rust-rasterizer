#![feature(slice_as_chunks, iter_next_chunk, portable_simd, pointer_is_aligned)]

use std::time::Instant;

use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use rasterization::world::World;

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

    let wsize = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(wsize.width, wsize.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture).unwrap()
    };

    let mut world = World::new(WIDTH, HEIGHT, model);

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
                (ElementState::Released, VirtualKeyCode::V | VirtualKeyCode::C) => {
                    world.axis.z = 0.0
                }
                (ElementState::Pressed, VirtualKeyCode::W) => world.axis.y = 1.0,
                (ElementState::Pressed, VirtualKeyCode::S) => world.axis.y = -1.0,
                (ElementState::Released, VirtualKeyCode::W | VirtualKeyCode::S) => {
                    world.axis.y = 0.0
                }
                (ElementState::Pressed, VirtualKeyCode::A) => world.axis.x = 1.0,
                (ElementState::Pressed, VirtualKeyCode::D) => world.axis.x = -1.0,
                (ElementState::Released, VirtualKeyCode::A | VirtualKeyCode::D) => {
                    world.axis.x = 0.0
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                let dt = start.elapsed();
                start = Instant::now();
                world.render(pixels.frame_mut(), dt);
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
                let cx = position.x as f32 / wsize.width as f32;
                let cy = position.y as f32 / wsize.height as f32;
                world.update_cursor(cx, cy);
            }
            _ => (),
        }
    });
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
