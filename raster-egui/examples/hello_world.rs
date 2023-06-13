#![feature(slice_as_chunks)]

use pixels::{Pixels, SurfaceTexture};
use egui_winit::winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use raster_egui::Painter;
use rasterization::texture::BorrowedMutTexture;

const WIDTH: usize = 600;
const HEIGHT: usize = 400;

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("rasterizer")
        .with_inner_size(LogicalSize::new(WIDTH as u32, HEIGHT as u32))
        .build(&event_loop)
        .unwrap();

    let wsize = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(wsize.width, wsize.height, &window);
        Pixels::new(WIDTH as u32, HEIGHT as u32, surface_texture).unwrap()
    };

    let ctx = egui::Context::default();
    let mut painter = Painter::new();
    let mut name = String::from("");
    let mut age = 0;

    let mut state = egui_winit::State::new(&window);
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_wait();

        match event {
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        is_synthetic: false,
                        ..
                    },
                ..
            }
            | Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => control_flow.set_exit(),
            Event::WindowEvent { event, .. } => {
                let response = state.on_event(&ctx, &event);
                if response.repaint {
                    window.request_redraw();
                }
            }
            Event::MainEventsCleared => {
                let (pixels, _) = pixels.frame_mut().as_chunks_mut::<4>();
                let pixels = BorrowedMutTexture::from_mut_slice(
                    wsize.width as usize,
                    wsize.height as usize,
                    pixels,
                );

                let input = state.take_egui_input(&window);
                let output = ctx.run(input, |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        ctx.set_pixels_per_point(10.);
                        ui.heading(egui::RichText::new("My egui Application").size(40.));
                        ui.horizontal(|ui| {
                            let name_label = ui.label(egui::RichText::new("Your name: ").size(20.));
                            ui.text_edit_singleline(&mut name)
                                .labelled_by(name_label.id);
                        });
                        ui.add(egui::Slider::new(&mut age, 0..=120).text("age"));
                        if ui.button("Click each year").clicked() {
                            age += 1;
                        }
                        ui.label(format!("Hello '{}', age {}", name, age));
                    });
                });
                let clipped_primitives = ctx.tessellate(output.shapes);

                painter.paint(output.textures_delta, clipped_primitives, pixels);
                window.request_redraw();
            }
            Event::RedrawRequested(_) => pixels.render().unwrap(),
            _ => (),
        }
    })
}
