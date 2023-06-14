#![feature(portable_simd, slice_as_chunks, slice_flatten)]

mod painter;

use pixels::{Pixels, SurfaceTexture};
use egui_winit::winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use rasterization::texture::BorrowedMutTexture;
use painter::Painter;

pub trait App: 'static {
    fn update(&mut self, ctx: &egui::Context);
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub window_size: Option<(u32, u32)>,
}

pub fn run_default<A: App>(screen_name: &str, app: A) -> ! {
    run(screen_name, Default::default(), app)
}

pub fn run<A: App>(screen_name: &str, options: Options, mut app: A) -> ! {
    let (width, height) = options.window_size.unwrap_or((600, 400));
    
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(screen_name)
        .with_inner_size(LogicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut wsize = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(wsize.width, wsize.height, &window);
        Pixels::new(wsize.width, wsize.height, surface_texture).unwrap()
    };
    let mut ctx = Context::new(egui::Context::default(), &window);
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_wait();

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => control_flow.set_exit(),
            Event::WindowEvent {
                event: WindowEvent::Resized(newsize),
                ..
            } => {
                wsize = newsize;
                pixels.resize_buffer(wsize.width, wsize.height).unwrap();
                pixels.resize_surface(wsize.width, wsize.height).unwrap();
            }
            Event::WindowEvent { event, .. } => {
                let response = ctx.io_state.on_event(&ctx.egui_ctx, &event);
                if response.repaint {
                    window.request_redraw();
                }
            }
            Event::RedrawRequested(_) => {
                {
                    let (pixels, _) = pixels.frame_mut().as_chunks_mut::<4>();
                    let pixels = BorrowedMutTexture::from_mut_slice(
                        wsize.width as usize,
                        wsize.height as usize,
                        pixels,
                    );
    
                    let input = ctx.io_state.take_egui_input(&window);
                    let output = ctx.egui_ctx.run(input, |ctx| app.update(ctx));
                    ctx.io_state.handle_platform_output(&window, &ctx.egui_ctx, output.platform_output);
                    let clipped_primitives = ctx.egui_ctx.tessellate(output.shapes);
    
                    ctx.painter.paint(output.textures_delta, clipped_primitives, &ctx.egui_ctx, pixels);
                }
                pixels.render().unwrap()
            }
            _ => (),
        }
    })
}

pub struct Context {
    painter: Painter,
    egui_ctx: egui::Context,
    io_state: egui_winit::State,
}

impl Context {
    pub fn new(egui_ctx: egui::Context, window: &Window) -> Self {
        Context {
            painter: Painter::new(),
            egui_ctx,
            io_state: egui_winit::State::new(&window),
        }
    }
}