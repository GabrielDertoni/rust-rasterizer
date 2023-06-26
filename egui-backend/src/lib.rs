#![feature(portable_simd, slice_as_chunks, slice_flatten)]

mod painter;

use egui_winit::winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    window::Window,
};
use egui_winit::egui;

use painter::Painter;
use rasterization::texture::BorrowedMutTexture;

pub trait App: 'static {
    fn update(&mut self, ctx: &egui::Context);
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub window_size: Option<(u32, u32)>,
}

#[cfg(feature = "event-loop")]
pub fn run_default<A: App>(screen_name: &str, app: A) -> ! {
    run(screen_name, Default::default(), app)
}

#[cfg(feature = "event-loop")]
pub fn run<A: App>(screen_name: &str, options: Options, mut app: A) -> ! {
    use pixels::{Pixels, SurfaceTexture};
    use egui_winit::winit::{dpi::LogicalSize, event_loop::EventLoop, window::WindowBuilder};

    let (width, height) = options.window_size.unwrap_or((600, 400));

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(screen_name)
        .with_inner_size(LogicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let wsize = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(wsize.width, wsize.height, &window);
        Pixels::new(wsize.width, wsize.height, surface_texture).unwrap()
    };
    let mut ctx = Context::new(egui::Context::default(), &window);
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_wait();

        match event {
            Event::RedrawRequested(_) => {
                let (color_buf, _) = pixels.frame_mut().as_chunks_mut::<4>();
                let color_buf = BorrowedMutTexture::from_mut_slice(
                    wsize.width as usize,
                    wsize.height as usize,
                    color_buf,
                );

                ctx.redraw(&mut app, color_buf, &window);
                pixels.render().unwrap()
            }
            event => match ctx.handle_event(event).cmd {
                Some(Cmd::Resize(wsize)) => {
                    pixels.resize_buffer(wsize.width, wsize.height).unwrap();
                    pixels.resize_surface(wsize.width, wsize.height).unwrap();
                }
                Some(Cmd::Redraw) => window.request_redraw(),
                Some(Cmd::Exit) => control_flow.set_exit(),
                None => (),
            },
        }
    })
}

pub struct Response {
    pub consumed: bool,
    pub cmd: Option<Cmd>,
}

impl Response {
    pub fn unconsumed_cmd(cmd: Cmd) -> Self {
        Response { consumed: false, cmd: Some(cmd) }
    }

    pub fn consumed_cmd(cmd: Cmd) -> Self {
        Response { consumed: true, cmd: Some(cmd) }
    }

    pub fn consumed() -> Self {
        Response { consumed: true, cmd: None }
    }

    pub fn empty() -> Self {
        Response { consumed: false, cmd: None }
    }
}

pub enum Cmd {
    Resize(PhysicalSize<u32>),
    Redraw,
    Exit,
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

    pub fn handle_event<T>(&mut self, event: Event<T>) -> Response {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => Response {
                consumed: true,
                cmd: Some(Cmd::Exit),
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(newsize),
                ..
            } => {
                Response {
                    consumed: false,
                    cmd: Some(Cmd::Resize(newsize)),
                }
            }
            Event::WindowEvent { event, .. } => {
                let response = self.io_state.on_event(&self.egui_ctx, &event);
                if response.repaint {
                    Response {
                        consumed: response.consumed,
                        cmd: Some(Cmd::Redraw),
                    }
                } else {
                    Response {
                        consumed: response.consumed,
                        cmd: None,
                    }
                }
            }
            _ => Response {
                consumed: false,
                cmd: None,
            },
        }
    }

    pub fn redraw<'a, A: App>(
        &mut self,
        app: &mut A,
        pixels: BorrowedMutTexture<'a, [u8; 4]>,
        window: &Window,
    ) {
        let input = self.io_state.take_egui_input(&window);
        let output = self.egui_ctx.run(input, |ctx| app.update(ctx));
        self.io_state
            .handle_platform_output(&window, &self.egui_ctx, output.platform_output);
        let clipped_primitives = self.egui_ctx.tessellate(output.shapes);

        self.painter.paint(
            output.textures_delta,
            clipped_primitives,
            &self.egui_ctx,
            pixels,
        );
    }
}
