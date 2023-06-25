use std::collections::HashMap;

use egui_winit::egui::{self, epaint};
use egui::{ClippedPrimitive, TextureId, TexturesDelta};
use epaint::{image::ImageData, Mesh, Primitive};

use rasterization::{
    pipeline::PipelineMode,
    texture::{BorrowedMutTexture, OwnedTexture},
    FragmentShader,
};

fn color_image_to_texture(image: epaint::image::ColorImage) -> OwnedTexture<[u8; 4]> {
    let tex = OwnedTexture::from_vec(image.width(), image.height(), image.pixels);
    // SAFETY: `Color32` is just a 32 bit color value with RGBA bytes, see https://docs.rs/epaint/0.22.0/epaint/struct.Color32.html.
    unsafe { tex.cast::<[u8; 4]>() }
}

fn font_image_to_texture(image: epaint::image::FontImage) -> OwnedTexture<[u8; 4]> {
    let vec = image
        .srgba_pixels(None)
        .map(|val| {
            let [r, g, b, a] = egui::Rgba::from(val).to_rgba_unmultiplied();
            [
                (r * 255.) as u8,
                (g * 255.) as u8,
                (b * 255.) as u8,
                (a * 255.) as u8,
            ]
        })
        .collect();
    OwnedTexture::from_vec(image.width(), image.height(), vec)
}

pub struct Painter {
    textures: HashMap<TextureId, OwnedTexture<[u8; 4]>>,
}

impl Painter {
    pub fn new() -> Self {
        Painter {
            textures: HashMap::default(),
        }
    }

    pub fn paint<'a>(
        &mut self,
        deltas: TexturesDelta,
        primitives: Vec<ClippedPrimitive>,
        egui_ctx: &egui::Context,
        mut color_buf: BorrowedMutTexture<'a, [u8; 4]>,
    ) {
        self.update_textures(deltas);
        // let start = std::time::Instant::now();
        for prim in primitives {
            if let Primitive::Mesh(mesh) = prim.primitive {
                self.paint_mesh(mesh, egui_ctx, color_buf.borrow_mut());
            } else {
                panic!("custom rendering callbacks are not implemented for this backend");
            }
        }
        // eprintln!("# textures: {}", self.textures.len());
        // eprintln!("draw time: {:?}", start.elapsed());
    }

    fn paint_mesh<'a>(
        &self,
        mesh: Mesh,
        egui_ctx: &egui::Context,
        color_buf: BorrowedMutTexture<'a, [u8; 4]>,
    ) {
        use rasterization::{config::CullingMode, pipeline::Pipeline};

        // eprintln!("painting mesh with {} triangles", mesh.indices.len() / 3);

        let tex = &self.textures[&mesh.texture_id];

        let indices: Vec<usize> = mesh.indices.into_iter().map(|i| i as usize).collect();
        let (indices, tail) = indices.as_chunks::<3>();
        assert_eq!(tail.len(), 0);
        let mut pipeline = Pipeline::new(
            mesh.vertices.as_slice(),
            indices,
            color_buf.size(),
            PipelineMode::Basic2D,
        );
        pipeline.set_culling_mode(CullingMode::Disabled);
        let width = color_buf.width();
        let height = color_buf.height();
        pipeline.set_color_buffer(color_buf);
        let mut pipeline = pipeline.process_vertices(
            &shaders::Vert {
                width: width as f32 / egui_ctx.pixels_per_point(),
                height: height as f32 / egui_ctx.pixels_per_point(),
            },
            None,
        );
        pipeline.draw(
            0..indices.len(),
            FragmentShader::into_impl(&shaders::ColoredFrag {
                texture: tex.borrow(),
            }),
        );
        pipeline.finalize();

        if cfg!(feature = "performance-counters") {
            let metrics = pipeline.get_metrics();
            println!("{metrics}");
        }
    }

    fn update_textures(&mut self, deltas: TexturesDelta) {
        for (id, delta) in deltas.set {
            if self.textures.contains_key(&id) {
                let update = match delta.image {
                    ImageData::Color(image) => color_image_to_texture(image),
                    ImageData::Font(image) => font_image_to_texture(image),
                };
                let curr = self.textures.get_mut(&id).unwrap();
                if let Some(pos) = delta.pos {
                    curr.slice_mut(
                        pos[0]..pos[0] + update.width(),
                        pos[1]..pos[1] + update.height(),
                    )
                    .copy_from(&update);
                } else {
                    *curr = update;
                }
            } else {
                assert!(
                    delta.pos.is_none(),
                    "expected `pos` to be none since texture didn't previously exist"
                );
                // self.textures.insert(id, delta.image.into());
                match delta.image {
                    ImageData::Color(image) => {
                        self.textures.insert(id, color_image_to_texture(image));
                    }
                    ImageData::Font(image) => {
                        self.textures.insert(id, font_image_to_texture(image));
                    }
                }
            }
        }

        for id in deltas.free {
            self.textures.remove(&id);
        }
    }
}

mod shaders {
    use std::simd::{LaneCount, Mask, Simd, SupportedLaneCount};
    use egui_winit::egui::epaint;

    use rasterization::{
        texture::{BorrowedTexture, TextureWrap},
        math::{Vec, Vec2, Vec2i, Vec4, Vec4xN},
        Attributes, AttributesSimd, IntoSimd, StructureOfArray,
    };

    pub struct Vert {
        pub width: f32,
        pub height: f32,
    }

    #[derive(Clone, Copy, IntoSimd, Attributes, AttributesSimd)]
    pub struct Attr {
        #[position]
        pub pos: Vec4,
        pub color: Vec4,
        pub uv: Vec2,
    }

    impl rasterization::VertexShader<epaint::Vertex> for Vert {
        type Output = Attr;

        fn exec(&self, vertex: epaint::Vertex) -> Attr {
            Attr {
                // Convert coordinates to NDC
                pos: Vec4::from([
                    2. * vertex.pos.x / self.width - 1.,
                    1. - 2. * vertex.pos.y / self.height,
                    0.,
                    1.,
                ]),
                color: Vec::from(epaint::Rgba::from(vertex.color).to_rgba_unmultiplied()),
                uv: Vec2::from([vertex.uv.x, 1. - vertex.uv.y]),
            }
        }
    }

    pub struct ColoredFrag<'a> {
        pub texture: BorrowedTexture<'a, [u8; 4]>,
    }

    impl<'a> rasterization::FragmentShader<Attr> for ColoredFrag<'a> {
        fn exec(&self, _pixel_coords: Vec2i, attrs: Attr) -> Vec4 {
            let tex = self.texture.index_uv(attrs.uv, TextureWrap::Clamp);
            let color = attrs.color;
            Vec4::from([color.x, color.y, color.z, color.w * tex.x])
        }
    }

    impl<'a, const LANES: usize> rasterization::FragmentShaderSimd<Attr, LANES> for ColoredFrag<'a>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        fn exec(
            &self,
            mask: Mask<i32, LANES>,
            _pixel_coords: Vec<Simd<i32, LANES>, 2>,
            attrs: AttrSimd<LANES>,
        ) -> Vec4xN<LANES> {
            let tex = self
                .texture
                .simd_index_uv(attrs.uv, mask, TextureWrap::Clamp);
            let color = attrs.color;
            Vec::from([color.x, color.y, color.z, color.w * tex.x])
        }
    }

    pub struct FontFrag<'a> {
        pub texture: BorrowedTexture<'a, f32>,
    }

    impl<'a> rasterization::FragmentShader<Attr> for FontFrag<'a> {
        fn exec(&self, _pixel_coords: Vec2i, attrs: Attr) -> Vec4 {
            let ix = self.texture.get_index_from_uv(attrs.uv, TextureWrap::Clamp);
            let color = attrs.color.xyz() * self.texture[(ix.x, ix.y)];
            Vec4::from([color.x, color.y, color.z, attrs.color.w])
        }
    }
}
