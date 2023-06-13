#![feature(portable_simd, slice_as_chunks, slice_flatten)]

use std::collections::HashMap;

use egui::{ClippedPrimitive, TextureId, TexturesDelta};
use epaint::{image::ImageData, Mesh, Primitive};

use rasterization::texture::{BorrowedMutTexture, OwnedTexture};

pub enum TextureKind {
    Color(OwnedTexture<[u8; 4]>),
    Font(OwnedTexture<f32>),
}

impl From<ImageData> for TextureKind {
    fn from(image: ImageData) -> Self {
        match image {
            ImageData::Color(im) => TextureKind::Color(color_image_to_texture(im)),
            ImageData::Font(im) => TextureKind::Font(font_image_to_texture(im)),
        }
    }
}

fn color_image_to_texture(image: epaint::image::ColorImage) -> OwnedTexture<[u8; 4]> {
    let tex = OwnedTexture::from_vec(image.width(), image.height(), image.pixels);
    // SAFETY: `Color32` is just a 32 bit color value with RGBA bytes, see https://docs.rs/epaint/0.22.0/epaint/struct.Color32.html.
    unsafe { tex.cast::<[u8; 4]>() }
}

fn font_image_to_texture(image: epaint::image::FontImage) -> OwnedTexture<f32> {
    OwnedTexture::from_vec(image.width(), image.height(), image.pixels)
}

pub struct Painter {
    textures: HashMap<TextureId, TextureKind>,
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
        mut color_buf: BorrowedMutTexture<'a, [u8; 4]>,
    ) {
        self.update_textures(deltas);
        let start = std::time::Instant::now();
        for prim in primitives {
            if let Primitive::Mesh(mesh) = prim.primitive {
                self.paint_mesh(mesh, color_buf.borrow_mut());
            } else {
                panic!("custom rendering callbacks are not implemented for this backend");
            }
        }
        eprintln!("# textures: {}", self.textures.len());
        eprintln!("draw time: {:?}", start.elapsed());
    }

    fn paint_mesh<'a>(&self, mesh: Mesh, color_buf: BorrowedMutTexture<'a, [u8; 4]>) {
        use rasterization::{pipeline::Pipeline, prim3d::CullingMode};

        eprintln!("painting mesh with {} triangles", mesh.indices.len() / 3);

        let tex = &self.textures[&mesh.texture_id];

        let indices: Vec<usize> = mesh.indices.into_iter().map(|i| i as usize).collect();
        let (indices, tail) = indices.as_chunks::<3>();
        assert_eq!(tail.len(), 0);
        let mut pipeline = Pipeline::new_basic(mesh.vertices.as_slice(), indices);
        pipeline.set_culling_mode(CullingMode::Disabled);
        let width = color_buf.width();
        let height = color_buf.height();
        pipeline.set_color_buffer(color_buf);
        let mut pipeline = pipeline.process_vertices(&shaders::Vert {
            width: width as f32,
            height: height as f32,
        });
        match tex {
            TextureKind::Color(tex) => {
                pipeline.draw(
                    0..indices.len(),
                    &shaders::ColoredFrag {
                        texture: tex.borrow(),
                    },
                );
            }
            TextureKind::Font(tex) => {
                pipeline.draw(
                    0..indices.len(),
                    &shaders::FontFrag {
                        texture: tex.borrow(),
                    },
                );
            }
        }
        pipeline.finalize();
    }

    fn update_textures(&mut self, deltas: TexturesDelta) {
        for (id, delta) in deltas.set {
            if self.textures.contains_key(&id) {
                match (self.textures.get_mut(&id).unwrap(), delta.image) {
                    (TextureKind::Color(curr), ImageData::Color(update)) => {
                        let update = color_image_to_texture(update);
                        if let Some(pos) = delta.pos {
                            curr.slice_mut(
                                pos[0]..pos[0] + update.width(),
                                pos[1]..pos[1] + update.height(),
                            )
                            .copy_from(&update);
                        } else {
                            *curr = update;
                        }
                    }
                    (TextureKind::Font(curr), ImageData::Font(update)) => {
                        let update = font_image_to_texture(update);
                        if let Some(pos) = delta.pos {
                            curr.slice_mut(
                                pos[0]..pos[0] + update.width(),
                                pos[1]..pos[1] + update.height(),
                            )
                            .copy_from(&update);
                        } else {
                            *curr = update;
                        }
                    }
                    (curr, update) => {
                        assert!(
                            delta.pos.is_none(),
                            "expected `pos` to be none since texture kinds are different"
                        );
                        *curr = update.into();
                    }
                }
            } else {
                assert!(
                    delta.pos.is_none(),
                    "expected `pos` to be none since texture didn't previously exist"
                );
                // self.textures.insert(id, delta.image.into());
                match delta.image {
                    ImageData::Color(image) => {
                        self.textures.insert(id, TextureKind::Color(color_image_to_texture(image)));
                    }
                    ImageData::Font(image) => {
                        let vec = image.srgba_pixels(Some(0.4))
                            .map(|val| {
                                let [r, g, b, a] = egui::Rgba::from(val).to_rgba_unmultiplied();
                                [
                                    (r * 255.) as u8,
                                    (g * 255.) as u8,
                                    (b * 255.) as u8,
                                    (a * 255.) as u8,
                                ]
                            }).collect();
                        let texture = OwnedTexture::from_vec(image.width(), image.height(), vec);
                        self.textures.insert(id, TextureKind::Color(texture));
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

    use rasterization::{
        texture::{BorrowedTexture, TextureWrap},
        vec::{Vec, Vec2, Vec4, Vec4xN},
        Attributes, IntoSimd, StructureOfArray,
    };

    pub struct Vert {
        pub width: f32,
        pub height: f32,
    }

    #[derive(Clone, Copy, Attributes, IntoSimd)]
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
        fn exec<const LANES: usize>(
            &self,
            _mask: Mask<i32, LANES>,
            _pixel_coords: Vec<Simd<i32, LANES>, 2>,
            _attrs: AttrSimd<LANES>,
        ) -> Vec4xN<LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            todo!();
        }

        fn exec_basic(&self, attrs: Attr) -> Vec4 {
            attrs
                .color
                .element_mul(self.texture.index_uv(attrs.uv, TextureWrap::Clamp))
        }
    }

    pub struct FontFrag<'a> {
        pub texture: BorrowedTexture<'a, f32>,
    }

    impl<'a> rasterization::FragmentShader<Attr> for FontFrag<'a> {
        fn exec<const LANES: usize>(
            &self,
            _mask: Mask<i32, LANES>,
            _pixel_coords: Vec<Simd<i32, LANES>, 2>,
            _attrs: AttrSimd<LANES>,
        ) -> Vec4xN<LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            todo!();
        }

        fn exec_basic(&self, attrs: Attr) -> Vec4 {
            let ix = self.texture.get_index_from_uv(attrs.uv, TextureWrap::Clamp);
            let color = attrs.color.xyz() * self.texture[(ix.x, ix.y)];
            Vec4::from([color.x, color.y, color.z, attrs.color.w])
        }
    }
}
