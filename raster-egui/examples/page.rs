fn main() {
    let options = raster_egui::Options {
        window_size: Some((600, 400)),
        ..Default::default()
    };
    raster_egui::run(
        "Page",
        options,
        App::default(),
    )
}

struct App {
}

impl Default for App {
    fn default() -> Self {
        Self {
        }
    }
}

/*
pub struct ATag {
    href: Option<String>,
    content: String,
}

pub enum ParagraphPart {
    Text(String),
    ATag(ATag)
}

#[derive(Default, Clone)]
pub struct Paragraph {
    parts: Vec<ParagraphPart>,
}

impl Paragraph {
    fn text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(ParagraphPart::String(text.into()));
        self
    }

    fn link(mut self, link: impl Into<String>) -> Self {
        self.parts.push(ParagraphPart::ATag(ATag {
            href: None,
            content: link.into(),
        }));
        self
    }
}

impl egui::Widget for Paragraph {
    fn ui(self, ui: &mut Ui) -> egui::Response {
        ui.horizontal_wrapped(|ui| {
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing = egui::Vec2::splat(0.);
                for part in self.parts {
                    match part {
                        ParagraphPart::Text(text) => ui.label(text),
                        ParagraphPart::ATag(tag) => {
                            if let Some(href) = tag.href {
                                ui.hyperlink_to(tag.content, href);
                            } else {
                                _ = ui.link(tag.content);
                            }
                        }
                    }
                }
            });
        }).response
    }
}
*/

pub struct Ol<'a> {
    num: i32,
    ui: &'a mut egui::Ui,
}

impl<'a> Ol<'a> {
    pub fn list_item<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> &mut Self {
        self.ui.horizontal_wrapped(|ui| {
            ui.label(format!("{}.", self.num));
            self.num += 1;
            add_contents(ui);
        });
        self
    }
}

pub trait UiExt {
    fn paragraph<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R>;
    fn ol(&mut self) -> Ol;
    fn ul_list_item<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R>;
}

impl UiExt for egui::Ui {
    fn paragraph<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
        self.horizontal_wrapped(|ui| {
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing = egui::Vec2::splat(0.);
                add_contents(ui)
            })
            .inner
        })
    }

    fn ol(&mut self) -> Ol {
        Ol { num: 1, ui: self }
    }

    fn ul_list_item<R>(&mut self, add_contents: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
        self.horizontal_wrapped(|ui| {
            let (response, painter) = ui.allocate_painter(egui::Vec2::splat(5.), egui::Sense::hover());
            painter.circle_filled(response.rect.center(), response.rect.width() / 2., egui::Color32::from_gray(128));
            add_contents(ui)
        })
    }
}

impl raster_egui::App for App {
    fn update(&mut self, ctx: &egui::Context) {
        const ORANGE: egui::Color32 = egui::Color32::from_rgb(0xff, 0xd9, 0x5D);

        ctx.set_pixels_per_point(1.2);
        let mut style = ctx.style();
        let style_mut = &mut std::sync::Arc::make_mut(&mut style);
        // std::sync::Arc::make_mut(&mut style).visuals = egui::Visuals::light();
        style_mut.spacing.item_spacing.y = 10.;
        // style_mut.visuals.panel_fill = egui::Color32::from_rgb(0x75, 0x75, 0x75);
        style_mut.visuals.panel_fill = egui::Color32::from_rgb(0x35, 0x35, 0x35);
        style_mut.visuals.override_text_color = Some(egui::Color32::from_rgb(0xfd, 0xfd, 0xfd));
        ctx.set_style(style);
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.paragraph(|ui| {
                ui.heading(egui::RichText::new("Crate ").strong());
                ui.heading(egui::RichText::new("xml_dom").color(ORANGE).strong())
            });

            ui.paragraph(|ui| {
                ui.label("This crate provides a trait-based implementation of the DOM with minimal changes to the style and semantics defined in the Level 2 specification. The specific mapping from the IDL in the specification is described ");
                _ = ui.link("below");
                ui.label(", however from a purely style point of view the implementation has the following characteristics:");
            });

            ui.ol()
                .list_item(|ui| {
                    ui.label("It maintains a reasonable separation between the node type traits and the tree implementation using opaque");
                    ui.code("NodeRef");
                    ui.label("reference types.");
                })
                .list_item(|ui| {
                    ui.label("Where possible the names from IDL are used with minimal conversion; see mapping section below");
                })
                .list_item(|ui| {
                    ui.label("All IDL attributes become trait functions; see mapping section below.");
                });

            ui.paragraph(|ui| {
                ui.label("This leads to a replication of the typical programmer experience where casting between the node traits is required. This is supported by the ");
                ui.code("xml_dom::level2::convert");
                ui.label("module.");
            });

            ui.heading(
                egui::RichText::new("Features")
                    .color(ORANGE)
                    .strong()
            );
        });
    }
}