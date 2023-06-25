fn main() {
    let options = raster_egui::Options {
        window_size: Some((600, 400)),
        ..Default::default()
    };
    raster_egui::run(
        "🖮 Code Editor",
        options,
        CodeEditor::default(),
    )
}

pub trait View {
    fn ui(&mut self, ui: &mut egui::Ui);
}

pub struct CodeEditor {
    language: String,
    code: String,
}

impl Default for CodeEditor {
    fn default() -> Self {
        Self {
            language: "rs".into(),
            code: "// A very simple example\n\
fn main() {\n\
\tprintln!(\"Hello world!\");\n\
}\n\
"
            .into(),
        }
    }
}

impl raster_egui::App for CodeEditor {
    fn update(&mut self, ctx: &egui::Context) {
        use View as _;
        ctx.set_pixels_per_point(1.5);
        egui::CentralPanel::default().show(ctx, |ui| self.ui(ui));
    }
}

impl View for CodeEditor {
    fn ui(&mut self, ui: &mut egui::Ui) {
        let Self { language, code } = self;

        ui.horizontal(|ui| {
            ui.set_height(0.0);
            ui.label("An example of syntax highlighting in a TextEdit.");
        });

        let mut theme = syntax_highlighting::CodeTheme::from_memory(ui.ctx());
        ui.collapsing("Theme", |ui| {
            ui.group(|ui| {
                theme.ui(ui);
                theme.clone().store_in_memory(ui.ctx());
            });
        });

        let mut layouter = |ui: &egui::Ui, string: &str, wrap_width: f32| {
            let mut layout_job =
                syntax_highlighting::highlight(ui.ctx(), &theme, string, language);
            layout_job.wrap.max_width = wrap_width;
            ui.fonts(|f| f.layout_job(layout_job))
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_sized(
                ui.available_size(),
                egui::TextEdit::multiline(code)
                    .font(egui::TextStyle::Monospace) // for cursor height
                    .code_editor()
                    .desired_rows(10)
                    .lock_focus(true)
                    .desired_width(f32::INFINITY)
                    .layouter(&mut layouter),
            );
        });
    }
}

mod syntax_highlighting {
    use egui::text::LayoutJob;

    /// Memoized Code highlighting
    pub fn highlight(ctx: &egui::Context, theme: &CodeTheme, code: &str, language: &str) -> LayoutJob {
        impl egui::util::cache::ComputerMut<(&CodeTheme, &str, &str), LayoutJob> for Highlighter {
            fn compute(&mut self, (theme, code, lang): (&CodeTheme, &str, &str)) -> LayoutJob {
                self.highlight(theme, code, lang)
            }
        }

        type HighlightCache = egui::util::cache::FrameCache<LayoutJob, Highlighter>;

        ctx.memory_mut(|mem| {
            mem.caches
                .cache::<HighlightCache>()
                .get((theme, code, language))
        })
    }

    // ----------------------------------------------------------------------------

    #[derive(Clone, Copy, PartialEq)]
    enum TokenType {
        Comment = 0,
        Keyword = 1,
        Literal = 2,
        StringLiteral = 3,
        Punctuation = 4,
        Whitespace = 5,
    }

    #[derive(Clone, Hash, PartialEq)]
    pub struct CodeTheme {
        dark_mode: bool,

        formats: [egui::TextFormat; 6],
    }

    impl Default for CodeTheme {
        fn default() -> Self {
            Self::dark()
        }
    }

    impl CodeTheme {
        pub fn from_memory(ctx: &egui::Context) -> Self {
            if ctx.style().visuals.dark_mode {
                ctx.data_mut(|d| {
                    d.get_persisted(egui::Id::new("dark"))
                        .unwrap_or_else(CodeTheme::dark)
                })
            } else {
                ctx.data_mut(|d| {
                    d.get_persisted(egui::Id::new("light"))
                        .unwrap_or_else(CodeTheme::light)
                })
            }
        }

        pub fn store_in_memory(self, ctx: &egui::Context) {
            if self.dark_mode {
                ctx.data_mut(|d| d.insert_persisted(egui::Id::new("dark"), self));
            } else {
                ctx.data_mut(|d| d.insert_persisted(egui::Id::new("light"), self));
            }
        }
    }

    impl CodeTheme {
        pub fn dark() -> Self {
            let font_id = egui::FontId::monospace(10.0);
            use egui::{Color32, TextFormat};
            Self {
                dark_mode: true,
                formats: [
                    TextFormat::simple(font_id.clone(), Color32::from_gray(120)),
                    TextFormat::simple(font_id.clone(), Color32::from_rgb(255, 100, 100)),
                    TextFormat::simple(font_id.clone(), Color32::from_rgb(87, 165, 171)),
                    TextFormat::simple(font_id.clone(), Color32::from_rgb(109, 147, 226)),
                    TextFormat::simple(font_id.clone(), Color32::LIGHT_GRAY),
                    TextFormat::simple(font_id.clone(), Color32::TRANSPARENT),
                ],
            }
        }

        pub fn light() -> Self {
            let font_id = egui::FontId::monospace(10.0);
            use egui::{Color32, TextFormat};
            Self {
                dark_mode: false,
                formats: [
                    TextFormat::simple(font_id.clone(), Color32::GRAY),
                    TextFormat::simple(font_id.clone(), Color32::from_rgb(235, 0, 0)),
                    TextFormat::simple(font_id.clone(), Color32::from_rgb(153, 134, 255)),
                    TextFormat::simple(font_id.clone(), Color32::from_rgb(37, 203, 105)),
                    TextFormat::simple(font_id.clone(), Color32::DARK_GRAY),
                    TextFormat::simple(font_id.clone(), Color32::TRANSPARENT),
                ],
            }
        }

        pub fn ui(&mut self, ui: &mut egui::Ui) {
            ui.horizontal_top(|ui| {
                let selected_id = egui::Id::null();
                let mut selected_tt: TokenType =
                    ui.data_mut(|d| *d.get_persisted_mut_or(selected_id, TokenType::Comment));

                ui.vertical(|ui| {
                    ui.set_width(150.0);
                    egui::widgets::global_dark_light_mode_buttons(ui);

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    ui.scope(|ui| {
                        for (tt, tt_name) in [
                            (TokenType::Comment, "// comment"),
                            (TokenType::Keyword, "keyword"),
                            (TokenType::Literal, "literal"),
                            (TokenType::StringLiteral, "\"string literal\""),
                            (TokenType::Punctuation, "punctuation ;"),
                            // (TokenType::Whitespace, "whitespace"),
                        ] {
                            let format = &mut self.formats[tt as usize];
                            ui.style_mut().override_font_id = Some(format.font_id.clone());
                            ui.visuals_mut().override_text_color = Some(format.color);
                            ui.radio_value(&mut selected_tt, tt, tt_name);
                        }
                    });

                    let reset_value = if self.dark_mode {
                        CodeTheme::dark()
                    } else {
                        CodeTheme::light()
                    };

                    if ui
                        .add_enabled(*self != reset_value, egui::Button::new("Reset theme"))
                        .clicked()
                    {
                        *self = reset_value;
                    }
                });

                ui.add_space(16.0);

                ui.data_mut(|d| d.insert_persisted(selected_id, selected_tt));

                egui::Frame::group(ui.style())
                    .inner_margin(egui::Vec2::splat(2.0))
                    .show(ui, |ui| {
                        // ui.group(|ui| {
                        ui.style_mut().override_text_style = Some(egui::TextStyle::Small);
                        ui.spacing_mut().slider_width = 128.0; // Controls color picker size
                        egui::widgets::color_picker::color_picker_color32(
                            ui,
                            &mut self.formats[selected_tt as usize].color,
                            egui::color_picker::Alpha::Opaque,
                        );
                    });
            });
        }
    }

    // ----------------------------------------------------------------------------

    #[derive(Default)]
    struct Highlighter {}

    impl Highlighter {
        #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
        fn highlight(&self, theme: &CodeTheme, mut text: &str, _language: &str) -> LayoutJob {
            // Extremely simple syntax highlighter for when we compile without syntect

            let mut job = LayoutJob::default();

            while !text.is_empty() {
                if text.starts_with("//") {
                    let end = text.find('\n').unwrap_or(text.len());
                    job.append(&text[..end], 0.0, theme.formats[TokenType::Comment as usize].clone());
                    text = &text[end..];
                } else if text.starts_with('"') {
                    let end = text[1..]
                        .find('"')
                        .map(|i| i + 2)
                        .or_else(|| text.find('\n'))
                        .unwrap_or(text.len());
                    job.append(
                        &text[..end],
                        0.0,
                        theme.formats[TokenType::StringLiteral as usize].clone(),
                    );
                    text = &text[end..];
                } else if text.starts_with(|c: char| c.is_ascii_alphanumeric()) {
                    let end = text[1..]
                        .find(|c: char| !c.is_ascii_alphanumeric())
                        .map_or_else(|| text.len(), |i| i + 1);
                    let word = &text[..end];
                    let tt = if is_keyword(word) {
                        TokenType::Keyword
                    } else {
                        TokenType::Literal
                    };
                    job.append(word, 0.0, theme.formats[tt as usize].clone());
                    text = &text[end..];
                } else if text.starts_with(|c: char| c.is_ascii_whitespace()) {
                    let end = text[1..]
                        .find(|c: char| !c.is_ascii_whitespace())
                        .map_or_else(|| text.len(), |i| i + 1);
                    job.append(
                        &text[..end],
                        0.0,
                        theme.formats[TokenType::Whitespace as usize].clone(),
                    );
                    text = &text[end..];
                } else {
                    let mut it = text.char_indices();
                    it.next();
                    let end = it.next().map_or(text.len(), |(idx, _chr)| idx);
                    job.append(
                        &text[..end],
                        0.0,
                        theme.formats[TokenType::Punctuation as usize].clone(),
                    );
                    text = &text[end..];
                }
            }

            job
        }
    }

    fn is_keyword(word: &str) -> bool {
        matches!(
            word,
            "as" | "async"
                | "await"
                | "break"
                | "const"
                | "continue"
                | "crate"
                | "dyn"
                | "else"
                | "enum"
                | "extern"
                | "false"
                | "fn"
                | "for"
                | "if"
                | "impl"
                | "in"
                | "let"
                | "loop"
                | "match"
                | "mod"
                | "move"
                | "mut"
                | "pub"
                | "ref"
                | "return"
                | "self"
                | "Self"
                | "static"
                | "struct"
                | "super"
                | "trait"
                | "true"
                | "type"
                | "unsafe"
                | "use"
                | "where"
                | "while"
        )
    }
}