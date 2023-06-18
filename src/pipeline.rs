use std::ops::Range;
use std::simd::Simd;

use crate::{
    config::CullingMode, simd_config::{LANES, SimdColorGamma, SimdPixels}, texture::BorrowedMutTexture, vert_buf::VertBuf,
    Attributes, AttributesSimd, FragmentShader, FragmentShaderImpl, FragmentShaderSimd, VertexBuf,
    VertexShader, common::count_cycles,
};

macro_rules! static_assert_eq {
    ($lhs:expr, $rhs:expr) => {{
        const _ASSERTION: [(); ($lhs == $rhs) as usize] = [()];
    }};
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum PipelineMode {
    #[default]
    Basic3D,
    Simd3D,
    Basic2D,
    Simd2D,
}

impl PipelineMode {
    pub fn is_simd(&self) -> bool {
        match self {
            PipelineMode::Basic3D => false,
            PipelineMode::Simd3D => true,
            PipelineMode::Basic2D => false,
            PipelineMode::Simd2D => true,
        }
    }
}

pub struct Pipeline<'a, B = &'a VertBuf> {
    vert_buf: B,
    index_buf: &'a [[usize; 3]],
    color_buf: Option<BorrowedMutTexture<'a, [u8; 4]>>,
    depth_buf: Option<BorrowedMutTexture<'a, f32>>,
    mode: PipelineMode,
    metrics: Metrics,
    culling_mode: CullingMode,
    alpha_clip: Option<f32>
}

impl<'a, B> Pipeline<'a, B>
where
    B: VertexBuf,
{
    pub fn new(vert_buf: B, index_buf: &'a [[usize; 3]], mode: PipelineMode) -> Self {
        Pipeline {
            vert_buf,
            index_buf,
            color_buf: None,
            depth_buf: None,
            mode,
            metrics: Metrics::new(),
            culling_mode: CullingMode::BackFace,
            alpha_clip: None,
        }
    }

    pub fn with_alpha_clip(&mut self, alpha_clip: Option<f32>) -> &mut Self {
        self.alpha_clip = alpha_clip;
        self
    }

    pub fn with_mode(&mut self, mode: PipelineMode) -> &mut Self {
        self.mode = mode;
        self
    }

    pub fn with_culling(&mut self, culling_mode: CullingMode) -> &mut Self {
        self.culling_mode = culling_mode;
        self
    }

    pub fn set_color_buffer(&mut self, mut color_buf: BorrowedMutTexture<'a, [u8; 4]>) {
        if let Some(depth_buf) = self.depth_buf.as_ref() {
            assert_eq!(color_buf.width(), depth_buf.width());
            assert_eq!(color_buf.height(), depth_buf.height());
        }

        if self.mode.is_simd() {
            // Coalesce the pixel layout to be [rrrrggggbbbbaaaa] repeating
            assert_eq!(
                color_buf.width() % LANES,
                0,
                "width must be a multiple of LANES"
            );
            assert!(color_buf
                .as_ptr()
                .is_aligned_to(std::mem::align_of::<SimdColorGamma>()));
            assert!(color_buf
                .as_ptr()
                .is_aligned_to(std::mem::align_of::<SimdPixels>()));

            let simd_color_buf = unsafe {
                let ptr = color_buf.as_mut_ptr().cast::<[[u8; 4]; LANES]>();
                std::slice::from_raw_parts_mut(ptr, color_buf.len() >> LANES.ilog2())
            };
            for pixels in simd_color_buf {
                let ff_mask = Simd::splat(0xff);
                let values = Simd::from(pixels.map(u32::from_ne_bytes));
                let v = SimdColorGamma::from([
                    ((values >> Simd::splat(24)) & ff_mask).cast::<u8>(),
                    ((values >> Simd::splat(16)) & ff_mask).cast::<u8>(),
                    ((values >> Simd::splat(8)) & ff_mask).cast::<u8>(),
                    (values & ff_mask).cast::<u8>(),
                ]);
                assert_eq!(std::mem::size_of_val(pixels), std::mem::size_of_val(&v));
                unsafe {
                    let ptr = pixels as *mut _ as *mut SimdColorGamma;
                    ptr.write(v);
                }
            }
        }
        self.color_buf.replace(color_buf);

    }

    pub fn take_color_buffer(&mut self) -> Option<BorrowedMutTexture<'a, [u8; 4]>> {
        use crate::vec::Vec;

        if self.mode.is_simd() {
            let mut color_buf = self.color_buf.take()?;
            let simd_color_buf = unsafe {
                let ptr = color_buf.as_mut_ptr().cast::<[[u8; LANES]; 4]>();
                std::slice::from_raw_parts_mut(ptr, color_buf.len() >> LANES.ilog2())
            };
            for color in simd_color_buf {
                let values = Vec::from(*color).map(|el| Simd::from(el).cast::<u32>());
                let v = (values.x << Simd::splat(24))
                | (values.y << Simd::splat(16))
                | (values.z << Simd::splat(8))
                | values.w;

                assert_eq!(std::mem::size_of_val(color), std::mem::size_of_val(&v));
                unsafe {
                    let ptr = color as *mut _ as *mut SimdPixels;
                    ptr.write(v);
                }
            }
            Some(color_buf)
        } else {
            self.color_buf.take()
        }
    }

    pub fn set_depth_buffer(&mut self, depth_buf: BorrowedMutTexture<'a, f32>) {
        if let Some(color_buf) = self.color_buf.as_ref() {
            assert_eq!(color_buf.width(), depth_buf.width());
            assert_eq!(color_buf.height(), depth_buf.height());
        }

        if self.mode.is_simd() {
            assert!(depth_buf
                .as_ptr()
                .is_aligned_to(std::mem::align_of::<Simd<f32, LANES>>()));
            assert_eq!(
                depth_buf.width() % LANES,
                0,
                "width must be a multiple of LANES"
            );
        }
        self.depth_buf.replace(depth_buf);
    }

    pub fn take_depth_buffer(&mut self) -> Option<BorrowedMutTexture<'a, f32>> {
        self.depth_buf.take()
    }

    pub fn set_culling_mode(&mut self, culling_mode: CullingMode) {
        self.culling_mode = culling_mode;
    }

    pub fn process_vertices<'b: 'a, V: VertexShader<B::Vertex>>(
        &'b mut self,
        vert_shader: &V,
    ) -> ProcessedVertices<'b, V::Output, B> {
        use crate::common::process_vertex;

        let start = std::time::Instant::now();

        let attributes = count_cycles! {
            #[counter(self.metrics.performance_counters.vertex_processing)]

            (0..self.vert_buf.len())
                .map(|i| process_vertex(self.vert_buf.index(i), vert_shader, &mut self.metrics))
                .collect()
        };

        println!("process vertices: {:?}", start.elapsed());

        ProcessedVertices {
            pipeline: self,
            attributes,
        }
    }

    pub fn finalize(&mut self) {
        self.take_color_buffer();
        self.take_depth_buffer();
    }
}

pub struct ProcessedVertices<'a, Attr, B = &'a VertBuf> {
    pipeline: &'a mut Pipeline<'a, B>,
    attributes: Vec<Attr>,
}

impl<'a, Attr, B> ProcessedVertices<'a, Attr, B>
where
    Attr: Attributes + AttributesSimd<LANES> + Copy,
    B: VertexBuf,
{
    pub fn draw<Scalar, Simd>(
        &mut self,
        range: Range<usize>,
        frag_shader: FragmentShaderImpl<Scalar, Simd>,
    ) where
        Scalar: FragmentShader<Attr>,
        Simd: FragmentShaderSimd<Attr, LANES>,
    {
        let color_buf = self
            .pipeline
            .color_buf
            .as_mut()
            .expect("no color buffer")
            .borrow_mut();

        match self.pipeline.mode {
            PipelineMode::Basic3D => {
                use crate::prim3d::draw_triangles;
                let depth_buf = self
                    .pipeline
                    .depth_buf
                    .as_mut()
                    .map(|depth_buf| depth_buf.borrow_mut());

                draw_triangles(
                    &self.attributes,
                    &self.pipeline.index_buf[range],
                    frag_shader.unwrap_scalar(),
                    color_buf,
                    depth_buf,
                    self.pipeline.culling_mode,
                    self.pipeline.alpha_clip,
                    &mut self.pipeline.metrics,
                );
            }

            PipelineMode::Simd3D => {
                use crate::prim3d::simd::draw_triangles;
                let depth_buf = self
                    .pipeline
                    .depth_buf
                    .as_mut()
                    .expect("no depth buffer")
                    .borrow_mut();

                let color_buf = unsafe { color_buf.cast::<u32>() };
                draw_triangles(
                    &self.attributes,
                    &self.pipeline.index_buf[range],
                    frag_shader.unwrap_simd(),
                    color_buf,
                    depth_buf,
                    self.pipeline.culling_mode,
                    self.pipeline.alpha_clip,
                    &mut self.pipeline.metrics,
                );
            }

            PipelineMode::Basic2D => {
                use crate::prim2d::draw_triangles;

                draw_triangles(
                    &self.attributes,
                    &self.pipeline.index_buf[range],
                    frag_shader.unwrap_scalar(),
                    color_buf,
                    &mut self.pipeline.metrics,
                );
            }

            PipelineMode::Simd2D => {
                use crate::prim2d::simd::draw_triangles;

                let color_buf = unsafe { color_buf.cast::<u32>() };
                draw_triangles(
                    &self.attributes,
                    &self.pipeline.index_buf[range],
                    frag_shader.unwrap_simd(),
                    color_buf,
                    &mut self.pipeline.metrics,
                );
            }
        }
    }

    pub fn get_metrics(&self) -> Metrics {
        self.pipeline.metrics
    }

    pub fn finalize(&mut self) {
        self.pipeline.finalize();
    }
}


#[derive(Default, Debug, Clone, Copy)]
pub struct Metrics {
    pub triangles_drawn: usize,
    pub backfaces_culled: usize,
    pub behind_culled: usize,
    pub sum_areas: i64,
    #[cfg(feature = "performance-counters")]
    pub performance_counters: perf_counters::PerformanceCounters,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            triangles_drawn: 0,
            backfaces_culled: 0,
            behind_culled: 0,
            sum_areas: 0,
            #[cfg(feature = "performance-counters")]
            performance_counters: Default::default(),
        }
    }

    pub fn clear(&mut self) {
        self.triangles_drawn = 0;
        self.backfaces_culled = 0;
        self.behind_culled = 0;
        self.sum_areas = 0;
        #[cfg(feature = "performance-counters")]
        self.performance_counters.clear();
    }
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let &Metrics {
            triangles_drawn,
            backfaces_culled,
            behind_culled,
            sum_areas,
            #[cfg(feature = "performance-counters")]
            performance_counters,
        } = self;
        writeln!(f, "render metrics:")?;
        writeln!(f, "\ttriangles drawn: {triangles_drawn}")?;
        writeln!(f, "\tbackfaces culled: {backfaces_culled}")?;
        writeln!(f, "\tbehind culled: {behind_culled}")?;
        let mean_area = sum_areas as f64 / (2. * triangles_drawn as f64);
        writeln!(f, "\tmean triangle area: {mean_area:.2}")?;
        #[cfg(feature = "performance-counters")]
        writeln!(f, "{performance_counters}")?;
        Ok(())
    }
}

#[cfg(feature = "performance-counters")]
mod perf_counters {
    #[derive(Debug, Clone, Copy)]
    pub struct Counter {
        pub name: &'static str,
        pub hits: u64,
        pub cycles: u64,
    }

    impl Counter {
        pub fn new(name: &'static str) -> Self {
            Counter {
                name,
                hits: 0,
                cycles: 0,
            }
        }

        pub fn clear(&mut self) {
            self.hits = 0;
            self.cycles = 0;
        }
    }

    impl std::fmt::Display for Counter {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            let &Counter { name, hits, cycles } = self;
            if hits > 0 {
                let cy_per_hit = cycles as f64 / hits as f64;
                write!(f, "{name}: {hits} hits, {cycles} cycles, {cy_per_hit:.2} cycles/hit")
            } else {
                write!(f, "{name}: -")
            }
        }
    }

    macro_rules! register_performance_counters {
        (
            pub struct $struct_name:ident;

            $($name:ident,)*
        ) => {

            #[derive(Debug, Clone, Copy)]
            pub struct $struct_name {
                $(pub $name: Counter,)*
            }

            impl Default for $struct_name {
                fn default() -> Self {
                    $struct_name {
                        $($name: Counter::new(stringify!($name)),)*
                    }
                }
            }

            impl $struct_name {
                pub fn clear(&mut self) {
                    $(self.$name.clear();)*
                }
            }

            impl std::fmt::Display for PerformanceCounters {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    let &PerformanceCounters { $($name,)* } = self;
                    writeln!(f, "performance counters:")?;
                    $(writeln!(f, "\t{}", $name)?;)*
                    Ok(())
                }
            }
        };
    }

    register_performance_counters! {
        pub struct PerformanceCounters;

        triangle_setup,
        process_pixel,
        fill_pixel,
        test_pixel,
        vertex_processing,
        single_vertex,
    }
}