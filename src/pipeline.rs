use std::{ops::Range, simd::Simd, time::Duration};

use rayon::prelude::*;

use crate::{
    common::count_cycles,
    config::CullingMode,
    math::{BBox, Size},
    simd_config::{SimdColorGamma, SimdPixels, LANES},
    texture::BorrowedMutTexture,
    vec::Vec2i,
    vert_buf::VertBuf,
    Attributes, AttributesSimd, FragmentShader, FragmentShaderImpl, FragmentShaderSimd, VertexBuf,
    VertexShader,
};

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
    alpha_clip: Option<f32>,
    viewport: Size<usize>,
}

impl<'a, B> Pipeline<'a, B>
where
    B: VertexBuf,
{
    pub fn new(
        vert_buf: B,
        index_buf: &'a [[usize; 3]],
        viewport: Size<usize>,
        mode: PipelineMode,
    ) -> Self {
        Pipeline {
            vert_buf,
            index_buf,
            color_buf: None,
            depth_buf: None,
            mode,
            metrics: Metrics::new(),
            culling_mode: CullingMode::BackFace,
            alpha_clip: None,
            viewport,
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
                    (values & ff_mask).cast::<u8>(),
                    ((values >> Simd::splat(8)) & ff_mask).cast::<u8>(),
                    ((values >> Simd::splat(16)) & ff_mask).cast::<u8>(),
                    ((values >> Simd::splat(24)) & ff_mask).cast::<u8>(),
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
                let v = values.x
                    | (values.y << Simd::splat(8))
                    | (values.z << Simd::splat(16))
                    | (values.w << Simd::splat(24));

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

    fn viewport_f32(&self) -> Size<f32> {
        Size {
            width: self.viewport.width as f32,
            height: self.viewport.height as f32,
        }
    }

    pub fn process_vertices<'pipeline, V: VertexShader<B::Vertex>>(
        &'pipeline mut self,
        vert_shader: &V,
        vertex_range: Option<Range<usize>>,
    ) -> ProcessedVertices<'pipeline, 'a, V::Output, B> {
        use crate::common::process_vertex;

        let vertex_range = vertex_range.unwrap_or(0..self.vert_buf.len());
        let start = std::time::Instant::now();

        let attributes = count_cycles! {
            #[counter(self.metrics.performance_counters.vertex_processing)]

            vertex_range.clone()
                .map(|i| process_vertex(self.vert_buf.index(i), vert_shader, self.viewport_f32(), &mut self.metrics))
                .collect()
        };

        self.metrics.vertex_processing_time.count(start.elapsed());

        ProcessedVertices {
            pipeline: self,
            vertex_range,
            attributes,
        }
    }

    pub fn process_vertices_par<'pipeline, V>(
        &'pipeline mut self,
        vert_shader: &V,
        vertex_range: Option<Range<usize>>,
    ) -> ProcessedVertices<'pipeline, 'a, V::Output, B>
    where
        B: Sync,
        V: Sync,
        V: VertexShader<B::Vertex>,
        V::Output: Send,
    {
        use crate::common::process_vertex;

        let vertex_range = vertex_range.unwrap_or(0..self.vert_buf.len());
        let start = std::time::Instant::now();

        let mut attributes = Vec::new();
        count_cycles! {
            #[counter(self.metrics.performance_counters.vertex_processing)]

            vertex_range.clone()
                .into_par_iter()
                .map(|i| {
                    let mut metrics = Metrics::new();
                    process_vertex(self.vert_buf.index(i), vert_shader, self.viewport_f32(), &mut metrics)
                })
                .collect_into_vec(&mut attributes);
        }

        self.metrics.vertex_processing_time.count(start.elapsed());

        ProcessedVertices {
            pipeline: self,
            vertex_range,
            attributes,
        }
    }

    pub fn finalize(&mut self) {
        self.take_color_buffer();
        self.take_depth_buffer();
    }

    pub fn get_metrics(&self) -> Metrics {
        self.metrics
    }
}

pub struct ProcessedVertices<'pipeline, 'a, Attr, B = &'a VertBuf> {
    pipeline: &'pipeline mut Pipeline<'a, B>,
    vertex_range: Range<usize>,
    attributes: Vec<Attr>,
}

impl<'pipeline, 'a, Attr, B> ProcessedVertices<'pipeline, 'a, Attr, B>
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

                let bbox = BBox::new(
                    Vec2i::zero(),
                    Size::new(
                        self.pipeline.viewport.width as i32,
                        self.pipeline.viewport.height as i32,
                    ),
                );

                draw_triangles(
                    &self.attributes,
                    &self.pipeline.index_buf[range],
                    self.vertex_range.clone(),
                    frag_shader.unwrap_simd(),
                    color_buf,
                    depth_buf,
                    bbox,
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

    pub fn draw_par<Scalar, Simd>(
        &mut self,
        range: Range<usize>,
        frag_shader: FragmentShaderImpl<Scalar, Simd>,
    ) where
        Scalar: FragmentShader<Attr> + Sync,
        Simd: FragmentShaderSimd<Attr, LANES> + Sync,
        Attr: Sync,
        B::Vertex: Attributes,
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

                let mut tiles = crate_tiles(Size::new(256, 256), color_buf, depth_buf);
                let n_tiles = tiles.len();
                let n_threads = std::thread::available_parallelism().unwrap().get();

                let bins_init = || -> Vec<Vec<Vec<[usize; 3]>>> {
                    (0..n_tiles)
                        .map(|_| Vec::with_capacity(n_threads))
                        .collect::<Vec<_>>()
                };

                let start = std::time::Instant::now();

                // TODO: This is pretty bad... Ideally we would just send an index to `bin_triangles` that tells it which
                // of the bins it can fill out for each tile based on the thread id. However, rust won't let that happen,
                // as it can't prove that the mutable accesses won't overlap.
                let mut bins = self.pipeline.index_buf[range.clone()]
                    .as_parallel_slice()
                    .par_chunks(((range.end - range.start) / n_threads).max(1))
                    .map(|indices| bin_triangles(&tiles, &self.attributes, indices, self.vertex_range.clone()))
                    .fold(bins_init, |mut acc, el| {
                        for (acc, el) in acc.iter_mut().zip(el) {
                            acc.push(el);
                        }
                        acc
                    })
                    // TODO: This isn't good. It's doing some needless O(n) operation. Find another way to merge the vecs.
                    .reduce(bins_init, |mut lhs, mut rhs| {
                        for (lhs, rhs) in lhs.iter_mut().zip(rhs.iter_mut()) {
                            lhs.append(rhs);
                        }
                        lhs
                    });

                for (tile, bin) in tiles.iter_mut().zip(&mut bins) {
                    tile.bins = std::mem::take(bin);
                }

                self.pipeline.metrics.binning_time.count(start.elapsed());

                let frag_shader = frag_shader.unwrap_simd();
                let metrics = tiles
                    .into_par_iter()
                    .map(|mut tile| {
                        let mut metrics = Metrics::new();
                        for bin in &tile.bins {
                            draw_triangles(
                                &self.attributes,
                                &bin,
                                self.vertex_range.clone(),
                                frag_shader,
                                tile.color_buf.borrow_mut(),
                                tile.depth_buf.borrow_mut(),
                                BBox {
                                    x: tile.bbox.x as i32,
                                    y: tile.bbox.y as i32,
                                    width: tile.bbox.width as i32,
                                    height: tile.bbox.height as i32,
                                },
                                self.pipeline.culling_mode,
                                self.pipeline.alpha_clip,
                                &mut metrics,
                            );
                        }
                        metrics
                    })
                    .reduce(
                        || Metrics::new(),
                        |mut lhs, rhs| {
                            lhs.merge(rhs);
                            lhs
                        },
                    );
                self.pipeline.metrics.merge(metrics);
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
        self.pipeline.get_metrics()
    }

    pub fn finalize(&mut self) {
        self.pipeline.finalize();
    }
}

pub struct Tile<'a> {
    bbox: BBox<f32>,
    color_buf: BorrowedMutTexture<'a, u32>,
    depth_buf: BorrowedMutTexture<'a, f32>,
    bins: Vec<Vec<[usize; 3]>>,
}

fn bin_triangles<'a, Attr>(
    tiles: &'a [Tile],
    attributes: &'a [Attr],
    indices: &'a [[usize; 3]],
    vertex_range: Range<usize>,
) -> Vec<Vec<[usize; 3]>>
where
    Attr: Attributes,
{
    const BIN_INIT_CAPACITY: usize = 1024;

    let mut processed: Vec<Vec<[usize; 3]>> =
        std::iter::repeat_with(|| Vec::with_capacity(BIN_INIT_CAPACITY))
            .take(tiles.len())
            .collect();

    for &[ix0, ix1, ix2] in indices {
        let v0 = attributes[ix0 - vertex_range.start].position().xy();
        let v1 = attributes[ix1 - vertex_range.start].position().xy();
        let v2 = attributes[ix2 - vertex_range.start].position().xy();

        let min = v0.min(v1).min(v2);
        let max = v0.max(v1).max(v2);

        let tri_bbox = BBox {
            x: min.x,
            y: min.y,
            width: max.x - min.x,
            height: max.y - min.y,
        };
        for (index, tile) in tiles.iter().enumerate() {
            if tile.bbox.intersects(tri_bbox) {
                processed[index].push([ix0, ix1, ix2]);
            }
        }
    }

    processed
}

fn crate_tiles<'a>(
    tile_size: Size<usize>,
    color_buf: BorrowedMutTexture<'a, u32>,
    depth_buf: BorrowedMutTexture<'a, f32>,
) -> Vec<Tile<'a>> {
    let color_tiles = tile_texture_sized(tile_size, color_buf);
    let depth_tiles = tile_texture_sized(tile_size, depth_buf);

    color_tiles
        .zip(depth_tiles)
        .map(|((color_buf, color_bbox), (depth_buf, depth_bbox))| {
            assert_eq!(color_bbox, depth_bbox);
            Tile {
                bbox: BBox {
                    x: color_bbox.x as f32,
                    y: color_bbox.y as f32,
                    width: color_bbox.width as f32,
                    height: color_bbox.height as f32,
                },
                color_buf,
                depth_buf,
                bins: Vec::new(),
            }
        })
        .collect()
}

fn tile_texture_sized<'a, T>(
    size: Size<usize>,
    tex: BorrowedMutTexture<'a, T>,
) -> impl Iterator<Item = (BorrowedMutTexture<'a, T>, BBox<usize>)> {
    let full_tiles_width = tex.width() / size.width;
    let remainder_tile_width = tex.width() % size.width;
    let full_tiles_height = tex.height() / size.height;
    let remainder_tile_height = tex.height() % size.height;
    let tile_widths = std::iter::repeat(size.width)
        .take(full_tiles_width)
        .chain((remainder_tile_width != 0).then_some(remainder_tile_width));
    let tile_heights = std::iter::repeat(size.height)
        .take(full_tiles_height)
        .chain((remainder_tile_height != 0).then_some(remainder_tile_height));
    tile_texture(tile_widths, tile_heights, tex)
}

fn tile_texture<'a, T>(
    tile_widths: impl IntoIterator<Item = usize>,
    tile_heights: impl IntoIterator<Item = usize> + Clone,
    tex: BorrowedMutTexture<'a, T>,
) -> impl Iterator<Item = (BorrowedMutTexture<'a, T>, BBox<usize>)> {
    tile_widths
        .into_iter()
        .scan((tex, 0), move |(rest_horz, x), width| {
            let (tile_horz, tail) = rest_horz.split_mut_vert(width);
            let curr_x = *x;
            *rest_horz = tail;
            *x += width;
            Some(tile_heights.clone().into_iter().scan(
                (tile_horz, 0),
                move |(rest_vert, y), height| {
                    let (tile, tail) = rest_vert.split_mut_horz(height);
                    let curr_y = *y;
                    *rest_vert = tail;
                    *y += height;
                    Some((
                        tile,
                        BBox {
                            x: curr_x,
                            y: curr_y,
                            width,
                            height,
                        },
                    ))
                },
            ))
        })
        .flatten()
}

#[derive(Debug, Clone, Copy)]
pub struct Metrics {
    pub triangles_drawn: usize,
    pub backfaces_culled: usize,
    pub behind_culled: usize,
    pub sum_areas: i64,
    pub binning_time: TimeCounter,
    pub vertex_processing_time: TimeCounter,
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
            binning_time: TimeCounter::new("binning time"),
            vertex_processing_time: TimeCounter::new("vertex processing time"),
            #[cfg(feature = "performance-counters")]
            performance_counters: Default::default(),
        }
    }

    pub fn merge(&mut self, other: Metrics) {
        self.triangles_drawn += other.triangles_drawn;
        self.backfaces_culled += other.backfaces_culled;
        self.sum_areas += other.sum_areas;
        self.binning_time.merge(other.binning_time);
        self.vertex_processing_time
            .merge(other.vertex_processing_time);
        #[cfg(feature = "performance-counters")]
        self.performance_counters.merge(other.performance_counters);
    }

    pub fn clear(&mut self) {
        self.triangles_drawn = 0;
        self.backfaces_culled = 0;
        self.behind_culled = 0;
        self.sum_areas = 0;
        self.binning_time.clear();
        self.vertex_processing_time.clear();
        #[cfg(feature = "performance-counters")]
        self.performance_counters.clear();
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            triangles_drawn: Default::default(),
            backfaces_culled: Default::default(),
            behind_culled: Default::default(),
            sum_areas: Default::default(),
            binning_time: TimeCounter::new("binning time"),
            vertex_processing_time: TimeCounter::new("vertex processing time"),
            #[cfg(feature = "performance-counters")]
            performance_counters: Default::default(),
        }
    }
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let &Metrics {
            triangles_drawn,
            backfaces_culled,
            behind_culled,
            sum_areas,
            binning_time,
            vertex_processing_time,
            #[cfg(feature = "performance-counters")]
            performance_counters,
        } = self;
        writeln!(f, "render metrics:")?;
        writeln!(f, "\ttriangles drawn: {triangles_drawn}")?;
        writeln!(f, "\tbackfaces culled: {backfaces_culled}")?;
        writeln!(f, "\tbehind culled: {behind_culled}")?;
        writeln!(f, "\t{binning_time}",)?;
        writeln!(f, "\t{vertex_processing_time}",)?;
        let mean_area = sum_areas as f64 / (2. * triangles_drawn as f64);
        writeln!(f, "\tmean triangle area: {mean_area:.2}")?;
        #[cfg(feature = "performance-counters")]
        writeln!(f, "{performance_counters}")?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TimeCounter {
    pub name: &'static str,
    pub hits: u64,
    pub total_time: Duration,
}

impl TimeCounter {
    pub fn new(name: &'static str) -> Self {
        TimeCounter {
            name: name,
            hits: 0,
            total_time: Duration::ZERO,
        }
    }

    pub fn count(&mut self, time: Duration) {
        self.hits += 1;
        self.total_time += time;
    }

    pub fn merge(&mut self, other: TimeCounter) {
        self.hits += other.hits;
        self.total_time += other.total_time;
    }

    pub fn clear(&mut self) {
        self.hits = 0;
        self.total_time = Duration::ZERO;
    }
}

impl std::fmt::Display for TimeCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let &TimeCounter {
            name,
            hits,
            total_time,
        } = self;
        if hits > 0 {
            let t_per_hit = total_time / hits as u32;
            write!(
                f,
                "{name}: {hits} hits, {total_time:?} total time, {t_per_hit:.2?} time/hit"
            )
        } else {
            write!(f, "{name}: -")
        }
    }
}

#[cfg(feature = "performance-counters")]
mod perf_counters {
    #[derive(Debug, Clone, Copy)]
    pub struct CycleCounter {
        pub name: &'static str,
        pub hits: u64,
        pub cycles: u64,
    }

    impl CycleCounter {
        pub fn new(name: &'static str) -> Self {
            CycleCounter {
                name,
                hits: 0,
                cycles: 0,
            }
        }

        pub fn merge(&mut self, other: Self) {
            self.hits += other.hits;
            self.cycles += other.cycles;
        }

        pub fn clear(&mut self) {
            self.hits = 0;
            self.cycles = 0;
        }
    }

    impl std::fmt::Display for CycleCounter {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            let &CycleCounter { name, hits, cycles } = self;
            if hits > 0 {
                let cy_per_hit = cycles as f64 / hits as f64;
                write!(
                    f,
                    "{name}: {hits} hits, {cycles} cycles, {cy_per_hit:.2} cycles/hit"
                )
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
                $(pub $name: CycleCounter,)*
            }

            impl Default for $struct_name {
                fn default() -> Self {
                    $struct_name {
                        $($name: CycleCounter::new(stringify!($name)),)*
                    }
                }
            }

            impl $struct_name {
                pub fn merge(&mut self, other: Self) {
                    $(self.$name.merge(other.$name);)*
                }

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
