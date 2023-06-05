use std::simd::Simd;
use std::ops::Range;

use crate::{Attributes, VertexShader, FragmentShader, VertexBuf, Vertex, VertBuf, texture::BorrowedMutTexture, simd_config::LANES};

macro_rules! static_assert_eq {
    ($lhs:expr, $rhs:expr) => {{
        const _ASSERTION: [(); ($lhs == $rhs) as usize] = [()];
    }};
}

pub struct Pipeline<'a> {
    vert_buf: &'a VertBuf,
    index_buf: &'a [[usize; 3]],
    color_buf: Option<BorrowedMutTexture<'a, [u8; 4]>>,
    depth_buf: Option<BorrowedMutTexture<'a, f32>>,
    use_simd: bool,
    metrics: Metrics,
}

impl<'a> Pipeline<'a> {
    pub fn new(vert_buf: &'a VertBuf, index_buf: &'a [[usize; 3]]) -> Self {
        Pipeline {
            vert_buf,
            index_buf,
            color_buf: None,
            depth_buf: None,
            use_simd: false,
            metrics: Metrics::new(),
        }
    }

    pub fn set_color_buffer(&mut self, mut color_buf: BorrowedMutTexture<'a, [u8; 4]>) {
        use crate::vec::Vec;

        if let Some(depth_buf) = self.depth_buf.as_ref() {
            assert_eq!(color_buf.width(), depth_buf.width());
            assert_eq!(color_buf.height(), depth_buf.height());
        }

        if self.use_simd {
            assert!(color_buf
                .as_ptr()
                .is_aligned_to(std::mem::align_of::<Simd<u32, LANES>>()));
        
            // Coalesce the pixel layout to be [rrrrggggbbbbaaaa] repeating
            assert_eq!(color_buf.width() % LANES, 0, "width must be a multiple of LANES");
            static_assert_eq!(LANES, 4);
            assert!(color_buf
                .as_ptr()
                .is_aligned_to(std::mem::align_of::<Vec<Simd<u8, 4>, 4>>()));
            let simd_color_buf = unsafe {
                let ptr = color_buf.as_mut_ptr().cast::<Vec<Simd<u8, 4>, 4>>();
                std::slice::from_raw_parts_mut(ptr, color_buf.len() >> 2)
            };
            for el in simd_color_buf {
                *el = el.simd_transpose_4();
            }
        }
        self.color_buf.replace(color_buf);
    }

    pub fn take_color_buffer(&mut self) -> Option<BorrowedMutTexture<'a, [u8; 4]>> {
        use crate::vec::Vec;

        if self.use_simd {
            let mut color_buf = self.color_buf.take()?;
            let simd_color_buf = unsafe {
                let ptr = color_buf.as_mut_ptr().cast::<Vec<Simd<u8, 4>, 4>>();
                std::slice::from_raw_parts_mut(ptr, color_buf.len() >> 2)
            };
            for el in simd_color_buf {
                *el = el.simd_transpose_4();
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

        if self.use_simd {
            assert!(depth_buf
                .as_ptr()
                .is_aligned_to(std::mem::align_of::<Simd<f32, LANES>>()));
            assert_eq!(depth_buf.width() % LANES, 0, "width must be a multiple of LANES");
        }
        self.depth_buf.replace(depth_buf);
    }

    pub fn take_depth_buffer(&mut self) -> Option<BorrowedMutTexture<'a, f32>> {
        self.depth_buf.take()
    }

    pub fn process_vertices<'b: 'a, V: VertexShader<Vertex>>(&'b mut self, vert_shader: &V) -> ProcessedVertices<'b, V::Output> {
        use crate::prim3d::common::process_vertex;

        let attributes = (0..self.vert_buf.len())
            .map(|i| process_vertex(self.vert_buf.index(i), vert_shader))
            .collect();
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

pub struct ProcessedVertices<'a, Attr> {
    pipeline: &'a mut Pipeline<'a>,
    attributes: Vec<Attr>,
}

impl<'a, Attr: Attributes + Copy> ProcessedVertices<'a, Attr> {
    pub fn draw<F: FragmentShader<Attr>>(&mut self, range: Range<usize>, frag_shader: &F) {

        let mut color_buf = self.pipeline.color_buf.as_mut().expect("no color buffer").borrow_mut();
        let mut depth_buf = self.pipeline.depth_buf.as_mut().expect("no depth buffer").borrow_mut();

        if self.pipeline.use_simd {
            use crate::prim3d::simd::draw_triangles;

            let color_buf = unsafe { color_buf.cast::<u32>() };
            draw_triangles(
                &self.attributes,
                &self.pipeline.index_buf[range],
                frag_shader,
                color_buf,
                depth_buf,
                &mut self.pipeline.metrics,
            );
        } else {
            use crate::prim3d::draw_triangles;

            draw_triangles(
                &self.attributes,
                &self.pipeline.index_buf[range],
                frag_shader,
                color_buf,
                depth_buf,
                &mut self.pipeline.metrics,
            );
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
    pub(crate) triangles_drawn: usize,
    pub(crate) backfaces_culled: usize,
    pub(crate) behind_culled: usize,
    pub(crate) sum_areas: i64,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            triangles_drawn: 0,
            backfaces_culled: 0,
            behind_culled: 0,
            sum_areas: 0,
        }
    }

    pub fn combine(&mut self, other: Metrics) {
        self.triangles_drawn += other.triangles_drawn;
        self.backfaces_culled += other.backfaces_culled;
        self.behind_culled += other.behind_culled;
        self.sum_areas += other.sum_areas;
    }
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let &Metrics {
            triangles_drawn,
            backfaces_culled,
            behind_culled,
            sum_areas,
        } = self;
        writeln!(f, "render metrics:")?;
        writeln!(f, "\ttriangles drawn: {triangles_drawn}")?;
        writeln!(f, "\tbackfaces culled: {backfaces_culled}")?;
        writeln!(f, "\tbehind culled: {behind_culled}")?;
        let mean_area = sum_areas as f64 / (2. * triangles_drawn as f64);
        writeln!(f, "\tmean triangle area: {mean_area}")?;
        Ok(())
    }
}