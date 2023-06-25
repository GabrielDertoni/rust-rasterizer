use crate::{
    pipeline::Metrics,
    math::{Size, Vec, Vec2, Vec2i, Vec3, Num},
    Attributes, VertexShader,
};

macro_rules! count_cycles {
    (
        #[counter($counter:expr $(, increment = $increment:expr)?)]
        $($code:tt)*
    ) => {
        {
            #[cfg(feature = "performance-counters")]
            let start = unsafe { core::arch::x86_64::_rdtsc() };

            let res = {
                $($code)*
            };

            #[cfg(feature = "performance-counters")]
            let cycles = unsafe { core::arch::x86_64::_rdtsc() - start };

            #[cfg(feature = "performance-counters")]
            {
                $counter.cycles += cycles;
                $($counter.hits += $increment - 1;)?
                $counter.hits += 1;
            }

            res
        }
    };
}

pub(crate) use count_cycles;

#[allow(dead_code)]
pub fn ndc_to_screen(ndc: Vec2, width: f32, height: f32) -> Vec2 {
    Vec2::from([
        ndc.x * width / 2. + width / 2.,
        -ndc.y * height / 2. + height / 2.,
    ])
}

pub fn ndc_to_viewport(ndc: Vec2, viewport: Size<f32>) -> Vec2 {
    Vec2::from([
        ndc.x * viewport.width / 2. + viewport.width / 2.,
        -ndc.y * viewport.height / 2. + viewport.height / 2.,
    ])
}

/// Returns the oriented area of the paralelogram formed by the points `from`, `to`, `p`, `from + (p - to)`. The sign
/// is positive if the points in the paralelogram wind counterclockwise (according to the order given prior) and
/// negative otherwise. In other words, if you were at `from` looking towards `to`, when `p` is to your left, the
/// value would be positive, and if it is to your right the value is negative.
///
/// ## Relationship with barycentric coordinates
///
/// This function's return value has a neat relationship with barycentric coordinates: for any triangle ABC, the barycentric
/// coordinate of a point P, named W has components:
///
/// - `W.x = orient_2d(A, B, P) / orient_2d(A, B, C)`
/// - `W.y = orient_2d(B, C, P) / orient_2d(A, B, C)`
/// - `W.z = orient_2d(C, A, P) / orient_2d(A, B, C)`
///
/// It's also worth noting that `orient_2d(A, B, C)` is twice the area of the triangle ABC.
pub fn orient_2d<T: Num>(from: Vec<T, 2>, to: Vec<T, 2>, p: Vec<T, 2>) -> T {
    let u = to - from;
    let v = p - from;
    u.x * v.y - u.y * v.x
}

pub fn is_inside_frustum(p0_ndc: Vec3, p1_ndc: Vec3, p2_ndc: Vec3) -> bool {
    ((-1.0..1.0).contains(&p0_ndc.x)
        && (-1.0..1.0).contains(&p0_ndc.y)
        && (-1.0..1.0).contains(&p0_ndc.z))
        || ((-1.0..1.0).contains(&p1_ndc.x)
            && (-1.0..1.0).contains(&p1_ndc.y)
            && (-1.0..1.0).contains(&p1_ndc.z))
        || ((-1.0..1.0).contains(&p2_ndc.x)
            && (-1.0..1.0).contains(&p2_ndc.y)
            && (-1.0..1.0).contains(&p2_ndc.z))
}

pub fn process_vertex<Vert, V>(
    v: Vert,
    vert_shader: &V,
    viewport: Size<f32>,
    #[allow(unused_variables)] metrics: &mut Metrics,
) -> V::Output
where
    V: VertexShader<Vert>,
{
    count_cycles! {
        #[counter(metrics.performance_counters.single_vertex)]

        let mut out = vert_shader.exec(v);
        let pos = out.position_mut();
        let inv_w = 1. / pos.w;
        *pos.xyz_mut() *= inv_w;
        *pos.xy_mut() = ndc_to_viewport(pos.xy(), viewport);
        pos.w = inv_w;
        out
    }
}

#[inline(always)]
pub fn orient_2d_step(from: Vec2i, to: Vec2i, p: Vec2i) -> (Vec2i, i32) {
    let u = to - from;
    let c = u.y * from.x - u.x * from.y;
    let w = u.x * p.y - u.y * p.x + c;
    let inc = Vec2i::from([-u.y, u.x]);
    (inc, w)
}

/// Check if a give edge is top or left. Acording to D3D10.
///
/// > A top edge, is an edge that is exactly horizontal and is above the other edges.
/// > A left edge, is an edge that is not exactly horizontal and is on the left side of the triangle.
///
/// This allows us to determine if a pixel should be filled when it is on the very edge of a triangle.
///
/// > Any pixel center which falls inside a triangle is drawn; a pixel is assumed to be inside if it
/// > passes the top-left rule. The top-left rule is that a pixel center is defined to lie inside of
/// > a triangle if it lies on the top edge or the left edge of a triangle.
///
/// ## Sources:
/// - [Rasterization Rules](https://learn.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules?redirectedfrom=MSDN#Triangle)
/// - [Triangle Rasterization in Practice](https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/)
///
pub fn is_top_left(from: Vec2i, to: Vec2i) -> bool {
    // TODO: Things look reversed... Try to figure out why! (It's the way it looks best)
    let edge = to - from;
    let is_top = edge.y == 0 && edge.x > 0;
    let is_left = edge.y < 0;
    is_top || is_left
}
