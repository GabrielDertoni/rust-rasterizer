use crate::{
    VertexShader, Attributes,
    vec::{Vec, Vec2i, Vec2, Vec3, Vec4}, pipeline::Metrics,
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

#[derive(Clone, Copy, Debug)]
pub struct BBox<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

/*
impl<T: Num> BBox<T> {
    pub fn intersects(&self, other: BBox<T>) -> bool {
        let x_intersection = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
        let y_intersection = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);
        x_intersection > T::zero() && y_intersection > T::zero()
    }
}
*/

pub fn ndc_to_screen(ndc: Vec2, width: f32, height: f32) -> Vec2 {
    Vec2::from([
        ndc.x * width / 2. + width / 2.,
        -ndc.y * height / 2. + height / 2.,
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
pub fn orient_2d<T: crate::vec::Num>(from: Vec<T, 2>, to: Vec<T, 2>, p: Vec<T, 2>) -> T {
    let u = to - from;
    let v = p - from;
    u.x * v.y - u.y * v.x
}

// source: https://www.cs.utexas.edu/~fussell/courses/cs354-fall2015/lectures/lecture9.pdf
// This assumes that if all points are outside the frustum, than so is the triangle
pub fn is_inside_frustum_clip(p0_clip: Vec4, p1_clip: Vec4, p2_clip: Vec4) -> bool {
    let range0 = -p0_clip.w..p0_clip.w;
    let range1 = -p1_clip.w..p1_clip.w;
    let range2 = -p2_clip.w..p2_clip.w;

    (range0.contains(&p0_clip.x) && range0.contains(&p0_clip.y) && range0.contains(&p0_clip.z))
        || (range1.contains(&p1_clip.x)
            && range1.contains(&p1_clip.y)
            && range1.contains(&p1_clip.z))
        || (range2.contains(&p2_clip.x)
            && range2.contains(&p2_clip.y)
            && range2.contains(&p2_clip.z))
}

pub fn is_inside_frustum_screen(
    p0_screen: Vec2i,
    p1_screen: Vec2i,
    p2_screen: Vec2i,
    z0: f32,
    z1: f32,
    z2: f32,
    width: i32,
    height: i32,
) -> bool {
    ((0..width).contains(&p0_screen.x)
        && (0..height).contains(&p0_screen.y)
        && (-1.0..1.0).contains(&z0))
        || ((0..width).contains(&p1_screen.x)
            && (0..height).contains(&p1_screen.y)
            && (-1.0..1.0).contains(&z1))
        || ((0..width).contains(&p2_screen.x)
            && (0..height).contains(&p2_screen.y)
            && (-1.0..1.0).contains(&z2))
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

pub fn process_vertex<Vert, V>(v: Vert, vert_shader: &V, metrics: &mut Metrics) -> V::Output
where
    V: VertexShader<Vert>,
{
    count_cycles! {
        #[counter(metrics.performance_counters.single_vertex)]

        let mut out = vert_shader.exec(v);
        let pos = out.position_mut();
        let inv_w = 1. / pos.w;
        *pos.xyz_mut() *= inv_w;
        pos.w = inv_w;
        out
    }
}

#[inline(always)]
pub fn orient_2d_step(
    from: Vec2i,
    to: Vec2i,
    p: Vec2i,
) -> (Vec2i, i32) {
    let u = to - from;
    let c = u.y * from.x - u.x * from.y;
    let w = u.x * p.y - u.y * p.x + c;
    let inc = Vec2i::from([-u.y, u.x]);
    (inc, w)
}