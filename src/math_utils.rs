use std::{simd::{Simd, SimdFloat, StdFloat, LaneCount, SupportedLaneCount}, ops::Range};

#[allow(non_upper_case_globals)]
pub const ZERO_f32x4: Simd<f32, 4> = Simd::from_array([0., 0., 0., 0.]);
#[allow(non_upper_case_globals)]
pub const ONE_f32x4: Simd<f32, 4> = Simd::from_array([1., 1., 1., 1.]);

#[inline(always)]
pub fn simd_clamp01<const LANES: usize>(value: Simd<f32, LANES>) -> Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    value.simd_clamp(Simd::splat(0.), Simd::splat(1.))
}

#[inline(always)]
pub fn simd_wrap01<const LANES: usize>(value: Simd<f32, LANES>) -> Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // The clamp is just there because the inner expression won't handle `-inf` or `inf`
    simd_clamp01(value - value.floor())
}

pub fn remap(val: f32, range: Range<f32>, dest: Range<f32>) -> f32 {
    dest.start + (dest.end - dest.start) * (val - range.start) / (range.end - range.start)
}

pub fn remap_simd(val: Simd<f32, 4>, range: Range<Simd<f32, 4>>, dest: Range<Simd<f32, 4>>) -> Simd<f32, 4> {
    dest.start + (dest.end - dest.start) * (val - range.start) / (range.end - range.start)
}