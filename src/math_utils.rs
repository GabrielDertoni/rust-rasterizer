use std::{simd::{Simd, SimdFloat, StdFloat, LaneCount, SupportedLaneCount}, ops::Range};

use crate::vec::Vec;

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

pub fn simd_remap<const LANES: usize>(val: Simd<f32, LANES>, range: Range<Simd<f32, LANES>>, dest: Range<Simd<f32, LANES>>) -> Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    dest.start + (dest.end - dest.start) * (val - range.start) / (range.end - range.start)
}

pub fn simd_mix<const LANES: usize>(c1: Vec<Simd<f32, LANES>, 3>, c2: Vec<Simd<f32, LANES>, 3>, factor: Simd<f32, LANES>) -> Vec<Simd<f32, LANES>, 3>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    c1 * factor + c2 * (Simd::splat(1.) - factor)
}

pub fn rgb_hex(c: u32) -> Vec<f32, 3> {
    let [_, r, g, b] = c.to_be_bytes();
    Vec::from([r as f32, g as f32, b as f32]) / 255.
}