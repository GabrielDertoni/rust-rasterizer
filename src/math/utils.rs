use std::{simd::{Simd, SimdFloat, StdFloat, LaneCount, SupportedLaneCount}, ops::Range};

use crate::{math::{Vec, Vec3, Vec4}, simd_config::{SimdColorSRGB, SimdColorRGB}};

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

pub fn color_to_u32(c: Vec4) -> u32 {
    u32::from_be_bytes(c.map(|chan| (chan * 255.) as u8).to_array())
}

#[inline]
pub fn simd_srgb_to_rgb(c: SimdColorSRGB) -> SimdColorRGB {
    c.map(simd_gamma_to_linear)
}

#[inline]
pub fn simd_srgb_to_rgb_f32<const LANES: usize>(c: Vec<Simd<f32, LANES>, 3>) -> Vec<Simd<f32, LANES>, 3>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    c.map(simd_gamma_to_linear_f32)
}

#[inline]
pub fn simd_gamma_to_linear<const LANES: usize>(chan: Simd<u8, LANES>) -> Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let chan = chan.cast::<f32>() / Simd::splat(255.);
    chan * chan
}

#[inline]
pub fn simd_gamma_to_linear_f32<const LANES: usize>(chan: Simd<f32, LANES>) -> Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    chan * chan
}

#[inline]
pub fn simd_rgb_to_srgb(c: SimdColorRGB) -> SimdColorSRGB {
    c.map(simd_linear_to_gamma)
}

#[inline]
pub fn simd_linear_to_gamma<const LANES: usize>(chan: Simd<f32, LANES>) -> Simd<u8, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    (chan.sqrt() * Simd::splat(255.)).cast::<u8>()
}

pub fn srgb_to_linear(c: Vec<u8, 3>) -> Vec3 {
    c.map(gamma_to_linear)
}

pub fn srgb_to_linear_f32(c: Vec<f32, 3>) -> Vec3 {
    c.map(gamma_to_linear_f32)
}

pub fn gamma_to_linear(chan: u8) -> f32 {
    let chan = chan as f32 / 255.;
    chan * chan
}

pub fn gamma_to_linear_f32(chan: f32) -> f32 {
    chan * chan
}