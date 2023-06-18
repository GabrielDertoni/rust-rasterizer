use std::simd::{Simd, Mask};

use crate::vec::Vec;

pub use lanes4::*;

#[allow(dead_code)]
mod lanes4 {
    use super::*;

    pub const LANES: usize = 4;
    
    pub(crate) const STEP_X: i32 = 4;
    pub(crate) const STEP_Y: i32 = 1;
    pub(crate) const X_OFF: Simd<i32, LANES> = Simd::from_array([0, 1, 2, 3]);
    pub(crate) const Y_OFF: Simd<i32, LANES> = Simd::from_array([0, 0, 0, 0]);
}

#[allow(dead_code)]
mod lanes8 {
    use super::*;

    pub const LANES: usize = 8;
    
    pub(crate) const STEP_X: i32 = 8;
    pub(crate) const STEP_Y: i32 = 1;
    pub(crate) const X_OFF: Simd<i32, LANES> = Simd::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    pub(crate) const Y_OFF: Simd<i32, LANES> = Simd::from_array([0, 0, 0, 0, 0, 0, 0, 0]);
}

pub type SimdF32 = Simd<f32, LANES>;
pub type SimdI32 = Simd<i32, LANES>;

pub type SimdVec2 = Vec<Simd<f32, LANES>, 2>;
pub type SimdVec3 = Vec<Simd<f32, LANES>, 3>;
pub type SimdVec4 = Vec<Simd<f32, LANES>, 4>;

pub type SimdMask = Mask<i32, LANES>;

pub type SimdPixels = Simd<u32, LANES>;
pub type SimdColor = SimdVec4;
pub type SimdColorGamma = Vec<Simd<u8, LANES>, 4>;