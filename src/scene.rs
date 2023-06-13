use std::path::{PathBuf, Path};

use serde::Deserialize;
use anyhow::{Result, Context};

use crate::{vec::Vec3, utils::FpvCamera};

#[derive(Clone, Deserialize)]
pub struct Scene {
    pub models: Vec<Model>,
    pub camera: Camera,
    pub rendering: RenderingConfig,
    #[serde(default)]
    pub rasterizer: RasterizerConfig,
    #[serde(default)]
    pub lights: Vec<Light>,
    #[serde(default)]
    pub config: toml::Table,
}

impl Scene {
    pub fn load_toml(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read file {path:?}"))?;
        let scene = toml::from_str(&contents)?;
        Ok(scene)
    }
}

#[derive(Clone, Deserialize)]
pub struct Model {
    pub path: PathBuf,
    #[serde(default = "Vec3::zero", deserialize_with = "detail::deser_vec3")]
    pub position: Vec3,
    /// Euler angles measured in degrees
    #[serde(default = "Vec3::zero", deserialize_with = "detail::deser_vec3")]
    pub rotation: Vec3,
    #[serde(default = "Vec3::one", deserialize_with = "detail::deser_vec3")]
    pub scale: Vec3,
}

#[derive(Clone, Copy, Deserialize)]
pub struct Camera {
    pub kind: CameraKind,
    #[serde(deserialize_with = "detail::deser_vec3")]
    pub position: Vec3,
    /// Euler angles measured in degrees
    #[serde(default = "Vec3::zero", deserialize_with = "detail::deser_vec3")]
    pub rotation: Vec3,
    #[serde(default = "Camera::default_axis")]
    pub up: Axis,
    pub fovy: f32,
    #[serde(default = "Camera::default_sensitivity")]
    pub sensitivity: f32,
    #[serde(default = "Camera::default_speed")]
    pub speed: f32,
}

impl Camera {
    pub fn into_fpv(self, aspect_ratio: f32) -> FpvCamera {
        match self.kind {
            CameraKind::Fpv => {
                if self.rotation.y != 0.0 {
                    eprintln!("[WARNING]: Ignoring y rotation when using FPV camera");
                }
                FpvCamera {
                    position: self.position,
                    up: self.up.into_vec(),
                    pitch: self.rotation.x,
                    yaw: self.rotation.z,
                    sensitivity: self.sensitivity,
                    speed: self.speed,
                    fovy: self.fovy,
                    ratio: aspect_ratio,
                }
            }
        }
    }

    fn default_axis() -> Axis {
        Axis::Z
    }

    fn default_sensitivity() -> f32 {
        0.05
    }

    fn default_speed() -> f32 {
        5.0
    }
}

#[derive(Clone, Copy, Deserialize)]
pub enum CameraKind {
    #[serde(rename = "fpv")]
    Fpv,
}

#[derive(Clone, Copy, Deserialize)]
pub enum Axis {
    #[serde(rename = "x")]
    X,
    #[serde(rename = "-x")]
    NegativeX,
    #[serde(rename = "y")]
    Y,
    #[serde(rename = "-y")]
    NegativeY,
    #[serde(rename = "z")]
    Z,
    #[serde(rename = "-z")]
    NegativeZ,
}

impl Axis {
    pub fn into_vec(self) -> Vec3 {
        use Axis::*;

        match self {
            X => Vec3::from([1., 0., 0.]),
            NegativeX => Vec3::from([-1., 0., 0.]),
            Y => Vec3::from([0., 1., 0.]),
            NegativeY => Vec3::from([0., -1., 0.]),
            Z => Vec3::from([0., 0., 1.]),
            NegativeZ => Vec3::from([0., 0., -1.]),
        }
    }
}

#[derive(Clone, Copy, Deserialize)]
pub struct RenderingConfig {
    pub width: usize,
    pub height: usize,
    #[serde(default = "RenderingConfig::near_default")]
    pub near: f32,
    #[serde(default = "RenderingConfig::far_default")]
    pub far: f32,
}

impl RenderingConfig {
    fn near_default() -> f32 {
        0.1
    }

    fn far_default() -> f32 {
        100.
    }
}

#[derive(Clone, Copy, Default, Deserialize)]
pub struct RasterizerConfig {
    pub implementation: RasterizerImplementation,
}

#[derive(Clone, Copy, Default, Deserialize)]
pub enum RasterizerImplementation {
    #[default]
    #[serde(rename = "bbox-simd")]
    BBoxSimd,
    #[serde(rename = "bbox")]
    BBox,
    #[serde(rename = "scanline")]
    Scanline,
}

#[derive(Clone, Copy, Deserialize)]
pub struct Light {
    pub kind: LightKind,
    #[serde(deserialize_with = "detail::deser_vec3")]
    pub position: Vec3,
    #[serde(deserialize_with = "detail::deser_vec3")]
    pub forward: Vec3,
    #[serde(default)]
    pub shadow_map_size: Option<usize>,
}

#[derive(Clone, Copy, Deserialize)]
pub enum LightKind {
    Point,
    Directional,
    Spot,
}

mod detail {
    use serde::de::{Deserializer, Deserialize};

    use crate::vec::Vec3;

    pub fn deser_vec3<'de, D>(deserializer: D) -> Result<Vec3, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Vec3::from(<[f32; 3] as Deserialize>::deserialize(deserializer)?))
    }
}