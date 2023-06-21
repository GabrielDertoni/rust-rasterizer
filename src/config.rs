use std::path::{PathBuf, Path};

use serde::Deserialize;
use anyhow::{Result, Context};

use crate::{vec::{Vec3, Vec4}, utils::FpvCamera};

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
    pub name: String,
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
    #[serde(default)]
    pub fps: Option<f32>,
    #[serde(default = "RenderingConfig::near_default")]
    pub near: f32,
    #[serde(default = "RenderingConfig::far_default")]
    pub far: f32,
    #[serde(default, rename = "alpha-clip")]
    pub alpha_clip: Option<f32>,
    #[serde(default, rename = "cull-mode")]
    pub culling_mode: CullingMode,
    // TODO: Make validation occurr on parsing
    #[serde(
        default = "RenderingConfig::default_fog_color",
        deserialize_with = "RenderingConfig::deserialize_fog_color",
    )]
    pub fog_color: Vec4,
}

impl RenderingConfig {
    pub fn near_default() -> f32 {
        0.1
    }

    pub fn far_default() -> f32 {
        100.
    }

    pub fn default_fog_color() -> Vec4 {
        use crate::vec::Vec;

        Vec::from(0x61_b7_e8_ff_u32.to_be_bytes()).map(|chan| chan as f32 / 255.)
    }

    fn deserialize_fog_color<'de, D: serde::Deserializer<'de>>(deser: D) -> Result<Vec4, D::Error> {
        use serde::de::Error;
        use crate::vec::Vec;

        let hex_color: &str = Deserialize::deserialize(deser)?;
        let rgb = u32::from_str_radix(hex_color.strip_prefix("#").unwrap_or(""), 16).map_err(Error::custom)?;
        let rgba = (rgb << 8) | 0xff;

        Ok(Vec::from(rgba.to_be_bytes()).map(|chan| chan as f32) / 255.)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub enum CullingMode {
    #[serde(rename = "front-face")]
    FrontFace,
    #[serde(rename = "back-face")]
    BackFace,
    #[serde(rename = "disabled")]
    #[default]
    Disabled,
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

#[derive(Clone, Deserialize)]
pub struct Light {
    pub name: String,
    pub kind: LightKind,
    #[serde(deserialize_with = "detail::deser_vec3")]
    pub position: Vec3,
    #[serde(deserialize_with = "detail::deser_vec3")]
    pub forward: Vec3,
    #[serde(default = "Light::default_color", deserialize_with = "detail::deser_vec3")]
    pub color: Vec3,
    #[serde(default)]
    pub shadow_map_size: Option<usize>,
}

#[derive(Clone, Copy, Deserialize)]
pub enum LightKind {
    #[serde(rename = "point")]
    Point,
    #[serde(rename = "directional")]
    Directional,
    #[serde(rename = "spot")]
    Spot,
}

impl Light {
    pub fn default_color() -> Vec3 {
        Vec3::one()
    }
}

impl CullingMode {
    pub fn enumerate() -> impl Iterator<Item = Self> {
        [CullingMode::Disabled, CullingMode::BackFace, CullingMode::FrontFace].into_iter()
    }
}

impl std::fmt::Display for CullingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
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