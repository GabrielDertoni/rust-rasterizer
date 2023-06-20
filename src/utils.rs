use std::time::{Duration, Instant};

use crate::vec::{Mat3x3, Mat4x4, Vec3};

#[macro_export]
macro_rules! unroll_array {
    ($v:ident : $ty:ty = $($values:literal),* => $e:expr) => {
        [$({
            const $v: $ty = $values;
            $e
        }),*]
    }
}

#[macro_export]
macro_rules! unroll {
    (for $v:ident in [$($values:literal),*] $block:tt) => {
        $(
            let $v = $values;
            $block;
        )*
    }
}

/// FPS counter that averages N samples in order to calculate framerate.
#[derive(Clone, Default)]
pub struct FpsCounter {
    measured_times: Vec<std::time::Duration>,
    max_samples: usize,
}

impl FpsCounter {
    pub fn new() -> Self {
        let max_samples = 60;
        FpsCounter {
            measured_times: Vec::with_capacity(max_samples),
            max_samples,
        }
    }

    pub fn measure<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let ret = f();
        let time = start.elapsed();
        self.record_measurement(time);
        ret
    }

    pub fn record_measurement(&mut self, measurement: Duration) {
        if self.measured_times.len() >= self.max_samples {
            self.measured_times.rotate_left(1);
            *self.measured_times.last_mut().unwrap() = measurement;
        } else {
            self.measured_times.push(measurement);
        }
    }

    pub fn measure_scoped<'a>(&'a mut self) -> MeasureScopeGuard<'a> {
        MeasureScopeGuard {
            start: Instant::now(),
            counter: self,
        }
    }

    pub fn mean_time(&self) -> Duration {
        if self.measured_times.len() == 0 {
            return Duration::ZERO;
        }
        self.measured_times.iter().sum::<Duration>() / self.measured_times.len() as u32
    }

    pub fn mean_fps(&self) -> f32 {
        1. / self.mean_time().as_secs_f32()
    }

    pub fn worst_mean_time(&mut self, fraction: f32) -> Duration {
        assert!(fraction > 0. && fraction <= 1., "fraction must be in range [0, 1]");
        self.measured_times.sort_unstable();
        let len = self.measured_times.len() / (1. / fraction) as usize;
        let len = len.clamp(1, self.measured_times.len());
        self.measured_times[..len].iter().sum::<Duration>() / len as u32
    }

    pub fn worst_fps(&mut self, fraction: f32) -> f32 {
        1. / self.worst_mean_time(fraction).as_secs_f32()
    }
}

pub struct MeasureScopeGuard<'a> {
    start: Instant,
    counter: &'a mut FpsCounter,
}

impl<'a> Drop for MeasureScopeGuard<'a> {
    fn drop(&mut self) {
        self.counter.record_measurement(self.start.elapsed());
    }
}

pub struct FpvCamera {
    pub position: Vec3,
    pub up: Vec3,
    /// Pitch measured in degrees
    pub pitch: f32,
    /// Yaw measured in degrees
    pub yaw: f32,
    pub sensitivity: f32,
    pub speed: f32,
    pub fovy: f32,
    pub ratio: f32,
}

impl FpvCamera {
    /// Rotate the camera by some delta in pitch and yaw, measured in degrees.
    pub fn rotate_delta(&mut self, delta_pitch: f32, delta_yaw: f32) {
        self.pitch -= delta_pitch * self.sensitivity;
        self.pitch = self.pitch.clamp(0., 180.);
        self.yaw -= delta_yaw * self.sensitivity;
    }

    /// Move the camera by `axis`. The vector `axis` has coordinates `x` for sideways motion
    /// (positive goes to the right), `y` for going forward and backwards (positive goes forward)
    /// and `z` for going up and down (positive goes up).
    pub fn move_delta(&mut self, axis: Vec3) {
        self.position += self.change_of_basis() * (self.speed * axis);
    }

    pub fn right(&self) -> Vec3 {
        self.front().cross(self.up).normalized()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.front())
    }

    pub fn front(&self) -> Vec3 {
        /*
        let yaw = self.yaw.to_radians();
        let pitch = -self.pitch.to_radians();
        Vec3::from([
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        ])
        */
        let yaw = -self.yaw.to_radians();
        let pitch = self.pitch.to_radians();
        Vec3::from([
            pitch.sin() * yaw.sin(),
            pitch.sin() * yaw.cos(),
            -pitch.cos(),
        ])
    }

    pub fn view_matrix(&self) -> Mat4x4 {
        Mat4x4::look_at(self.position, self.position + self.front(), self.up)
    }

    pub fn projection_matrix(&self, near: f32, far: f32) -> Mat4x4 {
        Mat4x4::perspective(self.ratio, self.fovy, near, far)
    }

    pub fn transform_matrix(&self, near: f32, far: f32) -> Mat4x4 {
        self.projection_matrix(near, far) * self.view_matrix()
    }

    // Change of basis matrix that allows changing from a vector in "camera" space, to world space.
    // Camera space has x coordinates going to the right, y coordinates going forward and z coordinates
    // going up.
    fn change_of_basis(&self) -> Mat3x3 {
        let front = self.front();
        let right = front.cross(self.up).normalized();
        let up = right.cross(front);
        Mat3x3::from([
            [right.x, front.x, up.x],
            [right.y, front.y, up.y],
            [right.z, front.z, up.z],
        ])
    }
}

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fovy: f32,
    pub ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn from_blender(
        position: Vec3,
        rotation_deg: Vec3,
        fovy: f32,
        ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let up = Vec3::from([0., 0., 1.]);
        let rotation = rotation_deg.map(|el| el.to_radians());
        Camera {
            position,
            target: position + (rotation.to_rotation() * (-up).to_hom()).xyz(),
            up,
            fovy,
            ratio,
            near,
            far,
        }
    }

    pub fn view_matrix(&self) -> Mat4x4 {
        Mat4x4::look_at(self.position, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> Mat4x4 {
        Mat4x4::perspective(self.ratio, self.fovy, self.near, self.far)
    }

    pub fn transform_matrix(&self) -> Mat4x4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn right(&self) -> Vec3 {
        self.front().cross(self.up).normalized()
    }

    pub fn up(&self) -> Vec3 {
        self.right().cross(self.front())
    }

    pub fn front(&self) -> Vec3 {
        (self.target - self.position).normalized()
    }
}
