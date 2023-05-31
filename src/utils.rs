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
        self.pitch = self.pitch.clamp(-90., 90.);
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
