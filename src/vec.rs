use std::ops::{
    Add, AddAssign, Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
    SubAssign,
};

pub type Mat4x4 = Mat<f32, 4, 4>;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mat<T, const M: usize, const N: usize>([[T; N]; M]);

impl<T: Num, const M: usize, const N: usize> Mat<T, M, N> {
    pub fn zero() -> Self {
        Mat([[T::zero(); N]; M])
    }

    pub fn one() -> Self {
        Mat([[T::one(); N]; M])
    }
}

impl<T: Num, const N: usize> Mat<T, N, N> {
    pub fn identity() -> Self {
        let mut ret = Self::zero();
        for i in 0..N {
            ret[(i, i)] = T::one();
        }
        ret
    }
}

impl<T: Num> Mat<T, 4, 4> {
    #[rustfmt::skip]
    pub fn rotation_x(theta: T) -> Self {
        let o = T::one();
        let z = T::zero();
        let cos = theta.cos();
        let sin = theta.sin();
        Mat([[   o,   z,   z,   z],
             [   z, cos,-sin,   z],
             [   z, sin, cos,   z],
             [   z,   z,   z,   o]])
    }

    #[rustfmt::skip]
    pub fn rotation_y(theta: T) -> Self {
        let o = T::one();
        let z = T::zero();
        let cos = theta.cos();
        let sin = theta.sin();
        Mat([[ cos,   z, sin,   z],
             [   z,   o,   z,   z],
             [-sin,   z, cos,   z],
             [   z,   z,   z,   o]])
    }

    #[rustfmt::skip]
    pub fn rotation_z(theta: T) -> Self {
        let o = T::one();
        let z = T::zero();
        let cos = theta.cos();
        let sin = theta.sin();
        Mat([[ cos, -sin,   z,   z],
             [ sin,  cos,   z,   z],
             [   z,    z,   o,   z],
             [   z,    z,   z,   o]])
    }
}

impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Mat<T, M, N> {
    fn from(value: [[T; N]; M]) -> Self {
        Mat(value)
    }
}

pub type Vec<T, const N: usize> = Mat<T, N, 1>;
pub type Vec2 = Vec<f32, 2>;
pub type Vec3 = Vec<f32, 3>;
pub type Vec4 = Vec<f32, 4>;

impl<T: Num, const N: usize> Vec<T, N> {
    pub fn mag_sq(&self) -> T {
        self.0.iter().map(|&[coord]| coord * coord).sum()
    }

    pub fn mag(&self) -> T {
        self.mag_sq().sqrt()
    }

    pub fn normalize(self) -> Self {
        self / self.mag()
    }
}

impl<T: Num> Vec<T, 3> {
    pub fn cross(self, rhs: Self) -> Self {
        Self::from([
            self.y * rhs.z - self.z * rhs.y,
            self.x * rhs.z - self.z * rhs.x,
            self.x * rhs.y - self.y * rhs.x,
        ])
    }

    pub fn to_rotation(self) -> Mat<T, 4, 4> {
        Mat::rotation_x(self.x) * Mat::rotation_y(self.y) * Mat::rotation_z(self.z)
    }

    pub fn to_translation(self) -> Mat<T, 4, 4> {
        let mut ret = Mat::identity();
        ret[(0, 3)] = self.x;
        ret[(1, 3)] = self.y;
        ret[(2, 3)] = self.z;
        ret
    }

    #[rustfmt::skip]
    pub fn to_scale(self) -> Mat<T, 4, 4> {
        let mut ret = Mat::zero();
        ret[(0, 0)] = self.x;
        ret[(1, 1)] = self.y;
        ret[(2, 2)] = self.z;
        ret[(3, 3)] = T::one();
        ret
    }
}

impl<T: Num> Vec<T, 4> {
    #[inline(always)]
    pub fn hom_translate(self, translation: Vec<T, 3>) -> Self {
        translation.to_translation() * self
    }

    #[inline(always)]
    pub fn hom_rotate(self, rotation: Vec<T, 3>) -> Self {
        rotation.to_rotation() * self
    }

    #[inline(always)]
    pub fn hom_scale(self, scale: Vec<T, 3>) -> Self {
        scale.to_scale() * self
    }
}

impl<T, const N: usize> From<[T; N]> for Vec<T, N> {
    fn from(value: [T; N]) -> Self {
        Mat(unsafe { std::mem::transmute_copy::<[T; N], [[T; 1]; N]>(&value) })
    }
}

impl<T, const M: usize, const N: usize> Index<(usize, usize)> for Mat<T, M, N> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.0[i][j]
    }
}

impl<T, const M: usize, const N: usize> IndexMut<(usize, usize)> for Mat<T, M, N> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.0[i][j]
    }
}

impl<T: Num, const M: usize, const N: usize> Add for Mat<T, M, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut ret = self;
        for i in 0..M {
            for j in 0..N {
                ret[(i, j)] += rhs[(i, j)];
            }
        }
        ret
    }
}

impl<T: Num, const M: usize, const N: usize> Sub for Mat<T, M, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut ret = self;
        for i in 0..M {
            for j in 0..N {
                ret[(i, j)] -= rhs[(i, j)];
            }
        }
        ret
    }
}

impl<T: Num, const M: usize, const N: usize> Mul<T> for Mat<T, M, N> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        let mut ret = self;
        for i in 0..M {
            for j in 0..N {
                ret[(i, j)] *= rhs;
            }
        }
        ret
    }
}

macro_rules! impl_mul_lhs {
    ($($ty:ty),+) => {
        $(impl<const M: usize, const N: usize> Mul<Mat<$ty, M, N>> for $ty {
            type Output = Mat<$ty, M, N>;

            fn mul(self, rhs: Mat<$ty, M, N>) -> Mat<$ty, M, N> {
                let mut ret = rhs;
                for i in 0..M {
                    for j in 0..N {
                        ret[(i, j)] *= self;
                    }
                }
                ret
            }
        })+
    };
}

impl_mul_lhs!(f32, f64);

impl<T: Num, const M: usize, const K: usize, const N: usize> Mul<Mat<T, K, N>> for Mat<T, M, K> {
    type Output = Mat<T, M, N>;

    fn mul(self, rhs: Mat<T, K, N>) -> Self::Output {
        let mut ret = Mat::zero();
        for i in 0..M {
            for j in 0..N {
                for k in 0..K {
                    ret[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        ret
    }
}

impl<T: Num, const M: usize, const N: usize> Div<T> for Mat<T, M, N> {
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        let mut ret = self;
        for i in 0..M {
            for j in 0..N {
                ret[(i, j)] /= rhs;
            }
        }
        ret
    }
}

impl<T: Num> Deref for Vec<T, 3> {
    type Target = XYZ<T>;

    fn deref(&self) -> &XYZ<T> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T: Num> DerefMut for Vec<T, 3> {
    fn deref_mut(&mut self) -> &mut XYZ<T> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T: Num> Deref for Vec<T, 4> {
    type Target = XYZW<T>;

    fn deref(&self) -> &XYZW<T> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T: Num> DerefMut for Vec<T, 4> {
    fn deref_mut(&mut self) -> &mut XYZW<T> {
        unsafe { std::mem::transmute(self) }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XYZ<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy> XYZ<T> {
    pub fn xy(&self) -> Vec<T, 2> {
        Vec::from([self.x, self.y])
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XYZW<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T: Copy> XYZW<T> {
    pub fn xy(&self) -> Vec<T, 2> {
        Vec::from([self.x, self.y])
    }

    pub fn xyz(&self) -> Vec<T, 3> {
        Vec::from([self.x, self.y, self.z])
    }
}

pub trait Num:
    Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
    + std::iter::Sum
{
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

impl Num for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }
}

impl Num for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }
}
