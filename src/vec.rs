use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
use std::simd::{LaneCount, Simd, SimdElement, SimdFloat, SimdOrd, StdFloat, SupportedLaneCount};

pub type Mat4x4 = Mat<f32, 4, 4>;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mat<T, const M: usize, const N: usize>([[T; N]; M]);

impl<T: Copy, const M: usize, const N: usize> Mat<T, M, N> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Mat<U, M, N> {
        Mat(self.0.map(|row| row.map(&mut f)))
    }

    /// # Panics
    ///
    /// If `f` panics. Note that in that case, some `R` values that were initialized might not have their destructors called.
    #[inline(always)]
    pub fn zip_with<U: Copy, R>(
        self,
        rhs: Mat<U, M, N>,
        mut f: impl FnMut(T, U) -> R,
    ) -> Mat<R, M, N> {
        use std::mem::{transmute_copy, MaybeUninit};

        // SAFETY: Transposing the `MaybeUninit` to the inner type is safe, since we still can't access
        // any real data that might be uninitialized.
        let mut arr = unsafe { MaybeUninit::<[[MaybeUninit<R>; N]; M]>::uninit().assume_init() };
        for i in 0..M {
            for j in 0..N {
                arr[i][j].write(f(self.0[i][j], rhs.0[i][j]));
            }
        }

        // SAFETY: Equivalent to `assume_init`, we have initialized every element.
        unsafe { Mat(transmute_copy(&arr)) }
    }

    pub fn splat<const LANES: usize>(self) -> Mat<Simd<T, LANES>, M, N>
    where
        T: SimdElement,
        LaneCount<LANES>: SupportedLaneCount,
    {
        self.map(|el| Simd::splat(el))
    }

    pub fn transpose(self) -> Mat<T, N, M> {
        use std::array::from_fn;
        Mat::from(from_fn(|j| from_fn(|i| self[(i, j)])))
    }
}

impl<T: Copy + Ord, const M: usize, const N: usize> Mat<T, M, N> {
    pub fn min(self, rhs: Self) -> Self {
        self.zip_with(rhs, |lhs, rhs| std::cmp::min(lhs, rhs))
    }

    pub fn max(self, rhs: Self) -> Self {
        self.zip_with(rhs, |lhs, rhs| std::cmp::max(lhs, rhs))
    }
}

impl<T: Num, const M: usize, const N: usize> Mat<T, M, N> {
    #[inline(always)]
    pub fn zero() -> Self {
        Mat([[T::zero(); N]; M])
    }

    #[inline(always)]
    pub fn one() -> Self {
        Mat([[T::one(); N]; M])
    }

    #[inline(always)]
    pub fn repeat(el: T) -> Self
    where
        T: Copy,
    {
        Mat([[el; N]; M])
    }

    #[inline(always)]
    pub fn element_mul(self, rhs: Self) -> Self {
        self.zip_with(rhs, |lhs, rhs| lhs * rhs)
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

impl<const M: usize, const N: usize> Mat<f32, M, N> {
    pub fn to_i32(self) -> Mat<i32, M, N> {
        self.map(|el| el as i32)
    }

    pub fn to_u8(self) -> Mat<u8, M, N> {
        self.map(|el| el as u8)
    }
}

impl<const M: usize, const N: usize> Mat<i32, M, N> {
    pub fn to_f32(self) -> Mat<f32, M, N> {
        self.map(|el| el as f32)
    }
}

impl<T: Num> Mat<T, 4, 4> {
    pub fn translate(self, translation: Vec<T, 3>) -> Self {
        translation.to_translation() * self
    }

    pub fn scale(self, factor: Vec<T, 3>) -> Self {
        factor.to_scale() * self
    }
}

impl<T: Float> Mat<T, 4, 4> {
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
        Mat([[ cos,-sin,   z,   z],
             [ sin, cos,   z,   z],
             [   z,   z,   o,   z],
             [   z,   z,   z,   o]])
    }

    pub fn rotate(self, euler_angles: Vec<T, 3>) -> Self {
        euler_angles.to_rotation() * self
    }

    pub fn determinant(&self) -> T {
        self.0[0][0] * (self.0[1][1] * self.0[2][2] - self.0[2][1] * self.0[1][2])
            - self.0[1][0] * (self.0[0][1] * self.0[2][2] - self.0[2][1] * self.0[0][2])
            - (self.0[2][0] * self.0[0][1] * self.0[1][2] - self.0[1][1] * self.0[0][2])
    }

    pub fn inverse(&self) -> Self {
        let det = self.determinant();
        self.adjugate() * (T::one() / det)
    }

    // source: https://docs.rs/ultraviolet/0.9.1/src/ultraviolet/mat.rs.html#1385-1443
    pub fn adjugate(&self) -> Self {
        let [m00, m01, m02, m03] = self.0[0];
        let [m10, m11, m12, m13] = self.0[1];
        let [m20, m21, m22, m23] = self.0[2];
        let [m30, m31, m32, m33] = self.0[3];

        let coef00 = (m22 * m33) - (m32 * m23);
        let coef02 = (m12 * m33) - (m32 * m13);
        let coef03 = (m12 * m23) - (m22 * m13);

        let coef04 = (m21 * m33) - (m31 * m23);
        let coef06 = (m11 * m33) - (m31 * m13);
        let coef07 = (m11 * m23) - (m21 * m13);

        let coef08 = (m21 * m32) - (m31 * m22);
        let coef10 = (m11 * m32) - (m31 * m12);
        let coef11 = (m11 * m22) - (m21 * m12);

        let coef12 = (m20 * m33) - (m30 * m23);
        let coef14 = (m10 * m33) - (m30 * m13);
        let coef15 = (m10 * m23) - (m20 * m13);

        let coef16 = (m20 * m32) - (m30 * m22);
        let coef18 = (m10 * m32) - (m30 * m12);
        let coef19 = (m10 * m22) - (m20 * m12);

        let coef20 = (m20 * m31) - (m30 * m21);
        let coef22 = (m10 * m31) - (m30 * m11);
        let coef23 = (m10 * m21) - (m20 * m11);

        let fac0 = Vec::from([coef00, coef00, coef02, coef03]);
        let fac1 = Vec::from([coef04, coef04, coef06, coef07]);
        let fac2 = Vec::from([coef08, coef08, coef10, coef11]);
        let fac3 = Vec::from([coef12, coef12, coef14, coef15]);
        let fac4 = Vec::from([coef16, coef16, coef18, coef19]);
        let fac5 = Vec::from([coef20, coef20, coef22, coef23]);

        let vec0 = Vec::from([m10, m00, m00, m00]);
        let vec1 = Vec::from([m11, m01, m01, m01]);
        let vec2 = Vec::from([m12, m02, m02, m02]);
        let vec3 = Vec::from([m13, m03, m03, m03]);

        let inv0 = vec1.element_mul(fac0) - (vec2.element_mul(fac1)) + vec3.element_mul(fac2);
        let inv1 = vec0.element_mul(fac0) - (vec2.element_mul(fac3)) + vec3.element_mul(fac4);
        let inv2 = vec0.element_mul(fac1) - (vec1.element_mul(fac3)) + vec3.element_mul(fac5);
        let inv3 = vec0.element_mul(fac2) - (vec1.element_mul(fac4)) + vec2.element_mul(fac5);

        let o = T::one();
        let sign_a = Vec::from([o, -o, o, -o]);
        let sign_b = Vec::from([-o, o, -o, o]);

        Self::from([
            inv0.element_mul(sign_a).to_array(),
            inv1.element_mul(sign_b).to_array(),
            inv2.element_mul(sign_a).to_array(),
            inv3.element_mul(sign_b).to_array(),
        ])
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

pub type Vec2i = Vec<i32, 2>;
pub type Vec3i = Vec<i32, 3>;
pub type Vec4i = Vec<i32, 4>;

pub type Vec2x4 = Vec<Simd<f32, 4>, 2>;
pub type Vec3x4 = Vec<Simd<f32, 4>, 3>;
pub type Vec4x4 = Vec<Simd<f32, 4>, 4>;

pub type IVec2x4 = Vec<Simd<i32, 4>, 2>;
pub type IVec3x4 = Vec<Simd<i32, 4>, 3>;
pub type IVec4x4 = Vec<Simd<i32, 4>, 4>;

impl<T, const N: usize> Vec<T, N> {
    pub fn to_array(&self) -> [T; N] {
        unsafe { std::mem::transmute_copy(&self.0) }
    }

    pub fn as_array(&self) -> &[T; N] {
        unsafe { std::mem::transmute(self) }
    }

    pub fn slice<const START: usize, const COUNT: usize>(&self) -> &Vec<T, COUNT> {
        if START + COUNT > N {
            panic!("index out of bounds");
        }
        let ptr = self.0.as_ptr();
        unsafe { std::mem::transmute(&*ptr.add(START)) }
    }

    pub fn slice_mut<const START: usize, const COUNT: usize>(&mut self) -> &mut Vec<T, COUNT> {
        if START + COUNT > N {
            panic!("index out of bounds");
        }
        let ptr = self.0.as_mut_ptr();
        unsafe { std::mem::transmute(&mut *ptr.add(START)) }
    }

    pub fn inplace(arr: &[T; N]) -> &Self {
        unsafe { std::mem::transmute(arr) }
    }

    pub fn inplace_mut(arr: &mut [T; N]) -> &mut Self {
        unsafe { std::mem::transmute(arr) }
    }

    pub fn vtranspose(&self) -> Mat<T, 1, N> {
        unsafe { std::mem::transmute_copy(self) }
    }
}

impl<T, const M: usize, const N: usize> Vec<[T; M], N> {
    pub fn as_mat(&self) -> &Mat<T, N, M> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T: Num, const N: usize> Vec<T, N> {
    pub fn mag_sq(&self) -> T {
        self.0.iter().map(|&[coord]| coord * coord).sum()
    }

    pub fn dot(&self, rhs: Self) -> T {
        let mut ret = T::zero();
        for i in 0..N {
            ret += self.0[i][0] * rhs.0[i][0];
        }
        ret
    }
}

impl<T: Float, const N: usize> Vec<T, N> {
    pub fn mag(&self) -> T {
        self.mag_sq().sqrt()
    }

    pub fn normalized(self) -> Self {
        self / self.mag()
    }

    pub fn normalize(&mut self) {
        *self /= self.mag()
    }
}

impl<T: Copy> Vec<T, 3> {
    #[inline(always)]
    pub fn map_3<U>(self, mut f: impl FnMut(T) -> U) -> Vec<U, 3> {
        Vec::from([f(self.x), f(self.y), f(self.z)])
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

    pub fn unit_y() -> Self {
        Self::from([T::zero(), T::one(), T::zero()])
    }

    pub fn to_hom(self) -> Vec<T, 4> {
        [self.x, self.y, self.z, T::one()].into()
    }
}

impl<T: Float> Vec<T, 3> {
    pub fn to_rotation(self) -> Mat<T, 4, 4> {
        Mat::rotation_x(self.x) * Mat::rotation_y(self.y) * Mat::rotation_z(self.z)
    }

    // source: https://docs.rs/ultraviolet/latest/src/ultraviolet/interp.rs.html#215-219
    pub fn slerp(self, target: Self, t: T) -> Self {
        let dot = self.dot(target).clamp(-T::one(), T::one());
        let theta = dot.acos() * t;

        let v = (target - self * dot).normalized();
        self * theta.cos() + v * theta.sin()
    }
}

impl<T: Copy> Vec<T, 4> {
    #[inline(always)]
    pub fn map_4<U>(self, mut f: impl FnMut(T) -> U) -> Vec<U, 4> {
        Vec::from([f(self.x), f(self.y), f(self.z), f(self.w)])
    }

    #[inline(always)]
    pub fn zip_with_4<U: Copy, R>(self, rhs: Vec<U, 4>, mut f: impl FnMut(T, U) -> R) -> Vec<R, 4> {
        Vec::from([
            f(self.x, rhs.x),
            f(self.y, rhs.y),
            f(self.z, rhs.z),
            f(self.w, rhs.w),
        ])
    }
}

impl<T: Num> Vec<T, 4> {
    #[inline(always)]
    pub fn hom_translate(self, translation: Vec<T, 3>) -> Self {
        translation.to_translation() * self
    }

    #[inline(always)]
    pub fn hom_scale(self, scale: Vec<T, 3>) -> Self {
        scale.to_scale() * self
    }
}

impl<T: Float> Vec<T, 4> {
    #[inline(always)]
    pub fn hom_rotate(self, rotation: Vec<T, 3>) -> Self {
        rotation.to_rotation() * self
    }
}

impl<T: SimdElement> Vec<Simd<T, 4>, 4> {
    // source: https://fgiesen.wordpress.com/2013/07/09/simd-transposes-1/
    pub fn simd_transpose_4(self) -> Self {
        // Initial (SoA):
        //
        //    X = { x0, x1, x2, x3 }
        //    Y = { y0, y1, y2, y3 }
        //    Z = { z0, z1, z2, z3 }
        //    W = { w0, w1, w2, w3 }

        let (a0, a2) = self.x.interleave(self.z);
        let (a1, a3) = self.y.interleave(self.w);

        //   a0 = { x0, z0, x1, z1 }
        //   a1 = { y0, w0, y1, w1 }
        //   a2 = { x2, z2, x3, z3 }
        //   a3 = { y2, w2, y3, w3 }

        let (p0, p1) = a0.interleave(a1);
        let (p2, p3) = a2.interleave(a3);

        // Final (AoS):
        //
        //   p0 = { x0, y0, z0, w0 }
        //   p1 = { x1, y1, z1, w1 }
        //   p2 = { x2, y2, z2, w2 }
        //   p3 = { x3, y3, z3, w3 }

        Vec::from([p0, p1, p2, p3])
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

    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<T: Num, const M: usize, const N: usize> AddAssign for Mat<T, M, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] += rhs[(i, j)];
            }
        }
    }
}

impl<T: Num, const M: usize, const N: usize> Sub for Mat<T, M, N> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<T: Num, const M: usize, const N: usize> SubAssign for Mat<T, M, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] -= rhs[(i, j)];
            }
        }
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
        $(
            impl<const LANES: usize, const M: usize, const N: usize> Mul<Mat<Simd<$ty, LANES>, M, N>> for Simd<$ty, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = Mat<Simd<$ty, LANES>, M, N>;

                fn mul(self, rhs: Mat<Simd<$ty, LANES>, M, N>) -> Mat<Simd<$ty, LANES>, M, N> {
                    let mut ret = rhs;
                    for i in 0..M {
                        for j in 0..N {
                            ret[(i, j)] *= self;
                        }
                    }
                    ret
                }
            }
        )+
    };
}

impl_mul_lhs!(f32, f64, i32, i64);

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

    fn div(mut self, rhs: T) -> Self {
        self /= rhs;
        self
    }
}

impl<T: Num, const M: usize, const N: usize> DivAssign<T> for Mat<T, M, N> {
    fn div_assign(&mut self, rhs: T) {
        for i in 0..M {
            for j in 0..N {
                self[(i, j)] /= rhs;
            }
        }
    }
}

mod swizzling {
    use super::Vec;

    use std::ops::{Deref, DerefMut};

    #[macro_export]
    macro_rules! __vec_ident_or_expr {
        ($self:expr; $field:ident) => {
            $self.$field
        };
        ($self:expr; $e:expr) => {
            $e
        };
    }

    #[macro_export]
    macro_rules! swizzle {
        ($self:expr, [$($field:tt),*]) => {
            $crate::vec::Vec::from([$($crate::__vec_ident_or_expr!($self; $field)),*])
        };
    }

    impl<T> Deref for Vec<T, 1> {
        type Target = X<T>;

        #[inline(always)]
        fn deref(&self) -> &X<T> {
            unsafe { std::mem::transmute(self) }
        }
    }

    impl<T> DerefMut for Vec<T, 1> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut X<T> {
            unsafe { std::mem::transmute(self) }
        }
    }

    impl<T> Deref for Vec<T, 2> {
        type Target = XY<T>;

        #[inline(always)]
        fn deref(&self) -> &XY<T> {
            unsafe { std::mem::transmute(self) }
        }
    }

    impl<T> DerefMut for Vec<T, 2> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut XY<T> {
            unsafe { std::mem::transmute(self) }
        }
    }

    impl<T> Deref for Vec<T, 3> {
        type Target = XYZ<T>;

        #[inline(always)]
        fn deref(&self) -> &XYZ<T> {
            unsafe { std::mem::transmute(self) }
        }
    }

    impl<T> DerefMut for Vec<T, 3> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut XYZ<T> {
            unsafe { std::mem::transmute(self) }
        }
    }

    impl<T> Deref for Vec<T, 4> {
        type Target = XYZW<T>;

        #[inline(always)]
        fn deref(&self) -> &XYZW<T> {
            unsafe { std::mem::transmute(self) }
        }
    }

    impl<T> DerefMut for Vec<T, 4> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut XYZW<T> {
            unsafe { std::mem::transmute(self) }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct X<T> {
        pub x: T,
    }

    impl<T: Copy> X<T> {
        #[inline(always)]
        pub fn x(&self) -> Vec<T, 1> {
            Vec::from([self.x])
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct XY<T> {
        _x: X<T>,
        pub y: T,
    }

    impl<T: Copy> XY<T> {
        #[inline(always)]
        pub fn xy(&self) -> Vec<T, 2> {
            Vec::from([self.x, self.y])
        }

        #[inline(always)]
        pub fn yx(&self) -> Vec<T, 2> {
            Vec::from([self.y, self.x])
        }
    }

    impl<T> Deref for XY<T> {
        type Target = X<T>;

        fn deref(&self) -> &X<T> {
            &self._x
        }
    }

    impl<T> DerefMut for XY<T> {
        fn deref_mut(&mut self) -> &mut X<T> {
            &mut self._x
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct XYZ<T> {
        _xy: XY<T>,
        pub z: T,
    }

    impl<T: Copy> XYZ<T> {
        #[inline(always)]
        pub fn xyz(&self) -> Vec<T, 3> {
            Vec::from([self.x, self.y, self.z])
        }

        #[inline(always)]
        pub fn xzy(&self) -> Vec<T, 3> {
            Vec::from([self.x, self.z, self.y])
        }

        #[inline(always)]
        pub fn yxz(&self) -> Vec<T, 3> {
            Vec::from([self.y, self.x, self.z])
        }

        #[inline(always)]
        pub fn yzx(&self) -> Vec<T, 3> {
            Vec::from([self.y, self.z, self.x])
        }

        #[inline(always)]
        pub fn zxy(&self) -> Vec<T, 3> {
            Vec::from([self.z, self.x, self.y])
        }

        #[inline(always)]
        pub fn zyx(&self) -> Vec<T, 3> {
            Vec::from([self.z, self.y, self.x])
        }
    }

    impl<T> Deref for XYZ<T> {
        type Target = XY<T>;

        fn deref(&self) -> &XY<T> {
            &self._xy
        }
    }

    impl<T> DerefMut for XYZ<T> {
        fn deref_mut(&mut self) -> &mut XY<T> {
            &mut self._xy
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct XYZW<T> {
        _xyz: XYZ<T>,
        pub w: T,
    }

    impl<T> Deref for XYZW<T> {
        type Target = XYZ<T>;

        #[inline(always)]
        fn deref(&self) -> &XYZ<T> {
            &self._xyz
        }
    }

    impl<T> DerefMut for XYZW<T> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut XYZ<T> {
            &mut self._xyz
        }
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
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;

    fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

pub trait Float: Num {
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn acos(self) -> Self;
}

macro_rules! impl_num_float {
    () => {};
    ($ty:ty, $($rest:tt)*) => {
        impl_num_float!($ty);
        impl_num_float!($($rest)*);
    };
    ($ty:ty) => {
        impl Num for $ty {
            #[inline(always)]
            fn zero() -> $ty { 0.0 }

            #[inline(always)]
            fn one() -> $ty { 1.0 }

            #[inline(always)]
            fn min(self, rhs: Self) -> $ty { self.min(rhs) }

            #[inline(always)]
            fn max(self, rhs: Self) -> $ty { self.max(rhs) }

            #[inline(always)]
            fn clamp(self, min: Self, max: Self) -> $ty { self.clamp(min, max) }
        }

        impl Float for $ty {
            #[inline(always)]
            fn sqrt(self) -> $ty { self.sqrt() }

            #[inline(always)]
            fn sin(self)  -> $ty { self.sin()  }

            #[inline(always)]
            fn cos(self)  -> $ty { self.cos()  }

            #[inline(always)]
            fn acos(self) -> $ty { self.acos() }
        }
    };
}

impl_num_float!(f32, f64);

macro_rules! impl_num_int {
    () => {};
    ($ty:ty, $($rest:tt)*) => {
        impl_num_int!($ty);
        impl_num_int!($($rest)*);
    };
    ($ty:ty) => {
        impl Num for $ty {
            #[inline(always)]
            fn zero() -> $ty { 0 }

            #[inline(always)]
            fn one() -> $ty { 1 }

            #[inline(always)]
            fn min(self, rhs: Self) -> $ty { Ord::min(self, rhs) }

            #[inline(always)]
            fn max(self, rhs: Self) -> $ty { Ord::max(self, rhs) }

            #[inline(always)]
            fn clamp(self, min: Self, max: Self) -> $ty { Ord::clamp(self, min, max) }
        }
    };
}

impl_num_int!(i32, i64);

macro_rules! impl_num_simd_int {
    () => {};
    ($ty:ty, $($rest:tt)*) => {
        impl_num_simd_int!($ty);
        impl_num_simd_int!($($rest)*);
    };
    ($ty:ty) => {
        impl<const N: usize> Num for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline(always)]
            fn zero() -> Self {
                Simd::splat(0)
            }

            #[inline(always)]
            fn one() -> Self {
                Simd::splat(1)
            }

            #[inline(always)]
            fn min(self, rhs: Self) -> Self {
                self.simd_min(rhs)
            }

            #[inline(always)]
            fn max(self, rhs: Self) -> Self {
                self.simd_max(rhs)
            }

            #[inline(always)]
            fn clamp(self, min: Self, max: Self) -> Self {
                self.simd_clamp(min, max)
            }
        }
    };
}

impl_num_simd_int!(i32, i64);

macro_rules! impl_num_float_simd {
    () => {};
    ($ty:ty, $($rest:tt)*) => {
        impl_num_float_simd!($ty);
    };
    ($ty:ty) => {
        impl<const N: usize> Num for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline(always)]
            fn zero() -> Self {
                Simd::splat(0.0)
            }

            #[inline(always)]
            fn one() -> Self {
                Simd::splat(1.0)
            }

            #[inline(always)]
            fn min(self, rhs: Self) -> Self {
                self.simd_min(rhs)
            }

            #[inline(always)]
            fn max(self, rhs: Self) -> Self {
                self.simd_max(rhs)
            }

            #[inline(always)]
            fn clamp(self, min: Self, max: Self) -> Self {
                self.simd_clamp(min, max)
            }
        }

        impl<const N: usize> Float for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline(always)]
            fn sqrt(self) -> Self {
                StdFloat::sqrt(self)
            }

            fn sin(mut self) -> Self {
                for el in self.as_mut_array() {
                    *el = el.sin();
                }
                self
            }

            fn cos(mut self) -> Self {
                for el in self.as_mut_array() {
                    *el = el.cos();
                }
                self
            }

            fn acos(mut self) -> Self {
                for el in self.as_mut_array() {
                    *el = el.acos();
                }
                self
            }
        }
    };
}

impl_num_float_simd!(f32, f64);
