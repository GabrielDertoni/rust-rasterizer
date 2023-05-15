use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
    SubAssign,
};
use std::simd::{LaneCount, Simd, SimdElement, StdFloat, SimdFloat, SimdOrd, SupportedLaneCount};

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
        use std::mem::{MaybeUninit, transmute_copy};

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
    pub fn zero() -> Self { Mat([[T::zero(); N]; M]) }

    #[inline(always)]
    pub fn one()  -> Self { Mat([[T::one() ; N]; M]) }

    #[inline(always)]
    pub fn repeat(el: T) -> Self
    where
        T: Copy,
    {
        Mat([[el; N]; M])
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

    pub fn slice<const START: usize, const COUNT: usize>(&self) -> &Vec<T, COUNT> {
        if START + COUNT > N { panic!("index out of bounds"); }
        let ptr = self.0.as_ptr();
        unsafe { std::mem::transmute(&*ptr.add(START)) }
    }

    pub fn slice_mut<const START: usize, const COUNT: usize>(&mut self) -> &mut Vec<T, COUNT> {
        if START + COUNT > N { panic!("index out of bounds"); }
        let ptr = self.0.as_mut_ptr();
        unsafe { std::mem::transmute(&mut *ptr.add(START)) }
    }

    pub fn inplace(arr: &[T; N]) -> &Self {
        unsafe { std::mem::transmute(arr) }
    }

    pub fn inplace_mut(arr: &mut [T; N]) -> &mut Self {
        unsafe { std::mem::transmute(arr) }
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

impl_mul_lhs!(f32, f64, i32);

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
        ($self:expr; $field:ident) => { $self.$field };
        ($self:expr; $e:expr) => { $e };
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

        fn deref(&self) -> &XY<T>  {
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
        impl_num_float!($ty);
        impl_num_float!($($rest)*);
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

impl_num_int!(i32);

macro_rules! impl_num_simd_int {
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
            fn zero() -> Self { Simd::splat(0) }

            #[inline(always)]
            fn one()  -> Self { Simd::splat(1)  }

            #[inline(always)]
            fn min(self, rhs: Self)  -> Self { self.simd_min(rhs)  }

            #[inline(always)]
            fn max(self, rhs: Self)  -> Self { self.simd_max(rhs)  }

            #[inline(always)]
            fn clamp(self, min: Self, max: Self)  -> Self { self.simd_clamp(min, max)  }
        }
    };
}

impl_num_simd_int!(i32);

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
            fn zero() -> Self { Simd::splat(0.0) }

            #[inline(always)]
            fn one()  -> Self { Simd::splat(1.0)  }

            #[inline(always)]
            fn min(self, rhs: Self)  -> Self { self.simd_min(rhs)  }

            #[inline(always)]
            fn max(self, rhs: Self)  -> Self { self.simd_max(rhs)  }

            #[inline(always)]
            fn clamp(self, min: Self, max: Self)  -> Self { self.simd_clamp(min, max)  }
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
