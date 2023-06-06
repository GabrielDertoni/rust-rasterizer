use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
    simd::{LaneCount, Simd, SupportedLaneCount},
};

pub struct Texture<T, S, Width: Dim = SomeNat, Height: Dim = SomeNat, I = RowMajor> {
    width: Width,
    height: Height,
    storage: S,
    indexer: I,
    _marker: PhantomData<T>,
}

pub trait Storage {
    type Elem;

    fn ptr(&self) -> *const Self::Elem;
    fn borrow<'a>(&'a self) -> BorrowedStorage<'a, Self::Elem>;
}

pub trait StorageMut: Storage {
    fn ptr_mut(&mut self) -> *mut Self::Elem;
    fn borrow_mut<'a>(&'a mut self) -> BorrowedMutStorage<'a, Self::Elem>;
}

#[derive(Clone, Copy)]
pub struct BorrowedStorage<'a, T> {
    ptr: *const T,
    _marker: PhantomData<&'a [T]>,
}

impl<'s, T> Storage for BorrowedStorage<'s, T> {
    type Elem = T;

    fn ptr(&self) -> *const T {
        self.ptr
    }

    fn borrow<'a>(&'a self) -> BorrowedStorage<'a, Self::Elem> {
        BorrowedStorage {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

pub struct BorrowedMutStorage<'a, T> {
    ptr: *mut T,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'s, T> Storage for BorrowedMutStorage<'s, T> {
    type Elem = T;

    fn ptr(&self) -> *const T {
        self.ptr
    }

    fn borrow<'a>(&'a self) -> BorrowedStorage<'a, Self::Elem> {
        BorrowedStorage {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<'s, T> StorageMut for BorrowedMutStorage<'s, T> {
    fn ptr_mut(&mut self) -> *mut T {
        self.ptr
    }

    fn borrow_mut<'a>(&'a mut self) -> BorrowedMutStorage<'a, Self::Elem> {
        BorrowedMutStorage {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

// TODO: This will store the length of the allocation, which isn't necessary since `MatrixSlice` already stores that
#[derive(Clone)]
pub struct OwnedStorage<T>(Box<[T]>);

impl<T> Storage for OwnedStorage<T> {
    type Elem = T;

    fn ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    fn borrow<'a>(&'a self) -> BorrowedStorage<'a, Self::Elem> {
        BorrowedStorage {
            ptr: self.0.as_ptr(),
            _marker: PhantomData,
        }
    }
}

impl<T> StorageMut for OwnedStorage<T> {
    fn ptr_mut(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    fn borrow_mut<'a>(&'a mut self) -> BorrowedMutStorage<'a, Self::Elem> {
        BorrowedMutStorage {
            ptr: self.0.as_mut_ptr(),
            _marker: PhantomData,
        }
    }
}

pub trait IndexLayout: Copy {
    fn from_size(width: usize, height: usize) -> Self;

    fn get_index(&self, x: usize, y: usize) -> usize;

    fn get_index_simd<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
    ) -> Simd<usize, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount;
}

#[derive(Clone, Copy)]
pub struct RowMajor<Stride: Dim = SomeNat> {
    stride: Stride,
}

impl<Stride: Dim> IndexLayout for RowMajor<Stride> {
    fn from_size(width: usize, _height: usize) -> Self {
        RowMajor {
            stride: Stride::from_usize(width),
        }
    }

    fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.stride.to_usize() + x
    }

    fn get_index_simd<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
    ) -> Simd<usize, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        y * Simd::splat(self.stride.to_usize()) + x
    }
}

#[derive(Clone, Copy)]
pub struct ColumnMajor<Stride: Dim = SomeNat> {
    stride: Stride,
}

impl<Stride: Dim> IndexLayout for ColumnMajor<Stride> {
    fn from_size(_width: usize, height: usize) -> Self {
        ColumnMajor {
            stride: Stride::from_usize(height),
        }
    }

    fn get_index(&self, x: usize, y: usize) -> usize {
        x * self.stride.to_usize() + y
    }

    fn get_index_simd<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
    ) -> Simd<usize, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        x * Simd::splat(self.stride.to_usize()) + y
    }
}

#[derive(Clone, Copy)]
pub struct Tiled<
    const TILE_X: usize,
    const TILE_Y: usize,
    Inner = RowMajor<KnownNat<TILE_X>>,
    Stride: Dim = SomeNat,
> {
    // The number of elements to skip in order to get to the first element of the tile below
    stride: Stride,
    inner: Inner,
}

impl<const TILE_X: usize, const TILE_Y: usize, Inner: IndexLayout, Stride: Dim> IndexLayout
    for Tiled<TILE_X, TILE_Y, Inner, Stride>
{
    fn from_size(width: usize, height: usize) -> Self {
        assert_eq!(width % TILE_X, 0, "width must be divisible by TILE_X");
        assert_eq!(height % TILE_Y, 0, "height must be divisible by TILE_Y");
        Tiled {
            stride: Stride::from_usize(width * TILE_Y),
            inner: Inner::from_size(TILE_X, TILE_Y),
        }
    }

    fn get_index(&self, x: usize, y: usize) -> usize {
        // (y / TILE_Y) * self.stride.to_usize()
        //     + (x / TILE_X) * (TILE_X * TILE_Y)
        //     + (y % TILE_Y) * TILE_X
        //     + x % TILE_X
        (y / TILE_Y) * self.stride.to_usize()
            + (x / TILE_X) * (TILE_X * TILE_Y)
            + self.inner.get_index(y % TILE_Y, x % TILE_X)
    }

    fn get_index_simd<const LANES: usize>(
        &self,
        x: Simd<usize, LANES>,
        y: Simd<usize, LANES>,
    ) -> Simd<usize, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        (y / Simd::splat(TILE_Y)) * Simd::splat(self.stride.to_usize())
            + (x / Simd::splat(TILE_X)) * (Simd::splat(TILE_X) * Simd::splat(TILE_Y))
            + self
                .inner
                .get_index_simd(y % Simd::splat(TILE_Y), x % Simd::splat(TILE_X))
    }
}

#[derive(Clone, Copy)]
pub struct ZCurve {
    _priv: (),
}

impl IndexLayout for ZCurve {
    fn from_size(width: usize, height: usize) -> Self {
        assert_eq!(
            width, height,
            "only square textures can be indexed by a z-curve"
        );
        assert!(width < u32::MAX as usize, "maximum size exceeded");
        ZCurve { _priv: () }
    }

    // source: https://lemire.me/blog/2018/01/08/how-fast-can-you-bit-interleave-32-bit-integers/
    fn get_index(&self, x: usize, y: usize) -> usize {
        use std::arch::x86_64::_pdep_u64;
        let idx = unsafe {
            _pdep_u64(x as u64, 0x5555555555555555) | _pdep_u64(y as u64, 0xaaaaaaaaaaaaaaaa)
        };
        idx as usize
    }

    // source: https://lemire.me/blog/2018/01/09/how-fast-can-you-bit-interleave-32-bit-integers-simd-edition/
    fn get_index_simd<const LANES: usize>(
        &self,
        _x: Simd<usize, LANES>,
        _y: Simd<usize, LANES>,
    ) -> Simd<usize, LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        todo!()
    }
}

pub type OwnedTexture<T, I = RowMajor> = Texture<T, OwnedStorage<T>, I>;
pub type BorrowedTexture<'a, T, I = RowMajor> = Texture<T, BorrowedStorage<'a, T>, I>;
pub type BorrowedMutTexture<'a, T, I = RowMajor> = Texture<T, BorrowedMutStorage<'a, T>, I>;

impl<T, S, Width, Height, I> Texture<T, S, Width, Height, I>
where
    Width: Dim,
    Height: Dim,
    S: Storage<Elem = T>,
    I: IndexLayout,
{
    pub fn len(&self) -> usize {
        self.width() * self.height()
    }

    pub fn to_vec(&self) -> std::vec::Vec<T>
    where
        T: Clone,
    {
        (0..self.len())
            .map(|i| {
                let ptr = self.storage.ptr();
                let value_ref = unsafe { &*ptr.add(i) };
                value_ref.clone()
            })
            .collect()
    }

    pub fn to_owned(&self) -> Texture<T, OwnedStorage<T>, Width, Height, I>
    where
        T: Clone,
    {
        Texture {
            width: self.width,
            height: self.height,
            storage: OwnedStorage(self.to_vec().into()),
            indexer: self.indexer.clone(),
            _marker: PhantomData,
        }
    }

    pub fn borrow<'a>(&'a self) -> Texture<T, BorrowedStorage<'a, T>, Width, Height, I> {
        Texture {
            width: self.width,
            height: self.height,
            storage: self.storage.borrow(),
            indexer: self.indexer,
            _marker: PhantomData,
        }
    }

    pub fn borrow_mut<'a>(&'a mut self) -> Texture<T, BorrowedMutStorage<'a, T>, Width, Height, I>
    where
        S: StorageMut,
    {
        Texture {
            width: self.width,
            height: self.height,
            storage: self.storage.borrow_mut(),
            indexer: self.indexer,
            _marker: PhantomData,
        }
    }

    pub fn width(&self) -> usize {
        self.width.to_usize()
    }

    pub fn height(&self) -> usize {
        self.height.to_usize()
    }
}

impl<'a, T, Width, Height, I> Texture<T, BorrowedStorage<'a, T>, Width, Height, I>
where
    Width: Dim,
    Height: Dim,
    I: IndexLayout,
{
    pub fn from_slice(width: usize, height: usize, slice: &'a [T]) -> Self {
        assert_eq!(width * height, slice.len());
        Texture {
            width: Width::from_usize(width),
            height: Height::from_usize(height),
            storage: BorrowedStorage {
                ptr: slice.as_ptr(),
                _marker: PhantomData,
            },
            indexer: I::from_size(width, height),
            _marker: PhantomData,
        }
    }
}

impl<'a, T, Width, Height, I> Texture<T, BorrowedMutStorage<'a, T>, Width, Height, I>
where
    Width: Dim,
    Height: Dim,
    I: IndexLayout,
{
    pub fn from_mut_slice(width: usize, height: usize, slice: &'a mut [T]) -> Self {
        assert_eq!(width * height, slice.len());
        Texture {
            width: Width::from_usize(width),
            height: Height::from_usize(height),
            storage: BorrowedMutStorage {
                ptr: slice.as_mut_ptr(),
                _marker: PhantomData,
            },
            indexer: I::from_size(width, height),
            _marker: PhantomData,
        }
    }
}

impl<T, S, Width, Height, I> Index<(usize, usize)> for Texture<T, S, Width, Height, I>
where
    Width: Dim,
    Height: Dim,
    S: Storage<Elem = T>,
    I: IndexLayout,
{
    type Output = T;

    fn index(&self, (x, y): (usize, usize)) -> &T {
        assert!(x < self.width() && y < self.height(), "out of bounds");
        let idx = self.indexer.get_index(x, y);
        let ptr = self.storage.ptr();
        unsafe { &*ptr.add(idx) }
    }
}

impl<T, S, Width, Height, I> IndexMut<(usize, usize)> for Texture<T, S, Width, Height, I>
where
    Width: Dim,
    Height: Dim,
    S: StorageMut<Elem = T>,
    I: IndexLayout,
{
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut T {
        assert!(x < self.width() && y < self.height(), "out of bounds");
        let idx = self.indexer.get_index(x, y);
        let ptr = self.storage.ptr_mut();
        unsafe { &mut *ptr.add(idx) }
    }
}

/// Singleton `usize` type
#[derive(Clone, Copy)]
pub struct KnownNat<const N: usize>;

/// Dynamic (runtime) `usize` type
#[derive(Clone, Copy)]
pub struct SomeNat(usize);

pub trait Dim: Copy {
    fn from_usize(value: usize) -> Self;
    fn to_usize(&self) -> usize;
}

impl<const N: usize> Dim for KnownNat<N> {
    fn from_usize(value: usize) -> Self {
        assert_eq!(N, value);
        KnownNat
    }

    fn to_usize(&self) -> usize {
        N
    }
}

impl Dim for SomeNat {
    fn from_usize(value: usize) -> Self {
        SomeNat(value)
    }

    fn to_usize(&self) -> usize {
        self.0
    }
}
