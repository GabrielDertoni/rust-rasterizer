use crate::vec::{Num, Vec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BBox<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T: Copy> BBox<T> {
    pub fn new(corner: Vec<T, 2>, size: Size<T>) -> Self {
        BBox {
            x: corner.x,
            y: corner.y,
            width: size.width,
            height: size.height,
        }
    }

    pub fn intersects(&self, other: BBox<T>) -> bool
    where
        T: Num + PartialOrd,
    {
        let x_intersection = Num::min(self.x + self.width, other.x + other.width) - Num::max(self.x, other.x);
        let y_intersection = Num::min(self.y + self.height, other.y + other.height) - Num::max(self.y, other.y);
        x_intersection > T::zero() && y_intersection > T::zero()
        // let x_intersection = Num::min(self.x + self.width, other.x + other.width) - Num::max(self.x, other.x);
        // let y_intersection = Num::min(self.y, other.y) - Num::max(self.y - self.height, other.y - other.height);
        // x_intersection >= T::zero() && y_intersection >= T::zero()
    }
}

impl<T: Copy> From<(Vec<T, 2>, Size<T>)> for BBox<T> {
    fn from((corner, size): (Vec<T, 2>, Size<T>)) -> Self {
        BBox::new(corner, size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

impl<T> Size<T> {
    pub fn new(width: T, height: T) -> Self {
        Size {
            width,
            height,
        }
    }
}