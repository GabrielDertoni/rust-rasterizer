// unused

pub struct SimpleLineToIter {
    x: i32,
    x1: i32,
    dy: i32,
}

impl SimpleLineToIter {
    pub fn new(x: i32, y: i32) -> Self {
        SimpleLineToIter {
            x: 0,
            x1: x,
            dy: y,
        }
    }
}

impl Iterator for SimpleLineToIter {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<(i32, i32)> {
        if self.x <= self.x1 {
            let x = self.x;
            self.x += 1;
            let y = x * self.dy / self.x1;
            Some((x, y))
        } else {
            None
        }
    }
}

impl DoubleEndedIterator for SimpleLineToIter {
    fn next_back(&mut self) -> Option<(i32, i32)> {
        if self.x <= self.x1 {
            let x = self.x1;
            self.x1 -= 1;
            let y = x * self.dy / self.x1;
            Some((x, y))
        } else {
            None
        }
    }
}
