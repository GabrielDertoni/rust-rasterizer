use std::{vec::Vec as AllocVec, iter::IntoIterator};

use crate::{vec::{Vec4, Vec3, Vec2}, Vertex, obj, VertexBuf};

/// INVARIANT: The length of all fields is the same
#[derive(Clone, Default)]
pub struct VertBuf {
    pub positions: AllocVec<Vec4>,
    pub normals: AllocVec<Vec3>,
    pub uvs: AllocVec<Vec2>,
}

impl VertBuf {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        VertBuf {
            positions: AllocVec::with_capacity(cap),
            normals: AllocVec::with_capacity(cap),
            uvs: AllocVec::with_capacity(cap),
        }
    }

    pub fn push(&mut self, vertex: Vertex) {
        self.positions.push(vertex.position);
        self.normals.push(vertex.normal);
        self.uvs.push(vertex.uv);
    }

    pub fn from_obj(obj: &obj::Obj) -> (Self, AllocVec<[usize; 3]>) {
        let mut vert_idxs_set = obj
            .tris
            .iter()
            .copied()
            .flatten()
            .collect::<std::vec::Vec<_>>();
        vert_idxs_set.sort_unstable();
        vert_idxs_set.dedup();
        let vert_buf = vert_idxs_set
            .iter()
            .map(|idxs| Vertex {
                position: obj.verts[idxs.position as usize],
                normal: obj.normals[idxs.normal as usize],
                uv: obj.uvs[idxs.uv as usize],
            })
            .collect();

        let index_buf = obj
            .tris
            .iter()
            .map(|tri| tri.map(|v| vert_idxs_set.binary_search(&v).unwrap()))
            .collect();

        (vert_buf, index_buf)
    }
}

impl VertexBuf for VertBuf {
    type Vertex = Vertex;

    fn index(&self, index: usize) -> Vertex {
        Vertex {
            position: self.positions[index],
            normal: self.normals[index],
            uv: self.uvs[index],
        }
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.positions.len(), self.normals.len());
        debug_assert_eq!(self.positions.len(), self.uvs.len());
        self.positions.len()
    }
}

impl FromIterator<Vertex> for VertBuf {
    fn from_iter<T: IntoIterator<Item = Vertex>>(iter: T) -> Self {
        let mut vert_buf = VertBuf::default();
        vert_buf.extend(iter);
        vert_buf
    }
}

impl std::iter::Extend<Vertex> for VertBuf {
    fn extend<T: IntoIterator<Item = Vertex>>(&mut self, iter: T) {
        for vertex in iter {
            self.push(vertex)
        }
    }
}

impl IntoIterator for VertBuf {
    type Item = Vertex;
    type IntoIter = VertBufIterator;

    fn into_iter(self) -> VertBufIterator {
        VertBufIterator {
            positions: self.positions.into_iter(),
            normals: self.normals.into_iter(),
            uvs: self.uvs.into_iter(),
        }
    }
}

pub struct VertBufIterator {
    positions: std::vec::IntoIter<Vec4>,
    normals: std::vec::IntoIter<Vec3>,
    uvs: std::vec::IntoIter<Vec2>,
}

impl Iterator for VertBufIterator {
    type Item = Vertex;

    fn next(&mut self) -> Option<Vertex> {
        Some(Vertex {
            position: self.positions.next()?,
            normal: self.normals.next()?,
            uv: self.uvs.next()?,
        })
    }
}

/// INVARIANT: The length of all fields is the same
#[derive(Clone, Copy)]
pub struct VertBufSlice<'a> {
    pub positions: &'a [Vec4],
    pub normals: &'a [Vec3],
    pub uvs: &'a [Vec2],
}

impl<'a> VertexBuf for VertBufSlice<'a> {
    type Vertex = Vertex;

    #[inline(always)]
    fn index(&self, index: usize) -> Vertex {
        Vertex {
            position: self.positions[index],
            normal: self.normals[index],
            uv: self.uvs[index],
        }
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.positions.len(), self.normals.len());
        debug_assert_eq!(self.positions.len(), self.uvs.len());
        self.positions.len()
    }
}