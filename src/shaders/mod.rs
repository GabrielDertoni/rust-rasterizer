use crate::{vec::Mat4x4, Vertex, VertexShader};

pub mod lit;
pub mod textured;

pub struct MVPVertShader {
    transform: Mat4x4,
}

impl MVPVertShader {
    pub fn new(transform: Mat4x4) -> Self {
        MVPVertShader { transform }
    }
}

impl VertexShader<Vertex> for MVPVertShader {
    type Output = Vertex;

    fn exec(&self, vertex: Vertex) -> Self::Output {
        Vertex {
            position: self.transform * vertex.position,
            normal: vertex.normal,
            uv: vertex.uv,
        }
    }
}
