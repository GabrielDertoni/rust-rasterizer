use std::path::Path;

use anyhow::{Context, Result, anyhow};

use crate::vec::{self, Vec2, Vec3, Vec4};

macro_rules! implies {
    ($a:expr => $b:expr) => {
        !($a) || $b
    };
}

pub struct Vertex {
    pub position: Vec4,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Clone, Copy, Debug)]
pub struct Index {
    pub position: u32,
    pub normal: u32,
    pub uv: u32,
}

pub struct Obj {
    pub verts: Vec<Vec4>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub tris: Vec<[Index; 3]>,
    pub materials: Vec<Material>,
    has_normals: bool,
    has_uvs: bool,
}

impl Obj {
    pub fn load(path: &Path) -> Result<Obj> {
        let obj = std::fs::read_to_string(path)
            .with_context(|| anyhow!("failed to open {path:?}"))?;
        let root = path.parent().unwrap(); // we know it's a file name
        let mut verts = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tris = Vec::new();
        let mut materials = Vec::new();
        for line in obj.lines() {
            let mut it = line.split_ascii_whitespace();
            match it.next() {
                Some("v") => {
                    let vec: Vec3 = parse_vertex_coords(it)
                        .with_context(|| "failed to parse position vertex")?;
                    verts.push(Vec4::from([vec.x, vec.y, vec.z, 1.0]))
                },
                Some("vn") => normals.push({
                    parse_vertex_coords(it)
                        .with_context(|| "failed to parse normal vertex")?
                }),
                Some("vt") => uvs.push({
                    let vec: Vec3 = parse_vertex_coords(it)
                        .with_context(|| "failed to parse uv vertex")?;
                    vec.xy()
                }),
                Some("f") => parse_face_idxs(it, &mut tris)?,
                Some("mtllib") => {
                    let fname = it.next().ok_or_else(|| anyhow!("expected .mtl file name"))?;
                    Material::load_lib(root.join(fname).as_ref(), &mut materials)?;
                }
                Some("#") => continue,
                _ => continue,
            }
        }
        // HACK: The default value for an index is 0, so we must have at least one element as a placeholder.
        let has_normals = normals.len() > 0;
        let has_uvs = uvs.len() > 0;
        if !has_normals {
            normals.push(Vec3::zero());
        }
        if !has_uvs {
            uvs.push(Vec2::zero());
        }
        let obj = Obj {
            verts,
            normals,
            uvs,
            tris,
            materials,
            has_normals,
            has_uvs,
        };
        if !obj.check() {
            return Err(anyhow!("failed in bounds check"));
        }
        Ok(obj)
    }

    pub fn has_normals(&self) -> bool {
        self.has_normals
    }

    pub fn has_uvs(&self) -> bool {
        self.has_uvs
    }

    pub fn iter_vertices<'a>(&'a self) -> impl Iterator<Item = Vertex> + 'a {
        self.verts.iter()
            .zip(&self.normals)
            .zip(&self.uvs)
            .map(|((&position, &normal), &uv)| Vertex { position, normal, uv })
    }

    pub fn compute_normals(&mut self) {
        self.normals.clear();
        for _ in 0..self.verts.len() {
            self.normals.push(Vec3::zero());
        }

        for [p0_ix, p1_ix, p2_ix] in &mut self.tris {
            let i0 = p0_ix.position as usize;
            let i1 = p1_ix.position as usize;
            let i2 = p2_ix.position as usize;
            let p0 = self.verts[i0].xyz();
            let p1 = self.verts[i1].xyz();
            let p2 = self.verts[i2].xyz();
            let n = (p0 - p1).cross(p2 - p1).normalized();
            self.normals[i0] += n;
            self.normals[i1] += n;
            self.normals[i2] += n;
            // Make the one normal per self.vertsex
            p0_ix.normal = p0_ix.position;
            p1_ix.normal = p1_ix.position;
            p2_ix.normal = p2_ix.position;
        }

        for n in &mut self.normals {
            n.normalize();
        }
        self.has_normals = true;
    }

    // Verify if all indices are inbound
    fn check(&self) -> bool {
        self.tris.iter().flatten().all(|idx| {
            idx.position < self.verts.len() as u32
                && implies!(self.has_normals() => idx.normal < self.normals.len() as u32)
                && implies!(self.has_uvs() => idx.uv < self.uvs.len() as u32)
        })
    }
}

fn parse_vertex_coords<'a, const N: usize>(it: impl Iterator<Item = &'a str>) -> Result<vec::Vec<f32, N>> {
    let v = it.map(|el| el.parse::<f32>()).collect::<Result<Vec<_>, _>>()?;
    let arr: [f32; N] = v
        .try_into()
        .map_err(|_| anyhow!("expected {N} coordinates"))?;
    Ok(vec::Vec::from(arr))
}

fn parse_face_idxs<'a>(it: impl Iterator<Item = &'a str>, v: &mut Vec<[Index; 3]>) -> Result<()> {
    let idxs =
        it.map(|el| {
            let idxs = el.split('/')
                .map(|idx| {
                    idx.parse::<u32>()
                        .map(|i| i - 1) // Convert to 0-based indexing
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()?;
            let idxs = match idxs.len() {
                1 => Index { position: idxs[0], uv: 0      , normal: 0       },
                2 => Index { position: idxs[0], uv: idxs[1], normal: 0       },
                3 => Index { position: idxs[0], uv: idxs[1], normal: idxs[2] },
                n => return Err(anyhow!("invalid number of face indices {n}")),
            };
            Ok(idxs)
        })
        .collect::<Result<Vec<_>>>()?;

    match idxs.len() {
        3 => v.push(idxs.try_into().unwrap()),
        4 => {
            v.push([idxs[0], idxs[1], idxs[2]]);
            v.push([idxs[0], idxs[2], idxs[3]]);
        },
        n => return Err(anyhow!("invalid number of face vertices {n}")),
    }
    Ok(())
}

pub struct Material {
    pub name: String,
    pub map_kd: image::RgbaImage,
}

impl Material {
    pub fn load_lib(path: &Path, materials: &mut Vec<Self>) -> Result<()> {
        let contents = std::fs::read_to_string(path)?;
        let root = path.parent().unwrap(); // we know it's a file name

        let mut lines = contents.lines();
        while let Some(line) = lines.next() {
            let line = line.trim();
            let mut spaces = line.split_ascii_whitespace();
            match spaces.next() {
                Some("newmtl") => {
                    let name = spaces.next().ok_or_else(|| anyhow!("expected material name"))?;
                    materials.push(parse_material(root, name.into(), &mut lines)?);
                }
                _ => continue,
            }
        }
        Ok(())
    }
}

fn parse_material<'a>(root: &Path, name: String, lines: impl Iterator<Item = &'a str>) -> Result<Material> {
    use std::collections::HashMap;

    let mut it = lines.peekable();

    let mut attrs = HashMap::new();

    while let Some(line) = it.next_if(|line| !line.trim().starts_with("newmtl")) {
        let mut spaces = line.split_ascii_whitespace();

        match spaces.next() {
            None | Some("#" | "") => continue,
            Some(s) => { attrs.insert(s, spaces.remainder().unwrap_or("")); }
        }
    }

    let get_attr = |name: &str| -> Result<&str> {
        Ok(*attrs.get(&name)
            .ok_or_else(|| anyhow!("required attribute material {name} not found"))?)
    };

    Ok(Material {
        name,
        map_kd: {
            let fname = get_attr("map_Kd")?.trim();
            decode_image(root.join(fname).as_ref())?
        }
    })
}

fn decode_image(path: &Path) -> Result<image::RgbaImage> {
    Ok(image::io::Reader::open(path)?.decode()?.to_rgba8())
}
