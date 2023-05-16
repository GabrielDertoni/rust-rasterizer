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
    pub texture: Vec2,
}

pub struct Obj {
    pub vert: Vec<Vec4>,
    pub normal: Vec<Vec3>,
    pub texture: Vec<Vec2>,
    // pub tris: Vec<[[u32; 3]; 3]>,
    pub tris: Vec<[u32; 3]>,
    pub materials: Vec<Material>,
}

impl Obj {
    pub fn load(path: &Path) -> Result<Obj> {
        let obj = std::fs::read_to_string(path)
            .with_context(|| anyhow!("failed to open {path:?}"))?;
        let root = path.parent().unwrap(); // we know it's a file name
        let mut vert = Vec::new();
        let mut normal = Vec::new();
        let mut texture = Vec::new();
        let mut tris = Vec::new();
        let mut materials = Vec::new();
        for line in obj.lines() {
            let mut it = line.split_ascii_whitespace();
            match it.next() {
                Some("v") => {
                    let vec: Vec3 = parse_vertex_coords(it)
                        .with_context(|| "failed to parse position vertex")?;
                    vert.push(Vec4::from([vec.x, vec.y, vec.z, 1.0]))
                },
                Some("vn") => normal.push({
                    parse_vertex_coords(it)
                        .with_context(|| "failed to parse normal vertex")?
                }),
                Some("vt") => texture.push({
                    let vec: Vec3 = parse_vertex_coords(it)
                        .with_context(|| "failed to parse texture vertex")?;
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
        Ok(Obj {
            vert,
            normal,
            texture,
            tris,
            materials,
        })
    }

    pub fn has_normals(&self) -> bool {
        implies!(self.vert.len() > 0 => self.normal.len() > 0)
    }

    pub fn has_texture(&self) -> bool {
        implies!(self.vert.len() > 0 => self.texture.len() > 0)
    }

    pub fn iter_vertices<'a>(&'a self) -> impl Iterator<Item = Vertex> + 'a {
        self.vert.iter()
            .zip(&self.normal)
            .zip(&self.texture)
            .map(|((&position, &normal), &texture)| Vertex { position, normal, texture })
    }
}

fn parse_vertex_coords<'a, const N: usize>(it: impl Iterator<Item = &'a str>) -> Result<vec::Vec<f32, N>> {
    let v = it.map(|el| el.parse::<f32>()).collect::<Result<Vec<_>, _>>()?;
    let arr: [f32; N] = v
        .try_into()
        .map_err(|_| anyhow!("expected {N} coordinates"))?;
    Ok(vec::Vec::from(arr))
}

// fn parse_face_idxs<'a>(it: impl Iterator<Item = &'a str>) -> Result<[[u32; 3]; 3]> {
fn parse_face_idxs<'a>(it: impl Iterator<Item = &'a str>, v: &mut Vec<[u32; 3]>) -> Result<()> {
    let idxs =
        it.map(|el| {
            let idxs = el.split('/')
                .map(|idx| {
                    idx.parse::<u32>()
                        .map(|i| i - 1)
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()?;

            /*
            let idxs = match idxs.len() {
                1 => [idxs[0], 0, 0],
                2 => [idxs[0], idxs[1], 0],
                3 => [idxs[0], idxs[1], idxs[2]],
                n => return Err(anyhow!("invalid number of face indices {n}")),
            };
            */
            anyhow::Ok(idxs[0])
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
