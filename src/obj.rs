use std::path::Path;

use crate::vec::Vec4;

pub struct Obj {
    pub vert: Vec<Vec4>,
    pub tris: Vec<[u32; 3]>,
}

pub fn load_obj(path: &Path) -> Result<Obj, String> {
    let obj = std::fs::read_to_string(path).unwrap();
    let mut vert = Vec::new();
    let mut tris = Vec::new();
    for line in obj.lines() {
        let mut it = line.split_ascii_whitespace();
        match it.next() {
            Some("v") => {
                let mut it = it.map(|el| {
                    el.parse::<f32>()
                        .map(|num| num / 4.0)
                        .map_err(|e| e.to_string())
                });
                let [x, y, z] = it
                    .next_chunk()
                    .map_err(|_| String::from("expected vertex x, y and z coordinates"))?;
                vert.push(Vec4::from([x?, y?, z?, 1.0]));
            }
            Some("f") => {
                let mut it =
                    it.map(|el| el.parse::<u32>().map(|i| i - 1).map_err(|e| e.to_string()));
                let [p0, p1, p2] = it
                    .next_chunk()
                    .map_err(|_| String::from("expected 3 indices for triangle face"))?;
                tris.push([p0?, p1?, p2?]);
            }
            _ => continue,
        }
    }
    Ok(Obj { vert, tris })
}
