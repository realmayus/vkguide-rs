use glam::{Vec2, Vec3};
use std::path::Path;
use crate::scene::mesh::Mesh;
use crate::scene::model::Model;

pub fn load_gltf(path: &Path) -> Vec<Model> {
    let (gltf, buffers, _) = gltf::import(path).unwrap();

    gltf.scenes()
        .flat_map(|scene| scene.nodes().map(|node| load_node(node, &buffers)))
        .collect()
}

fn load_node(node: gltf::Node, buffers: &Vec<gltf::buffer::Data>) -> Model {
    let mut model = Model::default();
    for mesh in node.mesh().iter() {
        for primitive in mesh.primitives() {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            let mut normals = Vec::new();
            let mut uvs = Vec::new();
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            if let Some(iter) = reader.read_positions() {
                for position in iter {
                    vertices.push(Vec3::from(position));
                }
            }
            if let Some(iter) = reader.read_indices() {
                for index in iter.into_u32() {
                    indices.push(index);
                }
            }
            if let Some(iter) = reader.read_normals() {
                for normal in iter {
                    normals.push(Vec3::from(normal));
                }
            }
            if let Some(iter) = reader.read_tex_coords(0) {
                for uv in iter.into_f32() {
                    uvs.push(Vec2::from(uv));
                }
            }

            model.meshes.push(Mesh {
                mem: None,
                vertices,
                indices,
                normals,
                uvs,
            })
        }
    }
    for child in node.children() {
        model.children.push(load_node(child, buffers));
    }
    model
}
