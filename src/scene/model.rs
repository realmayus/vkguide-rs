use crate::scene::mesh::Mesh;

#[derive(Default)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub children: Vec<Model>,
}
