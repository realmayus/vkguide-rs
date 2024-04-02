# vkguide-rs
## WIP!
This is a Rust implementation of https://vkguide.dev using the Ash Vulkan bindings for Rust.
I implemented everything up until the textures chapter (except for the Dear ImGUI integration), as I decided it's better to implement bindless textures rather than "bindful" textures.

Currently, the project has only been tested on Linux.

I'm using `gpu-alloc` instead of VMA, as it's a native Rust allocator implementation. Furthermore, `winit` is used instead of `SDL`, for the same reason (and `winit` is awesome!). As for the linear algebra library, `glam` is used instead of `glm`.
