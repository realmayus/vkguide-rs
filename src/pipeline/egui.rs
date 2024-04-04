use crate::pipeline::PipelineBuilder;
use crate::resources::{AllocUsage, AllocatedBuffer, Allocator, Texture, TextureId, TextureManager};
use crate::scene::mesh::Mesh;
use crate::util::{encode_4_u8_as_3_f32, load_shader_module, DeletionQueue};
use crate::{SubmitContext, FRAME_OVERLAP};
use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};
use egui::ahash::{HashMap, HashMapExt};
use egui::epaint::{ImageDelta, Primitive};
use egui::{Context, FullOutput, ImageData, TexturesDelta};
use glam::{Vec2, Vec3};
use log::debug;
use std::ffi::CStr;
use winit::window::Window;

type EguiTextureId = egui::TextureId;

pub struct EguiPipeline {
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    window_size: (u32, u32),
    context: Context,
    egui_winit: egui_winit::State,
    textures: HashMap<EguiTextureId, TextureId>,
    mesh_buffers: Vec<(AllocatedBuffer, AllocatedBuffer)>, // Vertex buffer, index buffer
    bindless_set_layout: vk::DescriptorSetLayout,
    raw_input: egui::RawInput,
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
struct PushConstants {
    screen_size: [f32; 2],
    vertex_buffer: vk::DeviceAddress,
    font_texture_id: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: u32,
    padding: u32,
}

impl EguiPipeline {
    const INDEX_BUFFER_SIZE: usize = 1024 * 1024;
    const VERTEX_BUFFER_SIZE: usize = 1024 * 1024;

    pub fn new(
        device: &ash::Device,
        window_size: (u32, u32),
        deletion_queue: &mut DeletionQueue,
        bindless_set_layout: vk::DescriptorSetLayout,
        window: &Window,
        submit_context: SubmitContext,
    ) -> Self {
        let vertex_shader =
            load_shader_module(device, include_bytes!("../shaders/spirv/egui.vert.spv")).expect("Failed to load vertex shader module");
        let fragment_shader =
            load_shader_module(device, include_bytes!("../shaders/spirv/egui.frag.spv")).expect("Failed to load fragment shader module");

        let push_constant_range = [vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let binding = [bindless_set_layout];
        let layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&binding)
            .push_constant_ranges(&push_constant_range);
        let layout = unsafe { device.create_pipeline_layout(&layout_create_info, None).unwrap() };
        let pipeline_builder = PipelineBuilder {
            layout: Some(layout),
            shader_stages: vec![
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader)
                    .name(CStr::from_bytes_with_nul(b"main\0").unwrap()),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader)
                    .name(CStr::from_bytes_with_nul(b"main\0").unwrap()),
            ],
            ..Default::default()
        };

        let pipeline = pipeline_builder.build(device);

        unsafe {
            device.destroy_shader_module(vertex_shader, None);
            device.destroy_shader_module(fragment_shader, None);
        }

        deletion_queue.push(move |device, allocator| unsafe {
            device.destroy_pipeline_layout(layout, None);
            device.destroy_pipeline(pipeline, None);
        });

        let viewport = vk::Viewport::default()
            .width(window_size.0 as f32)
            .height(window_size.1 as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(vk::Extent2D {
            width: window_size.0,
            height: window_size.1,
        });

        let context = Context::default();
        let egui_winit = egui_winit::State::new(
            context.clone(),
            context.viewport_id(),
            window,
            Some(window.scale_factor() as f32),
            None,
        );

        let mesh_buffers: [(AllocatedBuffer, AllocatedBuffer); FRAME_OVERLAP] = core::array::from_fn(|_| {
            let vertex_buffer = AllocatedBuffer::new(
                device,
                &mut submit_context.allocator.borrow_mut(),
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                AllocUsage::Shared,
                Self::VERTEX_BUFFER_SIZE as vk::DeviceSize,
                Some("Egui Vertex Buffer".into()),
            );
            let index_buffer = AllocatedBuffer::new(
                device,
                &mut submit_context.allocator.borrow_mut(),
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                AllocUsage::Shared,
                Self::INDEX_BUFFER_SIZE as vk::DeviceSize,
                Some("Egui Index Buffer".into()),
            );
            (vertex_buffer, index_buffer)
        });
        Self {
            viewport,
            scissor,
            pipeline,
            layout,
            window_size,
            context,
            egui_winit,
            textures: HashMap::new(),
            mesh_buffers: mesh_buffers.into(),
            bindless_set_layout,
            raw_input: egui::RawInput::default(),
        }
    }
    pub fn context(&self) -> &Context {
        &self.context
    }
    pub fn resize(&mut self, window_size: (u32, u32)) {
        self.window_size = window_size;
        self.viewport = vk::Viewport::default()
            .width(window_size.0 as f32)
            .height(window_size.1 as f32)
            .max_depth(1.0);
        self.scissor = vk::Rect2D::default().extent(vk::Extent2D {
            width: window_size.0,
            height: window_size.1,
        });
    }
    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.egui_winit.take_egui_input(window);
        self.context.begin_frame(raw_input);
    }
    pub fn input(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        let res = self.egui_winit.on_window_event(window, event);
        res.consumed
    }
    pub fn draw(
        &mut self,
        device: &Device,
        cmd: vk::CommandBuffer,
        target_view: vk::ImageView,
        depth_view: vk::ImageView,
        bindless_descriptor_set: vk::DescriptorSet,
        textures_delta: TexturesDelta,
        clipped_meshes: Vec<egui::ClippedPrimitive>,
        texture_manager: &mut TextureManager,
        submit_context: SubmitContext,
        image_index: usize,
        window: &Window,
    ) {
        for (id, image_delta) in textures_delta.set {
            self.update_texture(id, image_delta, submit_context.clone(), texture_manager);
        }
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(target_view)
            .image_layout(vk::ImageLayout::GENERAL);
        let color_attachments = [color_attachment];
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
            });
        let render_info = vk::RenderingInfo::default()
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.window_size.0,
                    height: self.window_size.1,
                },
            })
            .layer_count(1)
            .view_mask(0);

        let mut vertex_buffer_ptr = unsafe {
            self.mesh_buffers[image_index]
                .0
                .allocation
                .map(device, 0, Self::VERTEX_BUFFER_SIZE)
                .unwrap()
                .as_ptr() as *mut Vertex
        };
        let vertex_buffer_end = unsafe { vertex_buffer_ptr.add(Self::VERTEX_BUFFER_SIZE) };
        let mut index_buffer_ptr = unsafe {
            self.mesh_buffers[image_index]
                .1
                .allocation
                .map(device, 0, Self::INDEX_BUFFER_SIZE)
                .unwrap()
                .as_ptr() as *mut u32
        };
        let index_buffer_end = unsafe { index_buffer_ptr.add(Self::INDEX_BUFFER_SIZE) };
        for egui::ClippedPrimitive { clip_rect: _, primitive } in &clipped_meshes {
            let emesh = match primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => unimplemented!(),
            };
            if emesh.vertices.is_empty() || emesh.indices.is_empty() {
                continue;
            }
            let v_slice = &emesh
                .vertices
                .iter()
                .map(|v| Vertex {
                    pos: [v.pos.x, v.pos.y],
                    uv: [v.uv.x, v.uv.y],
                    color: v.color.a() as u32 | (v.color.b() as u32) << 8 | (v.color.g() as u32) << 16 | (v.color.r() as u32) << 24,
                    padding: 0,
                })
                .collect::<Vec<_>>();
            let v_copy_count = v_slice.len();

            let i_slice = &emesh.indices;
            let i_copy_count = i_slice.len();

            let vertex_buffer_ptr_next = unsafe { vertex_buffer_ptr.add(v_copy_count) };
            let index_buffer_ptr_next = unsafe { index_buffer_ptr.add(i_copy_count) };
            if vertex_buffer_ptr_next > vertex_buffer_end || index_buffer_ptr_next > index_buffer_end {
                panic!("egui vertex/index buffer overflow");
            }
            unsafe {
                vertex_buffer_ptr.copy_from(v_slice.as_ptr().cast(), v_copy_count);
                index_buffer_ptr.copy_from(i_slice.as_ptr().cast(), i_copy_count);
            }
            vertex_buffer_ptr = vertex_buffer_ptr_next;
            index_buffer_ptr = index_buffer_ptr_next;
        }

        unsafe {
            device.cmd_begin_rendering(cmd, &render_info);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_set_viewport(cmd, 0, &[self.viewport]);
            device.cmd_set_scissor(cmd, 0, &[self.scissor]);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[bindless_descriptor_set],
                &[],
            );
        }
        let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(self.mesh_buffers[image_index].0.buffer);
        let buffer_device_address = unsafe { device.get_buffer_device_address(&device_address_info) };

        let mut vertex_offset = 0u32;
        let mut index_offset = 0u32;

        let push_constants = PushConstants {
            screen_size: [self.window_size.0 as f32, self.window_size.1 as f32],
            vertex_buffer: buffer_device_address,
            font_texture_id: 0,
            padding: 0,
        };
        for egui::ClippedPrimitive { clip_rect: _, primitive } in &clipped_meshes {
            let emesh = match primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => unimplemented!(),
            };
            if emesh.vertices.is_empty() || emesh.indices.is_empty() {
                continue;
            }

            let vertices = &emesh.vertices;
            let indices = &emesh.indices;
            // copy vertices and indices into mesh buffer
            // todo!();
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    self.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&[push_constants]),
                );
                device.cmd_bind_index_buffer(cmd, self.mesh_buffers[image_index].1.buffer, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cmd, indices.len() as u32, 1, index_offset, vertex_offset as i32, 0);
            }
            vertex_offset += vertices.len() as u32;
            index_offset += indices.len() as u32;
        }
        unsafe {
            device.cmd_end_rendering(cmd);
            self.mesh_buffers[image_index].0.allocation.unmap(device);
            self.mesh_buffers[image_index].1.allocation.unmap(device);
        }
    }
    pub fn end_frame(&mut self, window: &Window) -> FullOutput {
        let output = self.context.end_frame();
        self.egui_winit.handle_platform_output(window, output.platform_output.clone());
        output
    }

    fn update_texture(
        &mut self,
        egui_texture_id: EguiTextureId,
        delta: ImageDelta,
        mut ctx: SubmitContext,
        texture_manager: &mut TextureManager,
    ) {
        let data = match &delta.image {
            ImageData::Color(image) => image.pixels.iter().flat_map(|color| color.to_array()).collect::<Vec<_>>(),
            ImageData::Font(image) => image.srgba_pixels(None).flat_map(|color| color.to_array()).collect::<Vec<_>>(),
        };
        ctx.immediate_submit(Box::new(|ctx| {
            if let Some(texture_id) = self.textures.get(&egui_texture_id) {
                debug!("Replacing egui texture");
                texture_manager.texture_mut(*texture_id).replace_image(
                    ctx,
                    Some(format!("egui texture, id: {:?}", egui_texture_id)),
                    data.as_slice(),
                    vk::Extent3D {
                        width: delta.image.width() as u32,
                        height: delta.image.height() as u32,
                        depth: 1,
                    },
                );
            } else {
                debug!("Adding egui texture");
                let texture = Texture::new(
                    TextureManager::DEFAULT_SAMPLER_LINEAR,
                    ctx,
                    Some(format!("egui texture, id: {:?}", egui_texture_id)),
                    data.as_slice(),
                    vk::Extent3D {
                        width: delta.image.width() as u32,
                        height: delta.image.height() as u32,
                        depth: 1,
                    },
                );
                texture_manager.add_texture(texture, &ctx.device, true);
            }
        }))
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_descriptor_set_layout(self.bindless_set_layout, None);
        }
        for buf in self.mesh_buffers.drain(..) {
            buf.0.destroy(device, allocator);
            buf.1.destroy(device, allocator);
        }
    }
}
