use std::ffi::CStr;
use ash::{Device, vk};
use crate::Mesh;
use crate::pipeline::{PipelineBuilder, PushConstants};
use crate::util::{DeletionQueue, load_shader_module};

pub struct MeshPipeline {
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    window_size: (u32, u32)
}

impl MeshPipeline {
    pub fn new(device: &ash::Device, window_size: (u32, u32), deletion_queue: &mut DeletionQueue) -> Self {
        let vertex_shader =
            load_shader_module(device, include_bytes!("../shaders/spirv/mesh.vert.spv")).expect("Failed to load vertex shader module");
        let fragment_shader =
            load_shader_module(device, include_bytes!("../shaders/spirv/mesh.frag.spv")).expect("Failed to load fragment shader module");
        
        let push_constant_range = [*vk::PushConstantRange::builder().offset(0).size(std::mem::size_of::<PushConstants>() as u32).stage_flags(vk::ShaderStageFlags::VERTEX)];
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&[]).push_constant_ranges(&push_constant_range);
        let layout = unsafe { device.create_pipeline_layout(&layout_create_info, None).unwrap() };
        let pipeline_builder = PipelineBuilder {
            layout: Some(layout),
            shader_stages: vec![
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader)
                    .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader)
                    .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build(),
            ],
            ..Default::default()
        };

        let pipeline = pipeline_builder.build(device);

        unsafe {
            device.destroy_shader_module(vertex_shader, None);
            device.destroy_shader_module(fragment_shader, None);
        }

        deletion_queue.push(move |device| {
            unsafe {
                device.destroy_pipeline_layout(layout, None);
                device.destroy_pipeline(pipeline, None);
            }
        });

        
        let viewport = *vk::Viewport::builder()
            .width(window_size.0 as f32)
            .height(window_size.1 as f32)
            .max_depth(1.0);
        let scissor = *vk::Rect2D::builder().extent(vk::Extent2D {
            width: window_size.0,
            height: window_size.1,
        });
        
        Self {
            viewport,
            scissor,
            pipeline,
            layout,
            window_size,
        }
    }
    
    pub fn resize(&mut self, window_size: (u32, u32)) {
        self.window_size = window_size;
        self.viewport = *vk::Viewport::builder()
            .width(window_size.0 as f32)
            .height(window_size.1 as f32)
            .max_depth(1.0);
        self.scissor = *vk::Rect2D::builder().extent(vk::Extent2D {
            width: window_size.0,
            height: window_size.1,
        });
    }
    pub fn draw(&self, device: &Device, cmd: vk::CommandBuffer, meshes: &[Mesh], target_view: vk::ImageView, depth_view: vk::ImageView) {
        let render_info = {
            let color_attachment = vk::RenderingAttachmentInfo::builder()
                .image_view(target_view)
                .image_layout(vk::ImageLayout::GENERAL);
            let color_attachments = [color_attachment.build()];
            let depth_attachment = *vk::RenderingAttachmentInfo::builder()
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                });

            *vk::RenderingInfo::builder()
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
                .view_mask(0)
        };

        unsafe {
            device.cmd_begin_rendering(cmd, &render_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            
            device.cmd_set_viewport(cmd, 0, &[self.viewport]);
            device.cmd_set_scissor(cmd, 0, &[self.scissor]);
            for mesh in meshes {
                let push_constants = PushConstants {
                    world_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    vertex_buffer: mesh.vertex_address
                };
                device.cmd_push_constants(
                    cmd,
                    self.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&[push_constants]),
                );
                device.cmd_bind_index_buffer(cmd, mesh.index_buffer.buffer, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cmd, 6, 1, 0, 0,0);
            }
            device.cmd_end_rendering(cmd);
        }
    }
}