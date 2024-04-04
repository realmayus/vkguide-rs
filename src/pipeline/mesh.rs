use crate::pipeline::PipelineBuilder;
use crate::util::{load_shader_module, DeletionQueue};
use crate::Mesh;
use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::ffi::CStr;

pub struct MeshPipeline {
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    window_size: (u32, u32),
}
#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
pub(crate) struct PushConstants {
    pub(crate) world_matrix: [[f32; 4]; 4],
    pub(crate) vertex_buffer: vk::DeviceAddress,
}

impl MeshPipeline {
    pub fn new(device: &ash::Device, window_size: (u32, u32), deletion_queue: &mut DeletionQueue, bindless_set_layout: vk::DescriptorSetLayout) -> Self {
        let vertex_shader =
            load_shader_module(device, include_bytes!("../shaders/spirv/mesh.vert.spv")).expect("Failed to load vertex shader module");
        let fragment_shader =
            load_shader_module(device, include_bytes!("../shaders/spirv/mesh.frag.spv")).expect("Failed to load fragment shader module");

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

        deletion_queue.push(move |device| unsafe {
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
        self.viewport = vk::Viewport::default()
            .width(window_size.0 as f32)
            .height(window_size.1 as f32)
            .max_depth(1.0);
        self.scissor = vk::Rect2D::default().extent(vk::Extent2D {
            width: window_size.0,
            height: window_size.1,
        });
    }
    pub fn draw(&self, device: &Device, cmd: vk::CommandBuffer, meshes: &[Mesh], target_view: vk::ImageView, depth_view: vk::ImageView, bindless_descriptor_set: vk::DescriptorSet) {
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
        let world = {
            let view = Mat4::look_at_rh(Vec3::new(2.0, 3.0, 5.0), Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
            let mut proj = Mat4::perspective_rh(
                60.0f32.to_radians(),
                self.window_size.0 as f32 / self.window_size.1 as f32,
                10000.0,
                0.1,
            );
            proj.y_axis.y *= -1.0;

            proj * view
        };

        unsafe {
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[bindless_descriptor_set], &[]);
            device.cmd_begin_rendering(cmd, &render_info);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_set_viewport(cmd, 0, &[self.viewport]);
            device.cmd_set_scissor(cmd, 0, &[self.scissor]);
            for mesh in meshes {
                let push_constants = PushConstants {
                    world_matrix: world.to_cols_array_2d(),
                    vertex_buffer: mesh.mem.as_ref().unwrap().vertex_address,
                };
                device.cmd_push_constants(
                    cmd,
                    self.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&[push_constants]),
                );
                device.cmd_bind_index_buffer(cmd, mesh.mem.as_ref().unwrap().index_buffer.buffer, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cmd, mesh.indices.len() as u32, 1, 0, 0, 0);
            }
            device.cmd_end_rendering(cmd);
        }
    }
}
