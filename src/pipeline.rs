pub mod egui;
pub mod mesh;

use crate::{DEPTH_FORMAT, SWAPCHAIN_IMAGE_FORMAT};
use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) uv_x: f32,
    pub(crate) normal: [f32; 3],
    pub(crate) uv_y: f32,
    pub(crate) color: [f32; 4],
}
pub struct PipelineBuilder<'a> {
    pub shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    pub input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    pub rasterization: vk::PipelineRasterizationStateCreateInfo<'a>,
    pub color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    pub multisample: vk::PipelineMultisampleStateCreateInfo<'a>,
    pub layout: Option<vk::PipelineLayout>,
    pub depth_stencil: vk::PipelineDepthStencilStateCreateInfo<'a>,
    pub render_info: vk::PipelineRenderingCreateInfo<'a>,
    pub color_attachment_format: vk::Format,
}
impl PipelineBuilder<'_> {
    pub(crate) fn build(mut self, device: &Device) -> vk::Pipeline {
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1) // dynamic state allows us to only specify count
            .scissor_count(1);
        let color_blend_attachments = [self.color_blend_attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachments)
            .logic_op(vk::LogicOp::COPY);
        // we don't need this as we're using dynamic state
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterization)
            .multisample_state(&self.multisample)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.layout.expect("Pipeline layout not set!"))
            .push_next(&mut self.render_info)
            .dynamic_state(&dynamic_state_info);

        unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        }
    }
}

impl Default for PipelineBuilder<'_> {
    fn default() -> Self {
        Self {
            shader_stages: vec![],
            input_assembly: vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST),
            rasterization: vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::CLOCKWISE)
                .line_width(1.0),
            color_blend_attachment: vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            multisample: vk::PipelineMultisampleStateCreateInfo::default()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .min_sample_shading(1.0)
                .alpha_to_coverage_enable(false)
                .alpha_to_one_enable(false),
            layout: None,
            depth_stencil: vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(Default::default())
                .back(Default::default())
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0),
            render_info: vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&[SWAPCHAIN_IMAGE_FORMAT])
                .depth_attachment_format(DEPTH_FORMAT),
            color_attachment_format: SWAPCHAIN_IMAGE_FORMAT,
        }
    }
}
