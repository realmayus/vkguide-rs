use ash::{Device, vk};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
pub(crate) struct Vertex {
    position: [f32; 3],
    uv_x: f32,
    normal: [f32; 3],
    uv_y: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
pub(crate) struct PushConstants {
    world_matrix: [[f32; 4]; 4],
    vertex_buffer: vk::DeviceAddress,
}

pub struct PipelineBuilder {
    pub shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    pub input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    pub rasterization: vk::PipelineRasterizationStateCreateInfo,
    pub color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    pub multisample: vk::PipelineMultisampleStateCreateInfo,
    pub layout: vk::PipelineLayout,
    pub depth_stencil: vk::PipelineDepthStencilStateCreateInfo,
    pub render_info: vk::PipelineRenderingCreateInfo,
    pub color_attachment_format: vk::Format,
}
impl PipelineBuilder {

    pub(crate) fn build(mut self, device: &Device) -> vk::Pipeline {
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)  // dynamic state allows us to only specify count
            .scissor_count(1);
        let color_blend_attachments = [self.color_blend_attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachments)
            .logic_op(vk::LogicOp::COPY);
        // we don't need this as we're using dynamic state
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder();
        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_state);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterization)
            .multisample_state(&self.multisample)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&self.depth_stencil)
            .layout(self.layout)
            .push_next(&mut self.render_info)
            .dynamic_state(&dynamic_state_info);

        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None).unwrap()[0] }
    }
}