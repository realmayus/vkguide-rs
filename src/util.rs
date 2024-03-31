use std::error::Error;
use ash::{Device, vk};
use glam::{Mat4, Vec4};
use crate::resources::{AllocatedBuffer, Allocator, DescriptorAllocator};


pub struct FrameData {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub swapchain_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
    pub deletion_queue: DeletionQueue,
    pub descriptor_allocator: DescriptorAllocator,
    pub stale_buffers: Vec<AllocatedBuffer>,
}

pub struct GpuSceneData {
    pub view: Mat4,
    pub proj: Mat4,
    pub viewproj: Mat4,
    pub ambient_color: Vec4,
    pub sun_dir: Vec4,
    pub sun_color: Vec4,
}

#[derive(Default)]
pub(crate) struct DeletionQueue {
    deletors: Vec<Box<dyn FnOnce(&Device) + Send>>,
}

impl DeletionQueue {
    pub(crate) fn push<T: 'static + FnOnce(&Device) + Send>(&mut self, deleter: T) {
        self.deletors.push(Box::new(deleter));
    }
    pub(crate) fn flush(&mut self, device: &Device) {
        for deleter in self.deletors.drain(..) {
            deleter(device);
        }
    }
}

#[derive(Default)]
pub struct DescriptorLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl DescriptorLayoutBuilder {
    pub fn add_binding(mut self, binding: u32, descriptor_type: vk::DescriptorType, stage_flags: vk::ShaderStageFlags) -> Self {
        self.bindings.push(vk::DescriptorSetLayoutBinding::builder()
            .binding(binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(1)
            .stage_flags(stage_flags)
            .build());
        self
    }
    
    pub fn clear(mut self) {
        self.bindings.clear();
    }

    pub fn build(self, device: &Device) -> vk::DescriptorSetLayout {
        
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&self.bindings);
        unsafe { device.create_descriptor_set_layout(&info, None) }.unwrap()
    }
}
pub mod device_discovery {
    use std::ffi::CStr;
    use ash::extensions::khr::Surface;
    use ash::{Instance, vk};
    use log::info;

    pub(crate) fn pick_physical_device(instance: &Instance, surface: &Surface, surface_khr: vk::SurfaceKHR) -> vk::PhysicalDevice {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| is_device_suitable(instance, surface, surface_khr, *device))
            .expect("No suitable physical device.");

        let props = unsafe { instance.get_physical_device_properties(device) };
        info!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });
        device
    }

    fn is_device_suitable(instance: &Instance, surface: &Surface, surface_khr: vk::SurfaceKHR, device: vk::PhysicalDevice) -> bool {
        let (graphics, present) = find_queue_families(instance, surface, surface_khr, device);
        graphics.is_some() && present.is_some()
    }
    pub(crate) fn find_queue_families(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe { surface.get_physical_device_surface_support(device, index, surface_khr).unwrap() };
            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }

        (graphics, present)
    }
}

pub(crate) fn transition_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    current_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let image_barrier = vk::ImageMemoryBarrier2::builder()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(current_layout)
        .new_layout(new_layout)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                })
                .level_count(vk::REMAINING_MIP_LEVELS)
                .layer_count(vk::REMAINING_ARRAY_LAYERS)
                .build(),
        )
        .image(image);

    let dependency_info = vk::DependencyInfoKHR::builder()
        .image_memory_barriers(&[image_barrier.build()])
        .build();

    unsafe { device.cmd_pipeline_barrier2(cmd, &dependency_info) }
}

pub(crate) fn copy_image_to_image(device: &Device, cmd: vk::CommandBuffer, source: vk::Image, destination: vk::Image, src_extent: vk::Extent2D, dst_extent: vk::Extent2D) {
    let blit_region = vk::ImageBlit2::builder()
        .src_offsets([
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D { x: src_extent.width as i32, y: src_extent.height as i32, z: 1 },
        ])
        .dst_offsets([
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D { x: dst_extent.width as i32, y: dst_extent.height as i32, z: 1 },
        ])
        .src_subresource(vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1)
            .build())
        .dst_subresource(vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1)
            .build());

    let regions = [*blit_region];
    let blit_info = vk::BlitImageInfo2::builder()
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(destination)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .filter(vk::Filter::LINEAR)
        .regions(&regions);

    unsafe { device.cmd_blit_image2(cmd, &blit_info) };
}

pub fn load_shader_module(device: &Device, code: &[u8]) -> Result<vk::ShaderModule, Box<dyn Error>> {
    // copy vec<u8> into vec<u32> where each u32 is a 4 byte chunk of u8s
    let code: Vec<u32> = code.chunks(4).map(|chunk| {
        let mut bytes = [0u8; 4];
        for (i, byte) in chunk.iter().enumerate() {
            bytes[i] = *byte;
        }
        u32::from_ne_bytes(bytes)
    }).collect();
    let info = vk::ShaderModuleCreateInfo::builder().code(&code);
    unsafe { Ok(device.create_shader_module(&info, None)?) }
}

#[macro_export]
macro_rules! frame {
    ($m: ident) => {
        &mut $m.frames[($m.current_frame % FRAME_OVERLAP as u32) as usize]
    };
}