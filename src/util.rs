use ash::{Device, vk};
use ash::vk::DeviceMemory;
use gpu_alloc_ash::AshMemoryDevice;
use log::log;

pub type Allocation = gpu_alloc::MemoryBlock<DeviceMemory>;
pub type Allocator = gpu_alloc::GpuAllocator<DeviceMemory>;

pub struct FrameData {
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) swapchain_semaphore: vk::Semaphore,
    pub(crate) render_semaphore: vk::Semaphore,
    pub(crate) render_fence: vk::Fence,
    pub(crate) deletion_queue: DeletionQueue,
}

#[derive(Default)]
pub(crate) struct DeletionQueue {
    deletors: Vec<Box<dyn FnOnce() + Send>>,
}

impl DeletionQueue {
    pub(crate) fn push<T: 'static + FnOnce() + Send>(&mut self, deleter: T) {
        self.deletors.push(Box::new(deleter));
    }
    pub(crate) fn flush(&mut self) {
        for deleter in self.deletors.drain(..) {
            deleter();
        }
    }
}

pub enum AllocUsage {
    GpuOnly,
    Shared,
    UploadToHost,
}

impl AllocUsage {
    pub fn flags(&self) -> gpu_alloc::UsageFlags {
        match self {
            AllocUsage::GpuOnly => gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            AllocUsage::Shared => gpu_alloc::UsageFlags::HOST_ACCESS | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS | gpu_alloc::UsageFlags::DOWNLOAD | gpu_alloc::UsageFlags::UPLOAD,
            AllocUsage::UploadToHost => gpu_alloc::UsageFlags::DOWNLOAD | gpu_alloc::UsageFlags::UPLOAD
        }
    }
}

pub struct AllocatedImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    allocation: Allocation,
    pub extent: vk::Extent3D,
    format: vk::Format,
}

impl AllocatedImage {
    pub fn new(
        device: Device,
        allocator: &mut Allocator,
        extent: vk::Extent3D,
        format: vk::Format,
        image_usages: vk::ImageUsageFlags,
        alloc_usages: AllocUsage,
    ) -> Self {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(image_usages)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&info, None) }.unwrap();
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let allocation = unsafe {
            allocator.alloc(
                AshMemoryDevice::wrap(&device),
                gpu_alloc::Request {
                    size: reqs.size,
                    align_mask: reqs.alignment - 1,
                    usage: alloc_usages.flags(),
                    memory_types: reqs.memory_type_bits,
                },
            ).unwrap()
        };
        log::info!("Creating image {:?} of size {:?} and format {:?}", image, extent, format);
        unsafe { device.bind_image_memory(image, *allocation.memory(), allocation.offset()).unwrap() };

        let view_create_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY)
                .build())
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build());
        let view  = unsafe { device.create_image_view(&view_create_info, None).unwrap() };
        Self {
            image,
            view,
            allocation,
            extent,
            format,
        }
    }

    pub fn destroy(self, device: &Device, allocator: &mut Allocator) {
        log::info!("Destroying image {:?} of size {:?} and format {:?}", self.image, self.extent, self.format);
        unsafe { device.destroy_image_view(self.view, None) };
        unsafe { device.destroy_image(self.image, None) };
        unsafe { allocator.dealloc(AshMemoryDevice::wrap(device), self.allocation) };
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