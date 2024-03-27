use ash::{Device, vk};
use ash::vk::{BufferCreateFlags, DeviceMemory, DeviceSize};
use gpu_alloc_ash::AshMemoryDevice;

pub type Allocation = gpu_alloc::MemoryBlock<DeviceMemory>;
pub type Allocator = gpu_alloc::GpuAllocator<DeviceMemory>;

pub struct PoolSizeRatio {
    pub(crate) descriptor_type: vk::DescriptorType,
    pub(crate) ratio: f32,
}
pub struct DescriptorAllocator {
    pool: vk::DescriptorPool,
}
impl DescriptorAllocator {
    pub fn new(device: &Device, max_sets: u32, pool_sizes: &[PoolSizeRatio]) -> Self {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = pool_sizes.iter().map(|pool_size| {
            vk::DescriptorPoolSize {
                ty: pool_size.descriptor_type,
                descriptor_count: (max_sets as f32 * pool_size.ratio).ceil() as u32,
            }
        }).collect();
        let info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.create_descriptor_pool(&info, None) }.unwrap();
        Self {
            pool,
        }
    }

    pub fn clear_descriptors(&self, device: &Device) {
        unsafe { device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty()).unwrap() }
    }

    pub fn destroy(self, device: &Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }

    pub fn allocate(&self, device: &Device, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        unsafe { device.allocate_descriptor_sets(&info).unwrap()[0] }
    }
}


pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub(crate) allocation: Allocation,
    pub(crate) size: DeviceSize,
}

impl AllocatedBuffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        buffer_usages: vk::BufferUsageFlags,
        alloc_usages: AllocUsage,
        size: DeviceSize,
    ) -> Self {
        let info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(buffer_usages);
        let buffer = unsafe { device.create_buffer(&info, None) }.unwrap();
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
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

        log::info!("Creating buffer {:?} of size {} B", buffer, size);

        unsafe { device.bind_buffer_memory(buffer, *allocation.memory(), allocation.offset()).unwrap() };

        Self {
            buffer,
            allocation,
            size,
        }
    }

    pub fn destroy(self, device: &Device, allocator: &mut Allocator) {
        log::info!("Destroying buffer {:?} of size {}", self.buffer, self.size);
        unsafe { device.destroy_buffer(self.buffer, None) };
        unsafe { allocator.dealloc(AshMemoryDevice::wrap(device), self.allocation) };
    }
}

pub struct AllocatedImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    allocation: Allocation,
    pub extent: vk::Extent3D,
    format: vk::Format,
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

