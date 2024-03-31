use ash::{Device, vk};
use ash::prelude::VkResult;
use ash::vk::{DeviceMemory, DeviceSize};
use gpu_alloc_ash::AshMemoryDevice;
use log::debug;

pub type Allocation = gpu_alloc::MemoryBlock<DeviceMemory>;
pub type Allocator = gpu_alloc::GpuAllocator<DeviceMemory>;

const LOG_ALLOCATIONS: bool = false;

#[derive(Clone)]
pub struct PoolSizeRatio {
    pub(crate) descriptor_type: vk::DescriptorType,
    pub(crate) ratio: f32,
}
pub struct DescriptorAllocator {
    ratios: Vec<PoolSizeRatio>,
    ready_pools: Vec<vk::DescriptorPool>,
    full_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}
impl DescriptorAllocator {
    pub fn new(device: &Device, max_sets: u32, pool_sizes: &[PoolSizeRatio]) -> Self {
        let pool = Self::create_pool(device, pool_sizes, max_sets);
        let sets_per_pool = (max_sets as f32 * 1.5) as u32;
        Self {
            ratios: pool_sizes.to_vec(),
            sets_per_pool,
            ready_pools: vec![pool],
            full_pools: vec![],
        }
    }

    pub fn clear_pools(&self, device: &Device) {
        for pool in self.ready_pools.as_slice() {
            unsafe { device.reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty()).unwrap() }
        }
        for pool in self.full_pools.as_slice() {
            unsafe { device.reset_descriptor_pool(*pool, vk::DescriptorPoolResetFlags::empty()).unwrap() }
        }
    }

    pub fn destroy_pools(&self, device: &Device) {
        for pool in self.ready_pools.as_slice() {
            unsafe { device.destroy_descriptor_pool(*pool, None) }
        }
        for pool in self.full_pools.as_slice() {
            unsafe { device.destroy_descriptor_pool(*pool, None) }
        }
    }

    pub fn allocate(&mut self, device: &Device, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let pool = self.get_or_create_pool(device);
        let layouts = [layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let (pool, descriptor_set) = match unsafe { device.allocate_descriptor_sets(&allocate_info) } {
            Ok(res) => (pool, res[0]),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                self.full_pools.push(pool);
                let new_pool = self.get_or_create_pool(device);
                let new_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(new_pool)
                    .set_layouts(&layouts);
                (new_pool, unsafe { device.allocate_descriptor_sets(&new_allocate_info).expect("Failed to allocate descriptor set")[0] })
            }
            Err(e) => panic!("Failed to allocate descriptor set: {:?}", e),
        };
        self.ready_pools.push(pool);
        descriptor_set
    }

    pub fn get_or_create_pool(&mut self, device: &Device) -> vk::DescriptorPool {
        if !self.ready_pools.is_empty() {
            self.ready_pools.pop().unwrap()
        } else {
            let new = Self::create_pool(device, &self.ratios, self.sets_per_pool);
            self.sets_per_pool = (self.sets_per_pool as f32 * 1.5) as u32;
            if self.sets_per_pool > 4092 {
                self.sets_per_pool = 4092;
            }
            new
        }
    }

    fn create_pool(device: &Device, ratios: &[PoolSizeRatio], sets_per_pool: u32) -> vk::DescriptorPool {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = ratios.iter().map(|pool_size| {
            vk::DescriptorPoolSize {
                ty: pool_size.descriptor_type,
                descriptor_count: (sets_per_pool as f32 * pool_size.ratio).ceil() as u32,
            }
        }).collect();
        let info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(sets_per_pool)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.create_descriptor_pool(&info, None).unwrap() };
        debug!("Created descriptor pool {:?}", pool);
        pool
    }
}

#[derive(Default)]
pub struct DescriptorWriter<'a> {
    image_infos: Vec<([vk::DescriptorImageInfo; 1], u32, vk::DescriptorType)>,
    buffer_infos: Vec<([vk::DescriptorBufferInfo; 1], u32, vk::DescriptorType)>,
    writes: Vec<vk::WriteDescriptorSetBuilder<'a>>,
}
impl DescriptorWriter<'_> {
    pub fn write_buffer(&mut self, binding: u32, buffer: vk::Buffer, size: vk::DeviceSize, offset: vk::DeviceSize, ty: vk::DescriptorType) {
        self.buffer_infos.push(([vk::DescriptorBufferInfo::builder()
            .buffer(buffer)
            .offset(offset)
            .range(size).build()],
            binding,
            ty)
        );
    }

    pub fn write_image(&mut self, binding: u32, image_view: vk::ImageView, sampler: vk::Sampler, layout: vk::ImageLayout, ty: vk::DescriptorType) {
        self.image_infos.push(([vk::DescriptorImageInfo::builder()
            .image_view(image_view)
            .sampler(sampler)
            .image_layout(layout).build()], binding, ty)
        );
    }

    pub fn update_set(mut self, device: &Device, set: vk::DescriptorSet) {
        let mut writes = vec![];
        for (buffer_info, binding, ty) in self.buffer_infos {
            writes.push(vk::WriteDescriptorSet::builder()
                                 .dst_binding(binding)
                                 .dst_array_element(0)
                                 .descriptor_type(ty)
                                 .buffer_info(&buffer_info)
                            .build()// TODO this is super sketchy
            );
        }
        for (image_info, binding, ty) in self.image_infos {
            writes.push(vk::WriteDescriptorSet::builder()
                                 .dst_binding(binding)
                                 .dst_array_element(0)
                                 .descriptor_type(ty)
                                 .image_info(&image_info)
                            .build()// TODO this is super sketchy
            );
        }
        
        for write in self.writes.as_mut_slice() {
            write.dst_set = set;
        }
        unsafe { device.update_descriptor_sets(&self.writes.into_iter().map(|x| x.build()).collect::<Vec<_>>(), &[]) }
    }
}


pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub(crate) allocation: Allocation,
    pub(crate) size: DeviceSize,
    pub label: Option<String>,
}

impl AllocatedBuffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        buffer_usages: vk::BufferUsageFlags,
        alloc_usages: AllocUsage,
        size: DeviceSize,
        label: Option<String>,
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
        
        if LOG_ALLOCATIONS {
            debug!("Creating buffer '{}' ({:?}) of size {} B", label.clone().unwrap_or_default(), buffer, size);
        }

        unsafe { device.bind_buffer_memory(buffer, *allocation.memory(), allocation.offset()).unwrap() };

        Self {
            buffer,
            allocation,
            size,
            label,
        }
    }

    pub fn destroy(self, device: &Device, allocator: &mut Allocator) {
        if LOG_ALLOCATIONS {
            debug!("Destroying buffer '{}' ({:?}) of size {}", self.label.unwrap_or_default(), self.buffer, self.size);
        }
        unsafe { device.destroy_buffer(self.buffer, None) };
        unsafe { allocator.dealloc(AshMemoryDevice::wrap(device), self.allocation) };
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
    label: Option<String>,
}

impl AllocatedImage {
    pub fn new(
        device: Device,
        allocator: &mut Allocator,
        extent: vk::Extent3D,
        format: vk::Format,
        image_usages: vk::ImageUsageFlags,
        alloc_usages: AllocUsage,
        image_aspect: vk::ImageAspectFlags,
        label: Option<String>,
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
        if LOG_ALLOCATIONS {
            debug!("Creating image '{}' ({:?}) of size {:?} and format {:?}", label.clone().unwrap_or_default(), image, extent, format);
        }
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
                .aspect_mask(image_aspect)
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
            label,
        }
    }

    pub fn destroy(self, device: &Device, allocator: &mut Allocator) {
        if LOG_ALLOCATIONS {
            debug!("Destroying image '{}' ({:?}) of size {:?} and format {:?}", self.label.unwrap_or_default(), self.image, self.extent, self.format);
        }
        unsafe { device.destroy_image_view(self.view, None) };
        unsafe { device.destroy_image(self.image, None) };
        unsafe { allocator.dealloc(AshMemoryDevice::wrap(device), self.allocation) };
    }
}

