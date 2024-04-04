use std::cell::RefCell;
use std::rc::Rc;
use crate::pipeline::Vertex;
use crate::util::{DeletionQueue, transition_image};
use crate::{App, SubmitContext};
use ash::prelude::VkResult;
use ash::vk::{DeviceMemory, DeviceSize};
use ash::{vk, Device};
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
        let allocate_info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts);
        let (pool, descriptor_set) = match unsafe { device.allocate_descriptor_sets(&allocate_info) } {
            Ok(res) => (pool, res[0]),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                self.full_pools.push(pool);
                let new_pool = self.get_or_create_pool(device);
                let new_allocate_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(new_pool)
                    .set_layouts(&layouts);
                (new_pool, unsafe {
                    device
                        .allocate_descriptor_sets(&new_allocate_info)
                        .expect("Failed to allocate descriptor set")[0]
                })
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
        let pool_sizes: Vec<vk::DescriptorPoolSize> = ratios
            .iter()
            .map(|pool_size| vk::DescriptorPoolSize {
                ty: pool_size.descriptor_type,
                descriptor_count: (sets_per_pool as f32 * pool_size.ratio).ceil() as u32,
            })
            .collect();
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(sets_per_pool)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { device.create_descriptor_pool(&info, None).unwrap() };
        debug!("Created descriptor pool {:?}", pool);
        pool
    }
}

pub struct DescriptorBufferWriteInfo {
    pub binding: u32,
    pub array_index: u32,
    pub buffer: vk::Buffer,
    pub size: vk::DeviceSize,
    pub offset: vk::DeviceSize,
    pub ty: vk::DescriptorType,
}

pub struct DescriptorImageWriteInfo {
    pub binding: u32,
    pub array_index: u32,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub layout: vk::ImageLayout,
    pub ty: vk::DescriptorType,
}

pub fn update_set(device: &Device, set: vk::DescriptorSet, image_writes: &[DescriptorImageWriteInfo], buffer_writes: &[DescriptorBufferWriteInfo]) {
    let mut writes = vec![];
    let buffer_infos = buffer_writes.iter().map(|write| [vk::DescriptorBufferInfo {
        buffer: write.buffer,
        offset: write.offset,
        range: write.size,
    }]).collect::<Vec<_>>();
    for (i, write) in buffer_writes.iter().enumerate() {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_binding(write.binding)
                .dst_set(set)
                .dst_array_element(write.array_index)
                .descriptor_type(write.ty)
                .buffer_info(&buffer_infos[i])
        );
    }
    let image_infos = image_writes.iter().map(|write| [vk::DescriptorImageInfo {
        image_view: write.image_view,
        sampler: write.sampler,
        image_layout: write.layout,
    }]).collect::<Vec<_>>();
    for (i, write) in image_writes.iter().enumerate() {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_binding(write.binding)
                .dst_set(set)
                .dst_array_element(write.array_index)
                .descriptor_type(write.ty)
                .image_info(&image_infos[i])
        );
    }

    // println!("Writes: {:#?}", writes);

    unsafe { device.update_descriptor_sets(&writes, &[]) }
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
        let info = vk::BufferCreateInfo::default().size(size).usage(buffer_usages);
        let buffer = unsafe { device.create_buffer(&info, None) }.unwrap();
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = unsafe {
            allocator
                .alloc(
                    AshMemoryDevice::wrap(&device),
                    gpu_alloc::Request {
                        size: reqs.size,
                        align_mask: reqs.alignment - 1,
                        usage: alloc_usages.flags(),
                        memory_types: reqs.memory_type_bits,
                    },
                )
                .unwrap()
        };

        if LOG_ALLOCATIONS {
            debug!(
                "Creating buffer '{}' ({:?}) of size {} B",
                label.clone().unwrap_or_default(),
                buffer,
                size
            );
        }

        unsafe {
            device
                .bind_buffer_memory(buffer, *allocation.memory(), allocation.offset())
                .unwrap()
        };

        Self {
            buffer,
            allocation,
            size,
            label,
        }
    }

    pub(crate) fn destroy(self, device: &Device, allocator: &mut Allocator) {
        if LOG_ALLOCATIONS {
            debug!(
                "Destroying buffer '{}' ({:?}) of size {}",
                self.label.unwrap_or_default(),
                self.buffer,
                self.size
            );
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
            AllocUsage::Shared => {
                gpu_alloc::UsageFlags::HOST_ACCESS
                    | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                    | gpu_alloc::UsageFlags::DOWNLOAD
                    | gpu_alloc::UsageFlags::UPLOAD
            }
            AllocUsage::UploadToHost => gpu_alloc::UsageFlags::DOWNLOAD | gpu_alloc::UsageFlags::UPLOAD,
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
        device: &Device,
        allocator: &mut Allocator,
        extent: vk::Extent3D,
        format: vk::Format,
        image_usages: vk::ImageUsageFlags,
        alloc_usages: AllocUsage,
        image_aspect: vk::ImageAspectFlags,
        label: Option<String>,
    ) -> Self {
        let info = vk::ImageCreateInfo::default()
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
            allocator

                .alloc(
                    AshMemoryDevice::wrap(&device),
                    gpu_alloc::Request {
                        size: reqs.size,
                        align_mask: reqs.alignment - 1,
                        usage: alloc_usages.flags(),
                        memory_types: reqs.memory_type_bits,
                    },
                )
                .unwrap()
        };
        if LOG_ALLOCATIONS {
            debug!(
                "Creating image '{}' ({:?}) of size {:?} and format {:?}",
                label.clone().unwrap_or_default(),
                image,
                extent,
                format
            );
        }
        unsafe { device.bind_image_memory(image, *allocation.memory(), allocation.offset()).unwrap() };

        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(
                vk::ComponentMapping::default()
                    .r(vk::ComponentSwizzle::IDENTITY)
                    .g(vk::ComponentSwizzle::IDENTITY)
                    .b(vk::ComponentSwizzle::IDENTITY)
                    .a(vk::ComponentSwizzle::IDENTITY),
            )
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_aspect)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { device.create_image_view(&view_create_info, None).unwrap() };
        Self {
            image,
            view,
            allocation,
            extent,
            format,
            label,
        }
    }

    // https://i.imgflip.com/8l3uzz.jpg
    pub fn write<'a>(&'a self, data: &'a [u8], ctx: &mut SubmitContext) {
        let mut staging = AllocatedBuffer::new(
            &ctx.device,
            &mut ctx.allocator.borrow_mut(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            AllocUsage::UploadToHost,
            data.len() as DeviceSize,
            Some(format!("Staging buffer for image '{}'", self.label.clone().unwrap_or_default())),
        );
        unsafe {
            let data_ptr = staging
                .allocation
                .map(AshMemoryDevice::wrap(&ctx.device), 0, staging.size as usize)
                .unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr.as_ptr(), data.len());
        }
        transition_image(
            &ctx.device,
            ctx.cmd_buffer,
            self.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_extent(self.extent);
        let copy_regions = [copy_region];
        unsafe {
            ctx.device.cmd_copy_buffer_to_image(
                ctx.cmd_buffer,
                staging.buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            );
        }
        transition_image(
            &ctx.device,
            ctx.cmd_buffer,
            self.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        ctx.cleanup = Some(Box::from(|device: &Device, allocator: &mut Allocator| {
            staging.destroy(device, allocator);
        }))
    }

    pub(crate) fn destroy(self, device: &Device, allocator: &mut Allocator) {
        if LOG_ALLOCATIONS {
            debug!(
                "Destroying image '{}' ({:?}) of size {:?} and format {:?}",
                self.label.unwrap_or_default(),
                self.image,
                self.extent,
                self.format
            );
        }
        unsafe { device.destroy_image_view(self.view, None) };
        unsafe { device.destroy_image(self.image, None) };
        unsafe { allocator.dealloc(AshMemoryDevice::wrap(device), self.allocation) };
    }
}

pub struct Texture {
    pub image: AllocatedImage,
}

pub struct TextureManager {
    pub textures: Vec<Texture>,
    pub samplers: Vec<vk::Sampler>,
}

impl TextureManager {
    pub fn new() -> Self {
        Self {
            textures: vec![],
            samplers: vec![],
        }
    }

    pub fn add_texture(&mut self, texture: AllocatedImage) {
        self.textures.push(Texture { image: texture });
    }

    pub fn destroy(self, device: &Device, allocator: &mut Allocator) {
        for texture in self.textures {
            texture.image.destroy(device, allocator);
        }
    }

    pub fn init_defaults(
        &mut self,
        device: &Device,
        allocator: &mut Allocator,
        bindless_descriptor_set: vk::DescriptorSet,
        deletion_queue: &mut DeletionQueue,
    ) {
        let white_img = AllocatedImage::new(
            device,
            allocator,
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            AllocUsage::GpuOnly,
            vk::ImageAspectFlags::COLOR,
            Some("White Image".into()),
        );
        let black_img = AllocatedImage::new(
            device,
            allocator,
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            AllocUsage::GpuOnly,
            vk::ImageAspectFlags::COLOR,
            Some("Black Image".into()),
        );
        let checkerboard_img = AllocatedImage::new(
            device,
            allocator,
            vk::Extent3D {
                width: 16,
                height: 16,
                depth: 1,
            },
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            AllocUsage::GpuOnly,
            vk::ImageAspectFlags::COLOR,
            Some("Checkerboard Image".into()),
        );
        let sampler_info = vk::SamplerCreateInfo::default()
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);
        let sampler_nearest = unsafe { device.create_sampler(&sampler_info, None).unwrap() };
        let sampler_info = vk::SamplerCreateInfo::default()
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);
        let sampler_linear = unsafe { device.create_sampler(&sampler_info, None).unwrap() };

        update_set(device, bindless_descriptor_set, &[
            DescriptorImageWriteInfo {
                binding: 2,
                array_index: 0,
                image_view: white_img.view,
                sampler: sampler_nearest,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            },
            DescriptorImageWriteInfo {
                binding: 2,
                array_index: 1,
                image_view: black_img.view,
                sampler: sampler_nearest,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            },
            DescriptorImageWriteInfo {
                binding: 2,
                array_index: 2,
                image_view: checkerboard_img.view,
                sampler: sampler_nearest,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            },
        ], &[]);

        self.textures.push(Texture { image: white_img });
        self.textures.push(Texture { image: black_img });
        self.textures.push(Texture { image: checkerboard_img });
        self.samplers.push(sampler_nearest);
        self.samplers.push(sampler_linear);
        deletion_queue.push(move |device| {
            unsafe {
                device.destroy_sampler(sampler_nearest, None);
                device.destroy_sampler(sampler_linear, None);
            }
        });
    }
}
