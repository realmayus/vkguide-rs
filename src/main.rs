mod pipeline;
mod resources;
mod util;

use ash::extensions::khr::{Surface, Swapchain};

use crate::pipeline::{PipelineBuilder, PushConstants, Vertex};
use crate::resources::{AllocUsage, AllocatedBuffer, AllocatedImage, Allocator, DescriptorAllocator, PoolSizeRatio};
use crate::util::{device_discovery, load_shader_module, DeletionQueue, DescriptorLayoutBuilder};
use ash::vk::{CommandBuffer, CommandPool};
use ash::{vk, Device, Instance};
use gpu_alloc::GpuAllocator;
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use log::debug;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::error::Error;
use std::ffi::CStr;
use util::FrameData;
use winit::{
    dpi::PhysicalSize,
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const FRAME_OVERLAP: usize = 2;

struct App {
    entry: ash::Entry,
    instance: Instance,
    surface: vk::SurfaceKHR,
    surface_fn: Surface,
    physical_device: vk::PhysicalDevice,
    device: Device,
    graphics_queue: (vk::Queue, u32),
    present_queue: (vk::Queue, u32),
    swapchain: (Swapchain, vk::SwapchainKHR),
    swapchain_images: Vec<vk::Image>,
    swapchain_views: Vec<vk::ImageView>,
    frames: [FrameData; FRAME_OVERLAP],
    current_frame: u32,
    window: winit::window::Window,
    allocator: Allocator,
    main_deletion_queue: DeletionQueue,
    draw_image: Option<AllocatedImage>,
    descriptor_allocator: DescriptorAllocator,
    draw_image_descriptor_set: vk::DescriptorSet,
    draw_image_descriptor_set_layout: vk::DescriptorSetLayout,
    triangle_pipeline_layout: vk::PipelineLayout,
    triangle_pipeline: vk::Pipeline,
    vertex_shader: vk::ShaderModule,
    frag_shader: vk::ShaderModule,
    immediate_fence: vk::Fence,
    immediate_command_pool: CommandPool,
    immediate_command_buffer: CommandBuffer,
    meshes: Vec<Mesh>,
}

struct Mesh {
    index_buffer: AllocatedBuffer,
    vertex_buffer: AllocatedBuffer,
    vertex_address: vk::DeviceAddress,
}


impl App {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;
    const SWAPCHAIN_IMAGE_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
    const API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);

    fn new(event_loop: &EventLoop<()>) -> Result<Self, Box<dyn Error>> {
        let (instance, surface_khr, surface, entry, window) = unsafe {
            let entry = ash::Entry::load()?;
            let surface_extensions = ash_window::enumerate_required_extensions(event_loop.raw_display_handle())?;
            let app_desc = vk::ApplicationInfo::builder().api_version(Self::API_VERSION);
            let instance_desc = vk::InstanceCreateInfo::builder()
                .application_info(&app_desc)
                .enabled_extension_names(surface_extensions);

            let instance = entry.create_instance(&instance_desc, None)?;

            let window = WindowBuilder::new()
                .with_inner_size(PhysicalSize::<u32>::from((Self::WIDTH, Self::HEIGHT)))
                .with_title("Vulkan Engine")
                .build(event_loop)?;

            // Create a surface from winit window.
            let surface_khr = ash_window::create_surface(&entry, &instance, window.raw_display_handle(), window.raw_window_handle(), None)?;
            let surface = Surface::new(&entry, &instance);
            (instance, surface_khr, surface, entry, window)
        };
        let mut deletion_queue = DeletionQueue::default();
        let physical_device = device_discovery::pick_physical_device(&instance, &surface, surface_khr);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_and_queue(&instance, &surface, surface_khr, physical_device);

        let config = gpu_alloc::Config::i_am_prototyping();
        let device_properties = unsafe { device_properties(&instance, Self::API_VERSION, physical_device)? };
        let mut allocator = GpuAllocator::new(config, device_properties);

        let capabilities = unsafe { surface.get_physical_device_surface_capabilities(physical_device, surface_khr) }?;
        let ((swapchain, swapchain_khr), swapchain_images, swapchain_views, draw_image) =
            Self::create_swapchain(&instance, &device, surface_khr, capabilities, &mut allocator);
        let (immediate_command_pool, immediate_command_buffer, immediate_fence, frames) = Self::init_commands(graphics_queue.1, &device, &mut deletion_queue);
        let (descriptor_allocator, draw_image_descriptor_set_layout, draw_image_descriptor_set) =
            Self::init_descriptors(&device, draw_image.view, &mut deletion_queue);
        let (triangle_pipeline_layout, triangle_pipeline, vertex_shader, frag_shader) =
            Self::init_pipelines(&device, &mut deletion_queue);

        Ok(App {
            entry,
            instance,
            surface: surface_khr,
            surface_fn: surface,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            swapchain: (swapchain, swapchain_khr),
            swapchain_images,
            swapchain_views,
            frames,
            current_frame: 0,
            window,
            allocator,
            main_deletion_queue: deletion_queue,
            draw_image: Some(draw_image),
            descriptor_allocator,
            draw_image_descriptor_set,
            draw_image_descriptor_set_layout,
            vertex_shader,
            frag_shader,
            triangle_pipeline_layout,
            triangle_pipeline,
            immediate_command_pool,
            immediate_command_buffer, 
            immediate_fence,
            meshes: Vec::new(),
        })
    }

    /// Pick the first physical device that supports graphics and presentation queue families.
    fn create_logical_device_and_queue(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Device, (vk::Queue, u32), (vk::Queue, u32)) {
        let (graphics_family_index, present_family_index) = device_discovery::find_queue_families(instance, surface, surface_khr, device);
        let graphics_family_index = graphics_family_index.unwrap();
        let present_family_index = present_family_index.unwrap();
        let queue_priorities = [1.0f32];

        let queue_create_infos = {
            // Vulkan specs does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to
            // deduplicate it.
            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            // Now we build an array of `DeviceQueueCreateInfo`.
            // One for each different family index.
            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect::<Vec<_>>()
        };

        let device_features = vk::PhysicalDeviceFeatures::builder().build();
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .build();
        let mut vk13_features = vk::PhysicalDeviceVulkan13Features::builder()
            .synchronization2(true)
            .dynamic_rendering(true)
            .build();

        let binding = [Swapchain::name().as_ptr()];
        let device_create_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&binding)
            .enabled_features(&device_features)
            .push_next(&mut vk12_features)
            .push_next(&mut vk13_features);
        let device_create_info = device_create_info_builder.build();

        // Build device and queues
        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("Failed to create logical device.")
        };
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

        (
            device,
            (graphics_queue, graphics_family_index),
            (present_queue, present_family_index),
        )
    }

    fn create_swapchain(
        instance: &Instance,
        device: &Device,
        surface_khr: vk::SurfaceKHR,
        capabilities: vk::SurfaceCapabilitiesKHR,
        allocator: &mut Allocator,
    ) -> ((Swapchain, vk::SwapchainKHR), Vec<vk::Image>, Vec<vk::ImageView>, AllocatedImage) {
        let create_info = vk::SwapchainCreateInfoKHR {
            surface: surface_khr,
            image_format: Self::SWAPCHAIN_IMAGE_FORMAT,
            present_mode: vk::PresentModeKHR::FIFO, // hard vsync
            image_extent: vk::Extent2D {
                width: Self::WIDTH,
                height: Self::HEIGHT,
            },
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            image_array_layers: 1,
            min_image_count: capabilities.min_image_count,

            ..Default::default()
        };
        let swapchain = Swapchain::new(instance, device);
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };

        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        debug!("Swapchain images: {:?}", images);
        let image_views = images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(Self::SWAPCHAIN_IMAGE_FORMAT)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&create_info, None).unwrap() }
            })
            .collect::<Vec<_>>();

        let draw_image = AllocatedImage::new(
            device.clone(),
            allocator,
            vk::Extent3D {
                width: Self::WIDTH,
                height: Self::HEIGHT,
                depth: 1,
            },
            Self::SWAPCHAIN_IMAGE_FORMAT,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            AllocUsage::GpuOnly,
        );

        ((swapchain, swapchain_khr), images, image_views, draw_image)
    }

    fn init_commands(queue_family_index: u32, device: &Device, deletion_queue: &mut DeletionQueue) -> (CommandPool, CommandBuffer, vk::Fence, [FrameData; FRAME_OVERLAP]) {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER) // we want to be able to reset individual command buffers, not the entire pool at once
            .queue_family_index(queue_family_index);
        let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

        let immediate_command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None).unwrap() };
        let immediate_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(immediate_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let immediate_command_buffer = unsafe { device.allocate_command_buffers(&immediate_alloc_info).unwrap()[0] };
        let immediate_fence = unsafe { device.create_fence(&fence_create_info, None).unwrap() };
        deletion_queue.push(move |device| {
            unsafe {
                device.destroy_command_pool(immediate_command_pool, None);
                device.destroy_fence(immediate_fence, None);
            }
        });
        (
            immediate_command_pool,
            immediate_command_buffer,
            immediate_fence,
            core::array::from_fn(|_| {
                let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None).unwrap() };
                let command_buffer = unsafe {
                    device
                        .allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::builder()
                                .command_pool(command_pool)
                                .level(vk::CommandBufferLevel::PRIMARY)
                                .command_buffer_count(1),
                        )
                        .unwrap()[0]
                };
                let render_fence = unsafe { device.create_fence(&fence_create_info, None).unwrap() };
                let swapchain_semaphore = unsafe { device.create_semaphore(&semaphore_create_info, None).unwrap() };
                let render_semaphore = unsafe { device.create_semaphore(&semaphore_create_info, None).unwrap() };

                FrameData {
                    command_pool,
                    command_buffer,
                    swapchain_semaphore,
                    render_semaphore,
                    render_fence,
                    deletion_queue: DeletionQueue::default(),
                }
            }),
        )
    }

    fn init_descriptors(device: &Device, draw_image: vk::ImageView, deletion_queue: &mut DeletionQueue) -> (DescriptorAllocator, vk::DescriptorSetLayout, vk::DescriptorSet) {
        let sizes = [PoolSizeRatio {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ratio: 1.0,
        }];
        let descriptor_pool = DescriptorAllocator::new(device, 1, &sizes);
        let builder = DescriptorLayoutBuilder::default().add_binding(0, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE);
        let layout = builder.build(device);
        let descriptor_set = descriptor_pool.allocate(device, layout);
        let img_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(draw_image)
            .build();
        let img_infos = [img_info];
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&img_infos);
        unsafe { device.update_descriptor_sets(&[write.build()], &[]) }
        deletion_queue.push(move |device| {
            unsafe {
                device.destroy_descriptor_set_layout(layout, None);
                device.destroy_descriptor_pool(descriptor_pool.pool, None);
            }
        });
        (descriptor_pool, layout, descriptor_set)
    }

    fn init_pipelines(
        device: &Device,
        deletion_queue: &mut DeletionQueue,
    ) -> (vk::PipelineLayout, vk::Pipeline, vk::ShaderModule, vk::ShaderModule) {
        Self::init_mesh_pipeline(device, deletion_queue)
    }

    fn init_mesh_pipeline(device: &Device, deletion_queue: &mut DeletionQueue) -> (vk::PipelineLayout, vk::Pipeline, vk::ShaderModule, vk::ShaderModule) {
        let vertex_shader =
            load_shader_module(device, include_bytes!("shaders/spirv/mesh.vert.spv")).expect("Failed to load vertex shader module");
        let fragment_shader =
            load_shader_module(device, include_bytes!("shaders/spirv/mesh.frag.spv")).expect("Failed to load fragment shader module");
        let push_constant_range = [*vk::PushConstantRange::builder().offset(0).size(std::mem::size_of::<PushConstants>() as u32).stage_flags(vk::ShaderStageFlags::VERTEX)];
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&[]).push_constant_ranges(&push_constant_range);
        let layout = unsafe { device.create_pipeline_layout(&layout_create_info, None).unwrap() };
        let pipeline_builder = PipelineBuilder {
            layout,
            depth_stencil: *vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(false)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::NEVER)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(Default::default())
                .back(Default::default())
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0),
            render_info: *vk::PipelineRenderingCreateInfo::builder()
                .color_attachment_formats(&[Self::SWAPCHAIN_IMAGE_FORMAT])
                .depth_attachment_format(vk::Format::UNDEFINED),
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
            input_assembly: *vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST),
            rasterization: *vk::PipelineRasterizationStateCreateInfo::builder()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::CLOCKWISE)
                .line_width(1.0),
            color_blend_attachment: *vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            multisample: *vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .min_sample_shading(1.0)
                .alpha_to_coverage_enable(false)
                .alpha_to_one_enable(false),
            color_attachment_format: Self::SWAPCHAIN_IMAGE_FORMAT,
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
        
        (layout, pipeline, vertex_shader, fragment_shader)
    }

    fn upload_mesh(&mut self, indices: &[u32], vertices: &[Vertex]) {
        let vertex_buffer_size = (vertices.len() * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
        let index_buffer_size = (indices.len() * std::mem::size_of::<u32>()) as vk::DeviceSize;
        let vertex_buffer = AllocatedBuffer::new(
            &self.device,
            &mut self.allocator,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            AllocUsage::GpuOnly,
            vertex_buffer_size,
        );
        let device_address_info = vk::BufferDeviceAddressInfo::builder().buffer(vertex_buffer.buffer);
        let buffer_device_address = unsafe { self.device.get_buffer_device_address(&device_address_info) };
        let index_buffer = AllocatedBuffer::new(
            &self.device,
            &mut self.allocator,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            AllocUsage::GpuOnly,
            index_buffer_size,
        );


        let mut staging = AllocatedBuffer::new(
            &self.device,
            &mut self.allocator,
            vk::BufferUsageFlags::TRANSFER_SRC,
            AllocUsage::Shared,
            vertex_buffer_size + index_buffer_size,
        );

        let map = unsafe {
            staging
                .allocation
                .map(AshMemoryDevice::wrap(&self.device), 0, staging.size as usize)
                .unwrap()
        };
        // copy vertex buffer
        let vertex_buffer_ptr = map.as_ptr() as *mut Vertex;
        unsafe {
            vertex_buffer_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
            let index_buffer_ptr = vertex_buffer_ptr.add(vertices.len()) as *mut u32;
            index_buffer_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());
        };
        self.immediate_submit(Box::new(move |this, cmd| {
            let vertex_copy = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: vertex_buffer_size,
            };
            unsafe {
                this.device.cmd_copy_buffer(cmd, staging.buffer, vertex_buffer.buffer, &[vertex_copy]);
            };
            let index_copy = vk::BufferCopy {
                src_offset: vertex_buffer_size,
                dst_offset: 0,
                size: index_buffer_size,
            };
            unsafe {
                this.device.cmd_copy_buffer(cmd, staging.buffer, index_buffer.buffer, &[index_copy]);
            };
        }));
        staging.destroy(&self.device, &mut self.allocator);
        let mesh = Mesh {
            vertex_buffer,
            vertex_address: buffer_device_address,
            index_buffer,
        };
        self.meshes.push(mesh);
    }
    fn immediate_submit(&mut self, cmd: Box<dyn Fn(&mut Self, CommandBuffer)>) {
        let cmd_buffer = self.immediate_command_buffer;
        unsafe {
            self.device.reset_fences(&[self.immediate_fence]).unwrap();
            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd_buffer, &begin_info).unwrap();
            cmd(self, cmd_buffer);
            self.device.end_command_buffer(cmd_buffer).unwrap();
            let cmd_buffer_submit = vk::CommandBufferSubmitInfo::builder().command_buffer(cmd_buffer);
            let cmd_buffer_submits = [*cmd_buffer_submit];
            let submit_info = vk::SubmitInfo2::builder().command_buffer_infos(&cmd_buffer_submits);
            let submits = [*submit_info];
            self.device
                .queue_submit2(self.graphics_queue.0, &submits, self.immediate_fence)
                .unwrap();
            self.device.wait_for_fences(&[self.immediate_fence], true, 1000000000).unwrap();
        }
    }
    fn current_frame(&self) -> &FrameData {
        &self.frames[(self.current_frame % FRAME_OVERLAP as u32) as usize]
    }

    fn current_frame_mut(&mut self) -> &mut FrameData {
        &mut self.frames[(self.current_frame % FRAME_OVERLAP as u32) as usize]
    }

    fn draw(&mut self) {
        unsafe {
            // wait until GPU has finished rendering the last frame, with a timeout of 1
            self.device
                .wait_for_fences(&[self.current_frame().render_fence], true, 1000000000)
                .unwrap();
            let device = self.device.clone();
            self.current_frame_mut().deletion_queue.flush(&device);
            self.device.reset_fences(&[self.current_frame().render_fence]).unwrap();

            // acquire the next image
            let (image_index, _) = self
                .swapchain
                .0
                .acquire_next_image(
                    self.swapchain.1,
                    1000000000,
                    self.current_frame().swapchain_semaphore,
                    vk::Fence::null(),
                )
                .unwrap();
            let cmd_buffer = self.current_frame().command_buffer;
            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            //begin command buffer recording
            let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd_buffer, &begin_info).unwrap();

            // transition draw image into writable mode before rendering. undefined = we don't care, we're fine with the GPU destroying the image. general = general purpose layout which allows reading and writing from the image.
            util::transition_image(
                &self.device,
                cmd_buffer,
                self.draw_image.as_ref().unwrap().image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            self.draw_background(cmd_buffer);

            util::transition_image(
                &self.device,
                cmd_buffer,
                self.draw_image.as_ref().unwrap().image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            self.draw_geometry(cmd_buffer);

            // prepare copying of the draw image to the swapchain image
            util::transition_image(
                &self.device,
                cmd_buffer,
                self.draw_image.as_ref().unwrap().image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            util::transition_image(
                &self.device,
                cmd_buffer,
                self.swapchain_images[image_index as usize],
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            // copy the draw image to the swapchain image
            util::copy_image_to_image(
                &self.device,
                cmd_buffer,
                self.draw_image.as_ref().unwrap().image,
                self.swapchain_images[image_index as usize],
                vk::Extent2D {
                    width: self.draw_image.as_ref().unwrap().extent.width,
                    height: self.draw_image.as_ref().unwrap().extent.height,
                },
                vk::Extent2D {
                    width: Self::WIDTH,
                    height: Self::HEIGHT,
                },
            );
            // transition the swapchain image to present mode
            util::transition_image(
                &self.device,
                cmd_buffer,
                self.swapchain_images[image_index as usize],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            self.device.end_command_buffer(cmd_buffer).unwrap();

            // submit command buffer to queue
            let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::builder().command_buffer(cmd_buffer);
            // we want to wait on the swapchain semaphore, as that signals when the swapchain image is available for rendering
            let wait_info = vk::SemaphoreSubmitInfo::builder()
                .semaphore(self.current_frame().swapchain_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
            // we want to signal the render semaphore, as that signals when the rendering is done
            let signal_info = vk::SemaphoreSubmitInfo::builder()
                .semaphore(self.current_frame().render_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS);
            let submit = vk::SubmitInfo2::builder()
                .command_buffer_infos(&[*cmd_buffer_submit_info])
                .wait_semaphore_infos(&[*wait_info])
                .signal_semaphore_infos(&[*signal_info])
                .build();
            self.device
                .queue_submit2(self.graphics_queue.0, &[submit], self.current_frame().render_fence)
                .unwrap();

            // present the image
            let swapchains = [self.swapchain.1];
            let indices = [image_index];
            let semaphores = [self.current_frame().render_semaphore];
            let present_info = vk::PresentInfoKHR::builder()
                .swapchains(&swapchains)
                .image_indices(&indices)
                .wait_semaphores(&semaphores);
            self.swapchain.0.queue_present(self.present_queue.0, &present_info).unwrap();
        }
        self.current_frame += 1;
    }

    fn draw_background(&self, cmd: vk::CommandBuffer) {
        let clear_range = vk::ImageSubresourceRange::builder()
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(vk::REMAINING_ARRAY_LAYERS)
            .aspect_mask(vk::ImageAspectFlags::COLOR);

        unsafe {
            self.device.cmd_clear_color_image(
                cmd,
                self.draw_image.as_ref().unwrap().image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
                &[clear_range.build()],
            );
        }
    }

    fn draw_geometry(&self, cmd: vk::CommandBuffer) {
        let color_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(self.draw_image.as_ref().unwrap().view)
            .image_layout(vk::ImageLayout::GENERAL);
        let color_attachments = [color_attachment.build()];
        let render_info = vk::RenderingInfo::builder()
            .color_attachments(&color_attachments)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: Self::WIDTH,
                    height: Self::HEIGHT,
                },
            })
            .layer_count(1)
            .view_mask(0);
        unsafe {
            self.device.cmd_begin_rendering(cmd, &render_info);
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.triangle_pipeline);
        };

        let viewport = vk::Viewport::builder()
            .width(Self::WIDTH as f32)
            .height(Self::HEIGHT as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::builder().extent(vk::Extent2D {
            width: Self::WIDTH,
            height: Self::HEIGHT,
        });
        unsafe {
            self.device.cmd_set_viewport(cmd, 0, &[viewport.build()]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor.build()]);
            for mesh in self.meshes.as_slice() {
                let push_constants = PushConstants {
                    world_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
                    vertex_buffer: mesh.vertex_address
                };
                self.device.cmd_push_constants(
                    cmd,
                    self.triangle_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::cast_slice(&[push_constants]),
                );
                self.device.cmd_bind_index_buffer(cmd, mesh.index_buffer.buffer, 0, vk::IndexType::UINT32);
                self.device.cmd_draw_indexed(cmd, 6, 1, 0, 0,0);
            }
            self.device.cmd_end_rendering(cmd);
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.main_deletion_queue.flush(&self.device);
            for mesh in self.meshes.drain(..) {
                mesh.index_buffer.destroy(&self.device, &mut self.allocator);
                mesh.vertex_buffer.destroy(&self.device, &mut self.allocator);
            }
            
            self.draw_image.take().unwrap().destroy(&self.device, &mut self.allocator);

            for frame in self.frames.iter() {
                self.device.destroy_command_pool(frame.command_pool, None);
                self.device.destroy_fence(frame.render_fence, None);
                self.device.destroy_semaphore(frame.swapchain_semaphore, None);
                self.device.destroy_semaphore(frame.render_semaphore, None);
            }
            self.swapchain.0.destroy_swapchain(self.swapchain.1, None);
            for view in self.swapchain_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let event_loop = EventLoop::new();

    let mut app = App::new(&event_loop)?;
    let vertices = [
        Vertex {
            position: [0.5, -0.5, 0.0],
            color: [0.0, 0.0, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
        Vertex {
            position: [0.5, 0.5, 0.0],
            color: [0.5, 0.5, 0.5, 1.0],
            normal: [0.0, 0.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
        Vertex {
            position: [-0.5, -0.5, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        },
        Vertex {
            position: [-0.5, 0.5, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            uv_x: 0.0,
            uv_y: 0.0,
        }
    ];

    let indices = [0, 1, 2, 2, 1, 3];
    app.upload_mesh(&indices, &vertices);
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event:
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                },
            window_id: _,
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::MainEventsCleared => {
            app.window.request_redraw();
        }
        Event::RedrawRequested(..) => {
            app.draw();
        }
        _ => {}
    })
}
