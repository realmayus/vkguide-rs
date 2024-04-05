extern crate core;

mod gltf;
mod pipeline;
mod resources;
mod scene;
mod ui;
mod util;

use crate::resources::{
    update_set, AllocUsage, AllocatedBuffer, AllocatedImage, Allocator, DescriptorAllocator, DescriptorBufferWriteInfo,
    DescriptorImageWriteInfo, PoolSizeRatio, TextureManager,
};
use crate::util::{device_discovery, DeletionQueue, DescriptorLayoutBuilder, GpuSceneData};
use ash::khr::swapchain;
use ash::vk::{DescriptorSet, DescriptorSetLayout};
use ash::{khr, vk, Device, Instance};
use glam::Vec3;
use gpu_alloc::GpuAllocator;
use gpu_alloc_ash::{device_properties, AshMemoryDevice};
use log::{debug, info};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::cell::RefCell;
use std::error::Error;
use std::path::Path;
use std::rc::Rc;
use util::FrameData;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use crate::pipeline::egui::EguiPipeline;
use crate::pipeline::mesh::MeshPipeline;
use crate::scene::mesh::Mesh;

const FRAME_OVERLAP: usize = 2;

struct App {
    entry: ash::Entry,
    instance: Instance,
    surface: vk::SurfaceKHR,
    surface_fn: ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    device: Rc<Device>,
    graphics_queue: (vk::Queue, u32),
    present_queue: (vk::Queue, u32),
    swapchain: (swapchain::Device, vk::SwapchainKHR),
    swapchain_images: Vec<vk::Image>,
    swapchain_views: Vec<vk::ImageView>,
    frames: [FrameData; FRAME_OVERLAP],
    current_frame: u32,
    window: winit::window::Window,
    window_size: (u32, u32),
    allocator: Rc<RefCell<Allocator>>,
    main_deletion_queue: DeletionQueue,
    draw_image: Option<AllocatedImage>,
    unorm_draw_image_view: vk::ImageView,
    draw_image_descriptor_set: vk::DescriptorSet,
    draw_image_descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_allocator: DescriptorAllocator,
    bindless_descriptor_pool: vk::DescriptorPool,
    mesh_pipeline: MeshPipeline,
    egui_pipeline: EguiPipeline,
    immediate_fence: vk::Fence,
    immediate_command_pool: vk::CommandPool,
    immediate_command_buffer: vk::CommandBuffer,
    depth_image: Option<AllocatedImage>,
    meshes: Vec<Mesh>,
    scene_data: GpuSceneData,
    scene_data_layout: vk::DescriptorSetLayout,
    texture_manager: TextureManager,
}

struct SubmitContext {
    device: Rc<Device>,
    allocator: Rc<RefCell<Allocator>>,
    fence: vk::Fence,
    cmd_buffer: vk::CommandBuffer,
    queue: vk::Queue,
    cleanup: Option<Box<dyn FnOnce(&Device, &mut Allocator)>>,
}

type ImmediateSubmitFn<'a, T> = Box<(dyn FnOnce(&mut SubmitContext) -> T + 'a)>;

impl SubmitContext {
    fn new(app: &App) -> Self {
        Self {
            device: app.device.clone(),
            allocator: app.allocator.clone(),
            fence: app.immediate_fence,
            cmd_buffer: app.immediate_command_buffer,
            queue: app.graphics_queue.0,
            cleanup: None,
        }
    }

    fn immediate_submit<T>(&mut self, cmd: ImmediateSubmitFn<T>) -> T {
        let cmd_buffer = self.cmd_buffer;
        unsafe {
            self.device.reset_fences(&[self.fence]).unwrap();
            self.device
                .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd_buffer, &begin_info).unwrap();
            let res = cmd(self);
            self.device.end_command_buffer(cmd_buffer).unwrap();
            let cmd_buffer_submit = vk::CommandBufferSubmitInfo::default().command_buffer(cmd_buffer);
            let cmd_buffer_submits = [cmd_buffer_submit];
            let submit_info = vk::SubmitInfo2::default().command_buffer_infos(&cmd_buffer_submits);
            let submits = [submit_info];
            self.device.queue_submit2(self.queue, &submits, self.fence).unwrap();
            self.device.wait_for_fences(&[self.fence], true, 1000000000).unwrap();
            if self.cleanup.is_some() {
                self.cleanup.take().unwrap()(&self.device, &mut self.allocator.borrow_mut());
            }
            res
        }
    }
}

impl Clone for SubmitContext {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            fence: self.fence,
            cmd_buffer: self.cmd_buffer,
            queue: self.queue,
            cleanup: None,
        }
    }
}

pub const SWAPCHAIN_IMAGE_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
pub const API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);

impl App {
    fn new(event_loop: &EventLoop<()>) -> Result<Self, Box<dyn Error>> {
        let window_size = (800, 600);
        let (instance, surface_khr, surface, entry, window) = unsafe {
            let entry = ash::Entry::load()?;
            let surface_extensions = ash_window::enumerate_required_extensions(event_loop.display_handle()?.as_raw())?;
            let app_desc = vk::ApplicationInfo::default().api_version(API_VERSION);
            let instance_desc = vk::InstanceCreateInfo::default()
                .application_info(&app_desc)
                .enabled_extension_names(surface_extensions);

            let instance = entry.create_instance(&instance_desc, None)?;

            let window = WindowBuilder::new()
                .with_inner_size(PhysicalSize::<u32>::from(window_size))
                .with_title("Vulkan Engine")
                .build(event_loop)?;

            // Create a surface from winit window.
            let surface_khr = ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?;
            let surface = khr::surface::Instance::new(&entry, &instance);
            (instance, surface_khr, surface, entry, window)
        };
        let mut deletion_queue = DeletionQueue::default();
        let physical_device = device_discovery::pick_physical_device(&instance, &surface, surface_khr);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_and_queue(&instance, &surface, surface_khr, physical_device);
        let config = gpu_alloc::Config::i_am_prototyping();
        let device_properties = unsafe { device_properties(&instance, API_VERSION, physical_device)? };
        let mut allocator = GpuAllocator::new(config, device_properties);

        let capabilities = unsafe { surface.get_physical_device_surface_capabilities(physical_device, surface_khr) }?;
        let ((swapchain, swapchain_khr), swapchain_images, swapchain_views, draw_image, unorm_draw_image_view, depth_image) =
            Self::create_swapchain(&instance, &device, surface_khr, capabilities, &mut allocator, window_size);
        let (immediate_command_pool, immediate_command_buffer, immediate_fence, frames) =
            Self::init_commands(graphics_queue.1, &device, &mut deletion_queue);
        let (descriptor_allocator, draw_image_descriptor_set_layout, draw_image_descriptor_set, scene_data_layout) =
            Self::init_descriptors(&device, draw_image.view, &mut deletion_queue);
        let (bindless_descriptor_pool, bindless_descriptor_set, bindless_set_layout) = Self::init_bindless(&device);
        let mesh_pipeline = MeshPipeline::new(&device, window_size, &mut deletion_queue, bindless_set_layout);
        let device = Rc::new(device);
        let allocator = Rc::new(RefCell::new(allocator));

        let egui_pipeline = EguiPipeline::new(
            &device,
            window_size,
            &mut deletion_queue,
            bindless_set_layout,
            &window,
            SubmitContext {
                device: device.clone(),
                allocator: allocator.clone(),
                fence: immediate_fence,
                cmd_buffer: immediate_command_buffer,
                queue: graphics_queue.0,
                cleanup: None,
            },
        );
        let texture_manager = SubmitContext {
            device: device.clone(),
            allocator: allocator.clone(),
            fence: immediate_fence,
            cmd_buffer: immediate_command_buffer,
            queue: graphics_queue.0,
            cleanup: None,
        }
        .immediate_submit(Box::new(|ctx| TextureManager::new(bindless_descriptor_set, ctx)));
        info!("Init done.");
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
            window_size,
            allocator,
            main_deletion_queue: deletion_queue,
            draw_image: Some(draw_image), // must be present at all times, Option<_> because we need ownership when destroying
            unorm_draw_image_view,
            depth_image: Some(depth_image),
            descriptor_allocator,
            draw_image_descriptor_set,
            draw_image_descriptor_set_layout,
            bindless_descriptor_pool,
            mesh_pipeline,
            egui_pipeline,
            immediate_command_pool,
            immediate_command_buffer,
            immediate_fence,
            meshes: vec![],
            scene_data_layout,
            scene_data: GpuSceneData {
                view: glam::Mat4::look_at_rh(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
                proj: glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, window_size.0 as f32 / window_size.1 as f32, 0.1, 100.0), // Todo update proj
                viewproj: Default::default(),
                ambient_color: Default::default(),
                sun_dir: Default::default(),
                sun_color: Default::default(),
            },
            texture_manager,
        })
    }

    /// Pick the first physical device that supports graphics and presentation queue families.
    fn create_logical_device_and_queue(
        instance: &Instance,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Device, (vk::Queue, u32), (vk::Queue, u32)) {
        let (graphics_family_index, present_family_index) = device_discovery::find_queue_families(instance, surface, surface_khr, device);
        let graphics_family_index = graphics_family_index.unwrap();
        let present_family_index = present_family_index.unwrap();

        // Vulkan specs does not allow passing an array containing duplicated family indices.
        // And since the family for graphics and presentation could be the same we need to
        // deduplicate it.
        let mut indices = vec![graphics_family_index, present_family_index];
        indices.dedup();

        // Now we build an array of `DeviceQueueCreateInfo`.
        // One for each different family index.
        let queue_priorities = [1.0f32];
        let queue_create_infos = indices
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::default();
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true);
        let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);

        let binding = [khr::swapchain::NAME.as_ptr()];
        let device_create_info_builder = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&binding)
            .enabled_features(&device_features)
            .push_next(&mut vk12_features)
            .push_next(&mut vk13_features);

        // Build device and queues
        let device = unsafe {
            instance
                .create_device(device, &device_create_info_builder, None)
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
        window_size: (u32, u32),
    ) -> (
        (khr::swapchain::Device, vk::SwapchainKHR),
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        AllocatedImage,
        vk::ImageView,
        AllocatedImage,
    ) {
        let create_info = vk::SwapchainCreateInfoKHR {
            surface: surface_khr,
            image_format: SWAPCHAIN_IMAGE_FORMAT,
            present_mode: vk::PresentModeKHR::FIFO, // hard vsync
            image_extent: vk::Extent2D {
                width: window_size.0,
                height: window_size.1,
            },
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            image_array_layers: 1,
            min_image_count: capabilities.min_image_count,

            ..Default::default()
        };
        let swapchain = swapchain::Device::new(instance, device);
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };

        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        debug!("Swapchain images: {:?}", images);
        let image_views = images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(SWAPCHAIN_IMAGE_FORMAT)
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
            device,
            allocator,
            vk::Extent3D {
                width: window_size.0,
                height: window_size.1,
                depth: 1,
            },
            SWAPCHAIN_IMAGE_FORMAT,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                // | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            AllocUsage::GpuOnly,
            vk::ImageAspectFlags::COLOR,
            vk::ImageCreateFlags::MUTABLE_FORMAT,
            Some("Draw Image".into()),
        );

        let unorm_draw_image_view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(draw_image.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::B8G8R8A8_UNORM)
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
                        }),
                    None,
                )
                .unwrap()
        };

        let depth_image = AllocatedImage::new(
            device,
            allocator,
            vk::Extent3D {
                width: window_size.0,
                height: window_size.1,
                depth: 1,
            },
            DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            AllocUsage::GpuOnly,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageCreateFlags::empty(),
            Some("Depth Image".into()),
        );

        (
            (swapchain, swapchain_khr),
            images,
            image_views,
            draw_image,
            unorm_draw_image_view,
            depth_image,
        )
    }

    fn init_commands(
        queue_family_index: u32,
        device: &Device,
        deletion_queue: &mut DeletionQueue,
    ) -> (vk::CommandPool, vk::CommandBuffer, vk::Fence, [FrameData; FRAME_OVERLAP]) {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER) // we want to be able to reset individual command buffers, not the entire pool at once
            .queue_family_index(queue_family_index);
        let fence_create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let immediate_command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None).unwrap() };
        let immediate_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(immediate_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let immediate_command_buffer = unsafe { device.allocate_command_buffers(&immediate_alloc_info).unwrap()[0] };
        let immediate_fence = unsafe { device.create_fence(&fence_create_info, None).unwrap() };
        deletion_queue.push(move |device, allocator| unsafe {
            device.destroy_command_pool(immediate_command_pool, None);
            device.destroy_fence(immediate_fence, None);
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
                            &vk::CommandBufferAllocateInfo::default()
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
                    descriptor_allocator: DescriptorAllocator::new(
                        device,
                        1000,
                        &[
                            PoolSizeRatio {
                                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                                ratio: 3.0,
                            },
                            PoolSizeRatio {
                                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                                ratio: 3.0,
                            },
                            PoolSizeRatio {
                                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                                ratio: 3.0,
                            },
                            PoolSizeRatio {
                                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                ratio: 4.0,
                            },
                        ],
                    ),
                    stale_buffers: Vec::new(),
                }
            }),
        )
    }
    pub const STORAGE_BUFFER_BINDING: u32 = 0;
    pub const STORAGE_IMAGE_BINDING: u32 = 1;
    pub const TEXTURE_BINDING: u32 = 2;
    fn init_bindless(device: &Device) -> (vk::DescriptorPool, DescriptorSet, DescriptorSetLayout) {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 65536,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 65536,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 65536,
            },
        ];
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&pool_sizes)
                        .max_sets(1)
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                    None,
                )
                .unwrap()
        };
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::STORAGE_BUFFER_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(65536)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::STORAGE_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(65536)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(Self::TEXTURE_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(65536)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&bindings)
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .push_next(&mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&[
                            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                        ])),
                    None,
                )
                .unwrap()
        };

        (
            descriptor_pool,
            unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(&[set_layout]),
                    )
                    .unwrap()[0]
            },
            set_layout,
        )
    }

    fn init_descriptors(
        device: &Device,
        draw_image: vk::ImageView,
        deletion_queue: &mut DeletionQueue,
    ) -> (
        DescriptorAllocator,
        vk::DescriptorSetLayout,
        vk::DescriptorSet,
        vk::DescriptorSetLayout,
    ) {
        let sizes = [PoolSizeRatio {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ratio: 1.0,
        }];
        let mut descriptor_pool = DescriptorAllocator::new(device, 1, &sizes);
        let builder = DescriptorLayoutBuilder::default().add_binding(0, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE);
        let layout = builder.build(device);
        let descriptor_set = descriptor_pool.allocate(device, layout);

        // update_set(device, descriptor_set, &[
        //     DescriptorImageWriteInfo {
        //         binding: 0,
        //         array_index: 0,
        //         image_view: draw_image,
        //         sampler: vk::Sampler::null(),
        //         layout: vk::ImageLayout::GENERAL,
        //         ty: vk::DescriptorType::STORAGE_IMAGE,
        //     },
        // ], &[]);

        let scene_info_builder = DescriptorLayoutBuilder::default().add_binding(
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        );
        let scene_info_layout = scene_info_builder.build(device);
        deletion_queue.push(move |device, allocator| unsafe {
            device.destroy_descriptor_set_layout(layout, None);
            device.destroy_descriptor_set_layout(scene_info_layout, None);
        });
        (descriptor_pool, layout, descriptor_set, scene_info_layout)
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.swapchain.0.destroy_swapchain(self.swapchain.1, None);
        for view in self.swapchain_views.drain(..) {
            self.device.destroy_image_view(view, None);
        }
    }

    fn resize(&mut self, size: (u32, u32)) {
        debug!("Resizing to {:?}", size);
        self.resize_swapchain(size);
        self.mesh_pipeline.resize(size);
        self.egui_pipeline.resize(size);
    }
    fn resize_swapchain(&mut self, size: (u32, u32)) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.destroy_swapchain();
            self.draw_image
                .take()
                .unwrap()
                .destroy(&self.device, &mut self.allocator.borrow_mut());
            self.depth_image
                .take()
                .unwrap()
                .destroy(&self.device, &mut self.allocator.borrow_mut());
        }
        self.window_size = size;
        let capabilities = unsafe {
            self.surface_fn
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap()
        };
        let (swapchain, swapchain_images, swapchain_views, draw_image, unorm_draw_image_view, depth_image) = Self::create_swapchain(
            &self.instance,
            &self.device,
            self.surface,
            capabilities,
            &mut self.allocator.borrow_mut(),
            self.window_size,
        );
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_views = swapchain_views;
        self.draw_image = Some(draw_image);
        self.depth_image = Some(depth_image);
        self.unorm_draw_image_view = unorm_draw_image_view;
    }

    // https://i.imgflip.com/8l3uzz.jpg
    fn current_frame(&self) -> &FrameData {
        &self.frames[(self.current_frame % FRAME_OVERLAP as u32) as usize]
    }

    fn draw(&mut self) {
        unsafe {
            // wait until GPU has finished rendering the last frame, with a timeout of 1
            self.device
                .wait_for_fences(&[self.current_frame().render_fence], true, 1000000000)
                .unwrap();
            let device = self.device.clone();
            frame!(self).deletion_queue.flush(&device, &mut self.allocator.borrow_mut());
            frame!(self).descriptor_allocator.clear_pools(&device);
            for buffer in frame!(self).stale_buffers.drain(..) {
                buffer.destroy(&device, &mut self.allocator.borrow_mut());
            }
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
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
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
            util::transition_image(
                &self.device,
                cmd_buffer,
                self.depth_image.as_ref().unwrap().image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            );

            let mut scene_data_buffer = AllocatedBuffer::new(
                &self.device,
                &mut self.allocator.borrow_mut(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                AllocUsage::Shared,
                std::mem::size_of::<GpuSceneData>() as vk::DeviceSize,
                Some("Scene Data Buffer".into()),
            );

            let map = scene_data_buffer
                .allocation
                .map(AshMemoryDevice::wrap(&self.device), 0, std::mem::size_of::<GpuSceneData>())
                .unwrap();
            // copy scene data to buffer
            let scene_data_ptr = map.as_ptr() as *mut GpuSceneData;
            scene_data_ptr.copy_from_nonoverlapping(&self.scene_data, 1);
            let global_descriptor = frame!(self).descriptor_allocator.allocate(&self.device, self.scene_data_layout);

            update_set(
                &self.device,
                global_descriptor,
                &[],
                &[DescriptorBufferWriteInfo {
                    binding: 0,
                    array_index: 0,
                    buffer: scene_data_buffer.buffer,
                    size: std::mem::size_of::<GpuSceneData>() as vk::DeviceSize,
                    offset: 0,
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                }],
            );

            frame!(self).stale_buffers.push(scene_data_buffer);

            self.mesh_pipeline.draw(
                &self.device,
                cmd_buffer,
                &self.meshes,
                self.draw_image.as_ref().unwrap().view,
                self.depth_image.as_ref().unwrap().view,
                self.texture_manager.descriptor_set(),
            );

            let ctx = SubmitContext::new(self);
            self.egui_pipeline.begin_frame(&self.window);
            let egui_ctx = self.egui_pipeline.context();
            egui::Window::new("Hello world!").show(egui_ctx, |ui| {
                ui.label("Hello World!");
                ui.add(egui::Slider::new(&mut self.scene_data.sun_dir.x, 69.0..=420.0).text("AWESOMENESS"));
                if ui.button("FINALLY!!!!!").clicked() {
                    debug!("huge W");
                }
                ui.colored_label(egui::Color32::RED, "This is red text");
                ui.colored_label(egui::Color32::GREEN, "This is green text");
                ui.colored_label(egui::Color32::BLUE, "This is blue text");
                ui.label("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse");
            });

            let output = self.egui_pipeline.end_frame(&self.window);
            let meshes = self
                .egui_pipeline
                .context()
                .tessellate(output.shapes, self.window.scale_factor() as f32);

            self.egui_pipeline.draw(
                &self.device,
                cmd_buffer,
                self.unorm_draw_image_view,
                self.depth_image.as_ref().unwrap().view,
                self.texture_manager.descriptor_set(),
                output.textures_delta,
                meshes,
                &mut self.texture_manager,
                ctx,
                (self.current_frame % FRAME_OVERLAP as u32) as usize,
                &self.window,
            );

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
                    width: self.window_size.0,
                    height: self.window_size.1,
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
            let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd_buffer);
            // we want to wait on the swapchain semaphore, as that signals when the swapchain image is available for rendering
            let wait_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.current_frame().swapchain_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
            // we want to signal the render semaphore, as that signals when the rendering is done
            let signal_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.current_frame().render_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS);
            let command_buffer_infos = [cmd_buffer_submit_info];
            let wait_infos = [wait_info];
            let signal_semaphore_infos = [signal_info];
            let submit = vk::SubmitInfo2::default()
                .command_buffer_infos(&command_buffer_infos)
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_semaphore_infos);
            self.device
                .queue_submit2(self.graphics_queue.0, &[submit], self.current_frame().render_fence)
                .unwrap();

            // present the image
            let swapchains = [self.swapchain.1];
            let indices = [image_index];
            let semaphores = [self.current_frame().render_semaphore];
            let present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchains)
                .image_indices(&indices)
                .wait_semaphores(&semaphores);
            self.swapchain.0.queue_present(self.present_queue.0, &present_info).unwrap();
        }
        self.current_frame = self.current_frame.wrapping_add(1);
    }

    fn draw_background(&self, cmd: vk::CommandBuffer) {
        let clear_range = vk::ImageSubresourceRange::default()
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
                &[clear_range],
            );
        }
    }

    fn input(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.egui_pipeline.input(&self.window, event)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.main_deletion_queue.flush(&self.device, &mut self.allocator.borrow_mut());
            self.descriptor_allocator.destroy_pools(&self.device);
            self.device.destroy_descriptor_pool(self.bindless_descriptor_pool, None);
            for mut mesh in self.meshes.drain(..) {
                mesh.destroy(&self.device, &mut self.allocator.borrow_mut());
            }

            self.device.destroy_image_view(self.unorm_draw_image_view, None);
            self.draw_image
                .take()
                .unwrap()
                .destroy(&self.device, &mut self.allocator.borrow_mut());
            self.depth_image
                .take()
                .unwrap()
                .destroy(&self.device, &mut self.allocator.borrow_mut());
            self.texture_manager.destroy(&self.device, &mut self.allocator.borrow_mut());
            for frame in self.frames.iter_mut() {
                frame.deletion_queue.flush(&self.device, &mut self.allocator.borrow_mut()); // take care of frames that were prepared but not reached in the render loop yet
                for buffer in frame.stale_buffers.drain(..) {
                    buffer.destroy(&self.device, &mut self.allocator.borrow_mut());
                }

                self.device.destroy_command_pool(frame.command_pool, None);
                self.device.destroy_fence(frame.render_fence, None);
                self.device.destroy_semaphore(frame.swapchain_semaphore, None);
                self.device.destroy_semaphore(frame.render_semaphore, None);
                frame.descriptor_allocator.destroy_pools(&self.device);
            }
            self.egui_pipeline.destroy(&self.device, &mut self.allocator.borrow_mut());
            self.destroy_swapchain();
            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let event_loop = EventLoop::new();

    let mut app = App::new(event_loop.as_ref().unwrap())?;
    let models = gltf::load_gltf(Path::new("src/assets/cube_light_tan.glb"));
    let meshes = models.into_iter().flat_map(|mo| mo.meshes).collect::<Vec<_>>();
    info!("Loaded gltf with {} meshes", meshes.len());

    for mut mesh in meshes {
        let mut submit_ctx = SubmitContext::new(&app);
        submit_ctx.immediate_submit(Box::new(|ctx| mesh.upload(ctx)));

        app.meshes.push(mesh);
    }

    Ok(event_loop.unwrap().run(move |event, target| match event {
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => {
            app.draw();
            app.window.request_redraw();
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            app.resize((size.width, size.height));
        }
        Event::WindowEvent {
            event:
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    event:
                        winit::event::KeyEvent {
                            logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                            ..
                        },
                    ..
                },
            window_id: _,
        } => {
            target.exit();
        }

        Event::WindowEvent { event, .. } => {
            app.input(&event);
        }
        _ => {}
    })?)
}
