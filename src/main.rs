mod util;

use ash::extensions::khr::{Surface, Swapchain};

use crate::util::{AllocUsage, AllocatedImage, Allocator, DeletionQueue, DescriptorAllocator, DescriptorLayoutBuilder, PoolSizeRatio, load_shader_module};
use ash::vk::{CommandBuffer, DescriptorSet, DescriptorSetLayout, Image, ImageView, Pipeline, PipelineLayout, ShaderModule, SwapchainKHR};
use ash::{vk, Device, Instance};
use gpu_alloc::GpuAllocator;
use gpu_alloc_ash::device_properties;
use log::{debug, info};
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
    draw_image: AllocatedImage,
    descriptor_allocator: DescriptorAllocator,
    draw_image_descriptor_set: vk::DescriptorSet,
    draw_image_descriptor_set_layout: vk::DescriptorSetLayout,
    gradient_pipeline: vk::Pipeline,
    gradient_pipeline_layout: vk::PipelineLayout,
    gradient_shader: ShaderModule,
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
        let physical_device = Self::pick_physical_device(&instance, &surface, surface_khr);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_and_queue(&instance, &surface, surface_khr, physical_device);

        let config = gpu_alloc::Config::i_am_prototyping();
        let device_properties = unsafe { device_properties(&instance, Self::API_VERSION, physical_device)? };
        let mut allocator = GpuAllocator::new(config, device_properties);

        let capabilities = unsafe { surface.get_physical_device_surface_capabilities(physical_device, surface_khr) }?;
        let ((swapchain, swapchain_khr), swapchain_images, swapchain_views, draw_image) =
            Self::create_swapchain(&instance, &device, surface_khr, capabilities, &mut allocator);
        let frames = Self::init_commands(graphics_queue.1, &device);
        let (descriptor_allocator, draw_image_descriptor_set_layout, draw_image_descriptor_set) = Self::init_descriptors(&device, draw_image.view);
        let (gradient_pipeline_layout, gradient_pipeline, gradient_shader) = Self::init_pipelines(&device, draw_image_descriptor_set_layout);
        
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
            main_deletion_queue: DeletionQueue::default(),
            draw_image,
            descriptor_allocator,
            draw_image_descriptor_set,
            draw_image_descriptor_set_layout,
            gradient_pipeline_layout,
            gradient_pipeline,
            gradient_shader,
        })
    }

    /// Pick the first physical device that supports graphics and presentation queue families.
    fn pick_physical_device(instance: &Instance, surface: &Surface, surface_khr: vk::SurfaceKHR) -> vk::PhysicalDevice {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, surface, surface_khr, *device))
            .expect("No suitable physical device.");

        let props = unsafe { instance.get_physical_device_properties(device) };
        info!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });
        device
    }

    fn is_device_suitable(instance: &Instance, surface: &Surface, surface_khr: vk::SurfaceKHR, device: vk::PhysicalDevice) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        graphics.is_some() && present.is_some()
    }
    fn find_queue_families(
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
    fn create_logical_device_and_queue(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Device, (vk::Queue, u32), (vk::Queue, u32)) {
        let (graphics_family_index, present_family_index) = Self::find_queue_families(instance, surface, surface_khr, device);
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
        let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
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
    ) -> ((Swapchain, SwapchainKHR), Vec<Image>, Vec<ImageView>, AllocatedImage) {
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

    fn init_commands(queue_family_index: u32, device: &Device) -> [FrameData; FRAME_OVERLAP] {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER) // we want to be able to reset individual command buffers, not the entire pool at once
            .queue_family_index(queue_family_index);
        let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder();

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
        })
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
            self.current_frame_mut().deletion_queue.flush();
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
                self.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );

            self.draw_background(cmd_buffer);

            // prepare copying of the draw image to the swapchain image
            util::transition_image(
                &self.device,
                cmd_buffer,
                self.draw_image.image,
                vk::ImageLayout::GENERAL,
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
                self.draw_image.image,
                self.swapchain_images[image_index as usize],
                vk::Extent2D {
                    width: self.draw_image.extent.width,
                    height: self.draw_image.extent.height,
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

    fn draw_background(&self, cmd: CommandBuffer) {
        // let flash = (self.current_frame as f32 / 120.0).sin().abs();
        // let clear_value = vk::ClearColorValue {
        //     float32: [0.0, 0.0, flash, 1.0],
        // };
        // let clear_range = vk::ImageSubresourceRange::builder()
        //     .level_count(vk::REMAINING_MIP_LEVELS)
        //     .layer_count(vk::REMAINING_ARRAY_LAYERS)
        //     .aspect_mask(vk::ImageAspectFlags::COLOR);
        // 
        // unsafe {
        //     self.device.cmd_clear_color_image(
        //         cmd,
        //         self.draw_image.image,
        //         vk::ImageLayout::GENERAL,
        //         &clear_value,
        //         &[clear_range.build()],
        //     );
        // }
        
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.gradient_pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline_layout,
                0,
                &[self.draw_image_descriptor_set],
                &[],
            );
            self.device.cmd_dispatch(cmd, (Self::WIDTH as f32/ 16.0).ceil() as u32, (Self::HEIGHT as f32 / 16.0).ceil() as u32, 1);
        }
    }

    fn init_descriptors(device: &Device, draw_image: ImageView) -> (DescriptorAllocator, DescriptorSetLayout, DescriptorSet) {
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
        (descriptor_pool, layout, descriptor_set)
    }
    
    fn init_pipelines(device: &Device, draw_image_descriptor_set_layout: vk::DescriptorSetLayout) -> (PipelineLayout, Pipeline, ShaderModule) {
        Self::init_background_pipelines(device, draw_image_descriptor_set_layout)
    }

    fn init_background_pipelines(device: &Device, draw_image_descriptor_set_layout: vk::DescriptorSetLayout) -> (PipelineLayout, Pipeline, ShaderModule) {
        let set_layouts = [draw_image_descriptor_set_layout];
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
        let layout = unsafe { device.create_pipeline_layout(&layout_create_info, None).unwrap() };
        let shader = load_shader_module(device, include_bytes!("shaders/spirv/gradient.comp.spv")).expect("Failed to load shader module");
        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(CStr::from_bytes_with_nul(b"main\0").unwrap());
        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*shader_stage_create_info)
            .layout(layout);
        
        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_create_info.build()], None).unwrap()[0] };
        (layout, pipeline, shader)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.main_deletion_queue.flush();
            self.device.destroy_shader_module(self.gradient_shader, None);
            self.device.destroy_pipeline_layout(self.gradient_pipeline_layout, None);
            self.device.destroy_pipeline(self.gradient_pipeline, None);
            
            for frame in self.frames.iter() {
                self.device.destroy_command_pool(frame.command_pool, None);
                self.device.destroy_fence(frame.render_fence, None);
                self.device.destroy_semaphore(frame.swapchain_semaphore, None);
                self.device.destroy_semaphore(frame.render_semaphore, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
            self.swapchain.0.destroy_swapchain(self.swapchain.1, None);
            for view in self.swapchain_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
            self.surface_fn.destroy_surface(self.surface, None);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let event_loop = EventLoop::new();

    let mut app = App::new(&event_loop)?;
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
        Event::RedrawRequested(id) => {
            app.draw();
        }
        _ => {}
    })
}
