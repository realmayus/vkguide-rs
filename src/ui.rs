use crate::resources::{Allocation, Allocator};
use ash::vk::DeviceMemory;
use ash::{vk, Device};
use egui_winit_ash_integration::Integration;
use gpu_alloc_ash::AshMemoryDevice;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use winit::window::Window;

pub struct AllocationCreateInfo(gpu_alloc::Request);
pub struct ArcAllocator(pub Arc<Mutex<Allocator>>);
pub struct RichAllocation {
    allocation: Option<Allocation>,
    device: Device,
    mapped_ptr: Option<NonNull<c_void>>,
}
pub struct RcRichAllocation(pub Rc<Mutex<RichAllocation>>);

impl egui_winit_ash_integration::AllocatorTrait for ArcAllocator {
    type Allocation = RcRichAllocation;
    type AllocationCreateInfo = AllocationCreateInfo;

    fn allocate(&self, desc: Self::AllocationCreateInfo) -> anyhow::Result<Self::Allocation> {
        unsafe {
            let mut allocator = self.0.lock().unwrap();
            let device = allocator.device.clone();
            Ok(RcRichAllocation(Rc::new(Mutex::new(RichAllocation {
                allocation: Some(Allocation(allocator.allocator.alloc(AshMemoryDevice::wrap(&device), desc.0)?)),
                device: allocator.device.clone(),
                mapped_ptr: None,
            }))))
        }
    }

    fn free(&self, allocation: Self::Allocation) -> anyhow::Result<()> {
        unsafe {
            let mut allocator = self.0.lock().unwrap();
            let device = allocator.device.clone();
            allocator
                .allocator
                .dealloc(AshMemoryDevice::wrap(&device), allocation.0.lock().unwrap().allocation.take().unwrap().0);
        }

        Ok(())
    }
}

impl egui_winit_ash_integration::AllocationTrait for RcRichAllocation {
    unsafe fn memory(&self) -> DeviceMemory {
        *self.0.lock().unwrap().allocation.as_ref().unwrap().0.memory()
    }

    fn offset(&self) -> u64 {
        self.0.lock().unwrap().allocation.as_ref().unwrap().0.offset()
    }

    fn size(&self) -> u64 {
        self.0.lock().unwrap().allocation.as_ref().unwrap().0.size()
    }

    fn mapped_ptr(&self) -> Option<NonNull<c_void>> {
        unsafe {
            let mut sf = self.0.lock().expect("Failed to lock mutex");
            if let Some(map) = sf.mapped_ptr {
                return Some(map);
            }
            let device = sf.device.clone();
            let alloc = sf.allocation.as_mut().expect("Failed to unwrap allocation");
            sf.mapped_ptr = Some(
                alloc
                    .0
                    .map(AshMemoryDevice::wrap(&device), 0, alloc.0.size() as usize)
                    .expect("Failed to map memory")
                    .cast(),
            );
            sf.mapped_ptr
        }
    }
}
impl egui_winit_ash_integration::AllocationCreateInfoTrait for AllocationCreateInfo {
    fn new(requirements: vk::MemoryRequirements, location: egui_winit_ash_integration::MemoryLocation, _linear: bool) -> Self {
        Self(gpu_alloc::Request {
            size: requirements.size,
            align_mask: requirements.alignment - 1,
            usage: if location == egui_winit_ash_integration::MemoryLocation::GpuOnly {
                gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
            } else {
                gpu_alloc::UsageFlags::HOST_ACCESS
                    | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                    | gpu_alloc::UsageFlags::DOWNLOAD
                    | gpu_alloc::UsageFlags::UPLOAD
            },
            memory_types: requirements.memory_type_bits,
        })
    }
}

pub fn draw(integration: &mut Integration<ArcAllocator>, window: &Window, cmd: vk::CommandBuffer, swapchain_image_index: usize) {
    integration.context().set_visuals(egui::Visuals::dark());
    integration.begin_frame(window);
    egui::Window::new("Hello World").show(&integration.context(), |ui| {
        ui.label("it fucking works!");
    });
    let output = integration.end_frame(window);
    let clipped_meshes = integration.context().tessellate(output.shapes);
    integration.paint(cmd, swapchain_image_index, clipped_meshes, output.textures_delta)
}
