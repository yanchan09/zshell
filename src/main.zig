const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

const wayland = @import("wayland");
const wl = wayland.client.wl;
const zwlr = wayland.client.zwlr;

const vk = @import("vulkan");
const vkalloc = @import("./vkalloc.zig");

const tracy = @import("./tracy.zig");

// linked from Vulkan-Loader (libvulkan.so)
extern fn vkGetInstanceProcAddr(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;

const vma = @cImport({
    @cInclude("vk_mem_alloc.h");
});

const RegistryContext = struct {
    shm: ?*wl.Shm = null,
    compositor: ?*wl.Compositor = null,
    layer_shell: ?*zwlr.LayerShellV1 = null,

    fn listener(registry: *wl.Registry, event: wl.Registry.Event, context: *RegistryContext) void {
        switch (event) {
            .global => |global| {
                if (std.mem.orderZ(u8, global.interface, wl.Shm.getInterface().name) == .eq) {
                    context.shm = registry.bind(global.name, wl.Shm, 1) catch return;
                } else if (std.mem.orderZ(u8, global.interface, wl.Compositor.getInterface().name) == .eq) {
                    context.compositor = registry.bind(global.name, wl.Compositor, 1) catch return;
                } else if (std.mem.orderZ(u8, global.interface, zwlr.LayerShellV1.getInterface().name) == .eq) {
                    context.layer_shell = registry.bind(global.name, zwlr.LayerShellV1, 1) catch return;
                }
            },
            .global_remove => {},
        }
    }
};

fn layerSurfaceListener(surface: *zwlr.LayerSurfaceV1, event: zwlr.LayerSurfaceV1.Event, _: *void) void {
    switch (event) {
        .configure => |configure| {
            std.log.info("zwlr_layer_surface_v1: configure(serial={}, width={}, height={})", .{
                configure.serial, configure.width, configure.height,
            });
            surface.ackConfigure(configure.serial);
        },
        .closed => {
            std.log.info("zwlr_layer_surface_v1: closed", .{});
        },
    }
}

const BaseDispatch = vk.BaseWrapper(.{
    .enumerateInstanceVersion = true,
    .createInstance = true,
    .getInstanceProcAddr = true,
});

const InstanceDispatch = vk.InstanceWrapper(.{
    .destroyInstance = true,
    .createWaylandSurfaceKHR = true,
    .destroySurfaceKHR = true,
    .enumeratePhysicalDevices = true,
    .getPhysicalDeviceProperties = true,
    .getPhysicalDeviceQueueFamilyProperties = true,
    .createDevice = true,
    .getDeviceProcAddr = true,
    .getPhysicalDeviceSurfaceSupportKHR = true,
    .getPhysicalDeviceSurfaceFormatsKHR = true,
    .getPhysicalDeviceSurfaceCapabilitiesKHR = true,
});

const DeviceDispatch = vk.DeviceWrapper(.{
    .destroyDevice = true,
    .createSwapchainKHR = true,
    .destroySwapchainKHR = true,
    .getSwapchainImagesKHR = true,
    .createImageView = true,
    .destroyImageView = true,
    .createShaderModule = true,
    .createPipelineLayout = true,
    .createRenderPass = true,
    .createGraphicsPipelines = true,
    .createFramebuffer = true,
    .createCommandPool = true,
    .allocateCommandBuffers = true,
    .beginCommandBuffer = true,
    .cmdBeginRenderPass = true,
    .cmdSetViewport = true,
    .cmdSetScissor = true,
    .cmdBindPipeline = true,
    .cmdDraw = true,
    .cmdEndRenderPass = true,
    .endCommandBuffer = true,
    .queueSubmit = true,
    .acquireNextImageKHR = true,
    .queuePresentKHR = true,
    .getDeviceQueue = true,
    .createSemaphore = true,
    .createFence = true,
    .waitForFences = true,
    .resetFences = true,
    .resetCommandBuffer = true,
    .cmdBindVertexBuffers = true,
});

const Vertex = packed struct {
    x: f32,
    y: f32,

    pub fn binding() vk.VertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(@This()),
            .input_rate = .vertex,
        };
    }

    pub fn attributes() []const vk.VertexInputAttributeDescription {
        return &[_]vk.VertexInputAttributeDescription{
            .{
                .binding = 0,
                .location = 0,
                .format = .r32g32_sfloat,
                .offset = 0,
            },
        };
    }
};

pub const frag_spv align(@alignOf(u32)) = @embedFile("frag.spv").*;
pub const vert_spv align(@alignOf(u32)) = @embedFile("vert.spv").*;

const trace_zone_render = tracy.PooledString.pool("Render");

pub fn main() !void {
    const t = tracy.Tracy.init();
    defer t.deinit();

    var allocator = std.heap.c_allocator;

    var vkb = try BaseDispatch.load(vkGetInstanceProcAddr);
    const version = try vkb.enumerateInstanceVersion();
    std.log.info("vkEnumerateInstanceVersion: {}.{}.{}.{}", .{
        vk.apiVersionVariant(version),
        vk.apiVersionMajor(version),
        vk.apiVersionMinor(version),
        vk.apiVersionPatch(version),
    });

    const instance = try vkb.createInstance(&.{
        .p_application_info = &.{
            .p_application_name = "zshell",
            .application_version = vk.makeApiVersion(0, 0, 0, 0),
            .p_engine_name = "zshell",
            .engine_version = vk.makeApiVersion(0, 0, 0, 0),
            .api_version = vk.API_VERSION_1_0,
        },
        .enabled_extension_count = 2,
        .pp_enabled_extension_names = &[_][*:0]const u8{
            "VK_KHR_surface",
            "VK_KHR_wayland_surface",
        },
        .enabled_layer_count = 1,
        .pp_enabled_layer_names = &[_][*:0]const u8{
            "VK_LAYER_KHRONOS_validation",
        },
    }, null);

    const vki = try InstanceDispatch.load(instance, vkb.dispatch.vkGetInstanceProcAddr);
    defer vki.destroyInstance(instance, null);

    const display = try wl.Display.connect(null);
    const registry = try display.getRegistry();

    var context = RegistryContext{};
    registry.setListener(*RegistryContext, RegistryContext.listener, &context);
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    const compositor = context.compositor orelse return error.MissingProtocol;
    const layer_shell = context.layer_shell orelse return error.MissingProtocol;

    const surface = try compositor.createSurface();
    defer surface.destroy();

    const layer_surface = try layer_shell.getLayerSurface(surface, null, .bottom, "zshell");
    defer layer_surface.destroy();

    var dummy = void{};
    layer_surface.setListener(*void, layerSurfaceListener, &dummy);

    layer_surface.setSize(0, 32);
    layer_surface.setAnchor(.{ .bottom = true });
    layer_surface.setExclusiveZone(32);
    surface.commit();
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    const vk_surface = try vki.createWaylandSurfaceKHR(instance, &.{
        .display = @ptrCast(display),
        .surface = @ptrCast(surface),
    }, null);
    defer vki.destroySurfaceKHR(instance, vk_surface, null);

    var device_count: u32 = undefined;
    _ = try vki.enumeratePhysicalDevices(instance, &device_count, null);

    const pdevs = try allocator.alloc(vk.PhysicalDevice, device_count);
    defer allocator.free(pdevs);

    _ = try vki.enumeratePhysicalDevices(instance, &device_count, pdevs.ptr);

    for (pdevs, 0..) |dev, i| {
        const props = vki.getPhysicalDeviceProperties(dev);
        std.log.info("Physical device #{}: {s}", .{ i, props.device_name });

        var property_count: u32 = undefined;
        vki.getPhysicalDeviceQueueFamilyProperties(dev, &property_count, null);

        const queues = try allocator.alloc(vk.QueueFamilyProperties, property_count);
        defer allocator.free(queues);

        vki.getPhysicalDeviceQueueFamilyProperties(dev, &property_count, queues.ptr);

        for (queues, 0..) |queue, j| {
            const can_present = try vki.getPhysicalDeviceSurfaceSupportKHR(dev, @intCast(j), vk_surface);
            std.log.info("- Queue family #{}: flags={}, count={}, can_present={}", .{ j, queue.queue_flags, queue.queue_count, can_present });
        }
    }

    var surface_format_count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdevs[0], vk_surface, &surface_format_count, null);

    const surface_formats = try allocator.alloc(vk.SurfaceFormatKHR, surface_format_count);

    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdevs[0], vk_surface, &surface_format_count, surface_formats.ptr);

    std.log.info("Supported surface formats:", .{});
    for (surface_formats) |fmt| {
        std.log.info("- {} ({})", .{ fmt.format, fmt.color_space });
    }

    const surface_caps = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(pdevs[0], vk_surface);
    std.log.info("Surface capabilities:", .{});
    std.log.info("- Image count: min={}, max={}", .{ surface_caps.min_image_count, surface_caps.max_image_count });
    std.log.info("- Extents: min={}x{}, max={}x{}, current={}x{}", .{
        surface_caps.min_image_extent.width,
        surface_caps.min_image_extent.height,
        surface_caps.max_image_extent.width,
        surface_caps.max_image_extent.height,
        surface_caps.current_extent.width,
        surface_caps.current_extent.height,
    });
    std.log.info("- Supported transforms: {}", .{surface_caps.supported_transforms});
    std.log.info("- Alpha: {}", .{surface_caps.supported_composite_alpha});

    const dev = try vki.createDevice(pdevs[0], &.{
        .p_queue_create_infos = &[_]vk.DeviceQueueCreateInfo{
            .{
                .queue_family_index = 0,
                .queue_count = 1,
                .p_queue_priorities = &[_]f32{1.0},
            },
        },
        .queue_create_info_count = 1,
        .p_enabled_features = &.{},
        .enabled_extension_count = 1,
        .pp_enabled_extension_names = &[_][*:0]const u8{
            "VK_KHR_swapchain",
        },
        .enabled_layer_count = 0,
        .pp_enabled_layer_names = &[_][*:0]const u8{
            "VK_LAYER_KHRONOS_validation",
        },
    }, null);
    const vkd = try DeviceDispatch.load(dev, vki.dispatch.vkGetDeviceProcAddr);
    defer vkd.destroyDevice(dev, null);

    const vk_allocator = try vkalloc.Allocator.create(.{
        .physical_device = pdevs[0],
        .device = dev,
        .instance = instance,
    });
    defer vk_allocator.deinit();

    const msaa_image = try vk_allocator.createImage(&vk.ImageCreateInfo{
        .image_type = .@"2d",
        .format = .b8g8r8a8_srgb,
        .extent = .{ .width = 1920, .height = 32, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"4_bit" = true },
        .tiling = .optimal,
        .usage = .{ .color_attachment_bit = true },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, .{
        .usage = .gpu_lazily_allocated,
    });
    defer msaa_image.destroy();

    const msaa_image_view = try vkd.createImageView(dev, &.{
        .image = msaa_image.deref(),
        .view_type = .@"2d",
        .format = .b8g8r8a8_srgb,
        .components = .{
            .r = .identity,
            .g = .identity,
            .b = .identity,
            .a = .identity,
        },
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
    }, null);

    const swapchain = try vkd.createSwapchainKHR(dev, &.{
        .surface = vk_surface,
        .min_image_count = 4,
        .image_format = .b8g8r8a8_srgb,
        .image_color_space = .srgb_nonlinear_khr,
        .image_extent = .{ .width = 1920, .height = 32 },
        .image_array_layers = 1,
        .image_usage = .{ .color_attachment_bit = true },
        .image_sharing_mode = .exclusive,
        .pre_transform = .{ .identity_bit_khr = true },
        .composite_alpha = .{ .pre_multiplied_bit_khr = true },
        .present_mode = .fifo_khr,
        .clipped = vk.TRUE,
    }, null);
    defer vkd.destroySwapchainKHR(dev, swapchain, null);

    const render_pass = try vkd.createRenderPass(dev, &.{
        .attachment_count = 2,
        .p_attachments = &[_]vk.AttachmentDescription{
            // Multisampled attachment
            .{
                .format = .b8g8r8a8_srgb,
                .samples = .{ .@"4_bit" = true },
                .load_op = .clear,
                .store_op = .dont_care,
                .stencil_load_op = .dont_care,
                .stencil_store_op = .dont_care,
                .initial_layout = .undefined,
                .final_layout = .color_attachment_optimal,
            },
            // Swapchain attachment
            .{
                .format = .b8g8r8a8_srgb,
                .samples = .{ .@"1_bit" = true },
                .load_op = .dont_care,
                .store_op = .store,
                .stencil_load_op = .dont_care,
                .stencil_store_op = .dont_care,
                .initial_layout = .undefined,
                .final_layout = .present_src_khr,
            },
        },
        .subpass_count = 1,
        .p_subpasses = &[_]vk.SubpassDescription{
            .{
                .color_attachment_count = 1,
                .p_color_attachments = &[_]vk.AttachmentReference{
                    // Render to multisampled attachment
                    .{
                        .attachment = 0,
                        .layout = .color_attachment_optimal,
                    },
                },
                .p_resolve_attachments = &[_]vk.AttachmentReference{
                    // Resolve multisampled attachment to swapchain image
                    .{
                        .attachment = 1,
                        .layout = .color_attachment_optimal,
                    },
                },
                .pipeline_bind_point = .graphics,
            },
        },
    }, null);

    var image_count: u32 = undefined;
    _ = try vkd.getSwapchainImagesKHR(dev, swapchain, &image_count, null);

    const images = try allocator.alloc(vk.Image, image_count);
    defer allocator.free(images);

    _ = try vkd.getSwapchainImagesKHR(dev, swapchain, &image_count, images.ptr);

    const image_views = try allocator.alloc(vk.ImageView, image_count);
    defer allocator.free(image_views);

    const framebuffers = try allocator.alloc(vk.Framebuffer, image_count);
    defer allocator.free(framebuffers);

    for (images, 0..) |image, i| {
        image_views[i] = try vkd.createImageView(dev, &.{
            .image = image,
            .view_type = .@"2d",
            .format = .b8g8r8a8_srgb,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        framebuffers[i] = try vkd.createFramebuffer(dev, &.{
            .render_pass = render_pass,
            .attachment_count = 2,
            .p_attachments = &[_]vk.ImageView{ msaa_image_view, image_views[i] },
            .width = 1920,
            .height = 32,
            .layers = 1,
        }, null);
    }
    defer for (image_views) |view| vkd.destroyImageView(dev, view, null);

    const shader_vert = try vkd.createShaderModule(dev, &.{
        .code_size = vert_spv.len,
        .p_code = @ptrCast(&vert_spv),
    }, null);
    const shader_frag = try vkd.createShaderModule(dev, &.{
        .code_size = frag_spv.len,
        .p_code = @ptrCast(&frag_spv),
    }, null);

    const pipeline_layout = try vkd.createPipelineLayout(dev, &.{}, null);
    var pipelines: [1]vk.Pipeline = undefined;
    _ = try vkd.createGraphicsPipelines(dev, .null_handle, 1, &[_]vk.GraphicsPipelineCreateInfo{
        .{
            .stage_count = 2,
            .p_stages = &[_]vk.PipelineShaderStageCreateInfo{
                .{
                    .stage = .{ .vertex_bit = true },
                    .module = shader_vert,
                    .p_name = "main",
                },
                .{
                    .stage = .{ .fragment_bit = true },
                    .module = shader_frag,
                    .p_name = "main",
                },
            },
            .p_vertex_input_state = &.{
                .vertex_binding_description_count = 1,
                .vertex_attribute_description_count = @intCast(Vertex.attributes().len),
                .p_vertex_binding_descriptions = &[_]vk.VertexInputBindingDescription{Vertex.binding()},
                .p_vertex_attribute_descriptions = Vertex.attributes().ptr,
            },
            .p_input_assembly_state = &.{
                .topology = .triangle_list,
                .primitive_restart_enable = vk.FALSE,
            },
            .p_viewport_state = &.{
                .viewport_count = 1,
                .scissor_count = 1,
            },
            .p_rasterization_state = &.{
                .depth_clamp_enable = vk.FALSE,
                .rasterizer_discard_enable = vk.FALSE,
                .polygon_mode = .fill,
                .front_face = .clockwise,
                .depth_bias_enable = vk.FALSE,
                .depth_bias_constant_factor = 0.0,
                .depth_bias_clamp = 0.0,
                .depth_bias_slope_factor = 0.0,
                .line_width = 1.0,
            },
            .p_multisample_state = &.{
                .rasterization_samples = .{ .@"4_bit" = true },
                .sample_shading_enable = vk.FALSE,
                .min_sample_shading = 1.0,
                .alpha_to_coverage_enable = vk.FALSE,
                .alpha_to_one_enable = vk.FALSE,
            },
            .p_color_blend_state = &.{
                .logic_op_enable = vk.FALSE,
                .logic_op = .copy,
                .attachment_count = 1,
                .p_attachments = &[_]vk.PipelineColorBlendAttachmentState{
                    .{
                        // should be this for blending with premultiplied alpha
                        .blend_enable = vk.TRUE,
                        .src_color_blend_factor = .one,
                        .dst_color_blend_factor = .one_minus_src_alpha,
                        .color_blend_op = .add,
                        .src_alpha_blend_factor = .one,
                        .dst_alpha_blend_factor = .one_minus_src_alpha,
                        .alpha_blend_op = .add,
                        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
                    },
                },
                .blend_constants = .{ 1.0, 1.0, 1.0, 1.0 },
            },
            .p_dynamic_state = &.{
                .dynamic_state_count = 2,
                .p_dynamic_states = &[_]vk.DynamicState{ .viewport, .scissor },
            },
            .layout = pipeline_layout,
            .render_pass = render_pass,
            .subpass = 0,
            .base_pipeline_index = -1,
        },
    }, null, &pipelines);

    const command_pool = try vkd.createCommandPool(dev, &.{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = 0,
    }, null);

    var command_buffers: [1]vk.CommandBuffer = undefined;
    _ = try vkd.allocateCommandBuffers(dev, &.{
        .command_pool = command_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, &command_buffers);

    const queue = vkd.getDeviceQueue(dev, 0, 0);

    const vertices = [_]Vertex{
        .{ .x = -1, .y = -1 },
        .{ .x = 1, .y = -1 },
        .{ .x = -1, .y = 1 },
        .{ .x = -1, .y = 1 },
        .{ .x = 1, .y = -1 },
        .{ .x = 1, .y = 1 },
    };

    const vertex_buffer = try vk_allocator.createBuffer(&.{
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
    }, .{ .host_writable = true });
    defer vertex_buffer.destroy();

    @memcpy(vertex_buffer.mapped(), @as([*]const u8, @ptrCast(&vertices)));

    const image_available = try vkd.createSemaphore(dev, &.{}, null);
    const render_finished = try vkd.createSemaphore(dev, &.{}, null);
    const in_flight_fence = try vkd.createFence(dev, &.{
        .flags = .{ .signaled_bit = true },
    }, null);

    surface.commit();
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    while (true) {
        t.frameStart(trace_zone_render);
        defer t.frameEnd(trace_zone_render);

        if (display.dispatchPending() != .SUCCESS) return error.DispatchFailed;

        {
            const zone = t.zone(@src()).name("waitForFences");
            defer zone.end();
            _ = try vkd.waitForFences(dev, 1, &[_]vk.Fence{in_flight_fence}, vk.TRUE, std.math.maxInt(u64));
        }
        _ = try vkd.resetFences(dev, 1, &[_]vk.Fence{in_flight_fence});

        const acquired_image = try vkd.acquireNextImageKHR(dev, swapchain, std.math.maxInt(u64), image_available, .null_handle);

        _ = try vkd.resetCommandBuffer(command_buffers[0], .{});
        _ = try vkd.beginCommandBuffer(command_buffers[0], &.{
            .flags = .{ .one_time_submit_bit = true },
        });

        _ = vkd.cmdBeginRenderPass(command_buffers[0], &.{
            .render_pass = render_pass,
            .framebuffer = framebuffers[acquired_image.image_index],
            .render_area = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = .{ .width = 1920, .height = 32 },
            },
            .clear_value_count = 1,
            .p_clear_values = &[_]vk.ClearValue{
                .{
                    .color = .{
                        .float_32 = .{ 0.0, 0.0, 0.0, 0.0 },
                    },
                },
            },
        }, .@"inline");
        _ = vkd.cmdBindPipeline(command_buffers[0], .graphics, pipelines[0]);
        _ = vkd.cmdSetViewport(command_buffers[0], 0, 1, &[_]vk.Viewport{
            .{
                .x = 0,
                .y = 0,
                .width = 1920,
                .height = 32,
                .min_depth = 0.0,
                .max_depth = 1.0,
            },
        });
        _ = vkd.cmdSetScissor(command_buffers[0], 0, 1, &[_]vk.Rect2D{
            .{
                .offset = .{
                    .x = 0,
                    .y = 0,
                },
                .extent = .{
                    .width = 1920,
                    .height = 32,
                },
            },
        });
        _ = vkd.cmdBindVertexBuffers(command_buffers[0], 0, 1, &[_]vk.Buffer{vertex_buffer.deref()}, &[_]vk.DeviceSize{0});
        _ = vkd.cmdDraw(command_buffers[0], vertices.len, 1, 0, 0);
        _ = vkd.cmdEndRenderPass(command_buffers[0]);
        _ = try vkd.endCommandBuffer(command_buffers[0]);

        _ = try vkd.queueSubmit(queue, 1, &[_]vk.SubmitInfo{
            .{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &[_]vk.Semaphore{image_available},
                .p_wait_dst_stage_mask = &[_]vk.PipelineStageFlags{
                    .{ .color_attachment_output_bit = true },
                },
                .command_buffer_count = 1,
                .p_command_buffers = &command_buffers,
                .signal_semaphore_count = 1,
                .p_signal_semaphores = &[_]vk.Semaphore{render_finished},
            },
        }, in_flight_fence);

        {
            const zone = t.zone(@src()).name("queuePresentKHR");
            defer zone.end();
            _ = try vkd.queuePresentKHR(queue, &.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = &[_]vk.Semaphore{render_finished},
                .swapchain_count = 1,
                .p_swapchains = &[_]vk.SwapchainKHR{swapchain},
                .p_image_indices = &[_]u32{acquired_image.image_index},
                .p_results = null,
            });
        }
    }
}
