const c = @cImport({
    @cInclude("vk_mem_alloc.h");
});

const vk = @import("vulkan");

pub const Error = error{ApiError};

fn checkResult(result: c.VkResult) Error!void {
    if (result != c.VK_SUCCESS) {
        return Error.ApiError;
    }
}

pub const CreateOptions = struct {
    instance: vk.Instance,
    device: vk.Device,
    physical_device: vk.PhysicalDevice,
};

pub const AllocationUsage = enum {
    gpu_lazily_allocated,
    auto,
    auto_prefer_device,
    auto_prefer_host,
};

pub const AllocationOptions = struct {
    usage: AllocationUsage = .auto,
    host_writable: bool = false,
};

pub const Allocator = struct {
    obj: c.VmaAllocator,

    pub fn create(options: CreateOptions) Error!Allocator {
        var obj: c.VmaAllocator = undefined;
        const result = c.vmaCreateAllocator(&.{
            .physicalDevice = @ptrFromInt(@intFromEnum(options.physical_device)),
            .device = @ptrFromInt(@intFromEnum(options.device)),
            .instance = @ptrFromInt(@intFromEnum(options.instance)),
        }, &obj);
        try checkResult(result);
        return .{ .obj = obj };
    }

    pub fn deinit(self: @This()) void {
        c.vmaDestroyAllocator(self.obj);
    }

    pub fn createImage(self: @This(), create_info: *const vk.ImageCreateInfo, options: AllocationOptions) Error!Image {
        var image: vk.Image = undefined;
        var allocation: c.VmaAllocation = undefined;
        var result: c.VkResult = undefined;

        var flags: u32 = 0;
        if (options.host_writable) {
            flags |= c.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            flags |= c.VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        var opts = c.VmaAllocationCreateInfo{
            .usage = switch (options.usage) {
                .gpu_lazily_allocated => c.VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED,
                .auto => c.VMA_MEMORY_USAGE_AUTO,
                .auto_prefer_device => c.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .auto_prefer_host => c.VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            },
            .flags = flags,
        };

        result = c.vmaCreateImage(self.obj, @ptrCast(create_info), &opts, @ptrCast(&image), &allocation, null);
        if (result != c.VK_SUCCESS and opts.usage == c.VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED) {
            opts.usage = c.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            opts.flags |= c.VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
            result = c.vmaCreateImage(self.obj, @ptrCast(create_info), &opts, @ptrCast(&image), &allocation, null);
        }
        try checkResult(result);
        return .{
            .allocator = self.obj,
            .allocation = allocation,
            .image = image,
        };
    }

    pub fn createBuffer(self: @This(), create_info: *const vk.BufferCreateInfo, options: AllocationOptions) Error!Buffer {
        var buffer: vk.Buffer = undefined;
        var allocation: c.VmaAllocation = undefined;
        var allocation_info: c.VmaAllocationInfo = undefined;
        var result: c.VkResult = undefined;

        var flags: u32 = 0;
        if (options.host_writable) {
            flags |= c.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            flags |= c.VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        var opts = c.VmaAllocationCreateInfo{
            .usage = switch (options.usage) {
                .gpu_lazily_allocated => c.VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED,
                .auto => c.VMA_MEMORY_USAGE_AUTO,
                .auto_prefer_device => c.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .auto_prefer_host => c.VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            },
            .flags = flags,
        };

        result = c.vmaCreateBuffer(self.obj, @ptrCast(create_info), &opts, @ptrCast(&buffer), &allocation, &allocation_info);
        if (result != c.VK_SUCCESS and opts.usage == c.VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED) {
            opts.usage = c.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            opts.flags |= c.VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
            result = c.vmaCreateBuffer(self.obj, @ptrCast(create_info), &opts, @ptrCast(&buffer), &allocation, &allocation_info);
        }
        try checkResult(result);
        return .{
            .allocator = self.obj,
            .allocation = allocation,
            .allocation_info = allocation_info,
            .buffer = buffer,
        };
    }
};

pub const Image = struct {
    allocator: c.VmaAllocator,
    allocation: c.VmaAllocation,
    image: vk.Image,

    pub fn deref(self: @This()) vk.Image {
        return self.image;
    }

    pub fn destroy(self: @This()) void {
        c.vmaDestroyImage(self.allocator, @ptrFromInt(@intFromEnum(self.image)), self.allocation);
    }
};

pub const Buffer = struct {
    allocator: c.VmaAllocator,
    allocation: c.VmaAllocation,
    allocation_info: c.VmaAllocationInfo,
    buffer: vk.Buffer,

    pub fn deref(self: @This()) vk.Buffer {
        return self.buffer;
    }

    pub fn mapped(self: @This()) []u8 {
        const ptr: [*]u8 = @ptrCast(self.allocation_info.pMappedData);
        return ptr[0..self.allocation_info.size];
    }

    pub fn destroy(self: @This()) void {
        c.vmaDestroyBuffer(self.allocator, @ptrFromInt(@intFromEnum(self.buffer)), self.allocation);
    }
};
