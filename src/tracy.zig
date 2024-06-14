const std = @import("std");

const c = @cImport({
    @cDefine("TRACY_ENABLE", "1");
    @cDefine("TRACY_MANUAL_LIFETIME", "1");
    @cInclude("tracy/TracyC.h");
});

pub const Zone = struct {
    ctx: c.TracyCZoneCtx,

    pub fn name(self: @This(), text: []const u8) @This() {
        c.___tracy_emit_zone_name(self.ctx, text.ptr, text.len);
        return self;
    }

    pub fn end(self: @This()) void {
        c.___tracy_emit_zone_end(self.ctx);
    }
};

pub const PooledString = struct {
    ptr: [*:0]const u8,

    pub fn pool(str: [:0]const u8) @This() {
        return .{ .ptr = str.ptr };
    }
};

pub const Tracy = struct {
    pub fn init() @This() {
        c.___tracy_startup_profiler();
        return @This(){};
    }

    pub fn deinit(_: @This()) void {
        c.___tracy_shutdown_profiler();
    }

    pub fn zone(_: @This(), loc: std.builtin.SourceLocation) Zone {
        const srcloc = c.___tracy_alloc_srcloc(loc.line, loc.file.ptr, loc.file.len, loc.fn_name.ptr, loc.fn_name.len, 0);
        const ctx = c.___tracy_emit_zone_begin_alloc(srcloc, 1);
        return .{ .ctx = ctx };
    }

    pub fn frameStart(_: @This(), name: PooledString) void {
        c.___tracy_emit_frame_mark_start(name.ptr);
    }

    pub fn frameEnd(_: @This(), name: PooledString) void {
        c.___tracy_emit_frame_mark_end(name.ptr);
    }
};
