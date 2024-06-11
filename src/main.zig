const std = @import("std");
const posix = std.posix;

const wayland = @import("wayland");
const wl = wayland.client.wl;
const zwlr = wayland.client.zwlr;

const c = @cImport({
    @cInclude("ft2build.h");
    @cInclude("freetype/freetype.h");
    @cInclude("freetype/ftbitmap.h");
    @cInclude("hb.h");
    @cInclude("hb-ft.h");
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

pub const Position = struct { x: i32, y: i32 };
pub const Size = struct { w: u32, h: u32 };

pub const Color = struct { r: u8, g: u8, b: u8, a: u8 };

pub const Painter = struct {
    buffer: []u8,
    width: usize,
    height: usize,

    fn draw_rect(self: *Painter, position: Position, size: Size, fill: Color) void {
        const base_x: usize = @intCast(position.x);
        const base_y: usize = @intCast(position.y);
        for (0..size.h) |y| {
            for (0..size.w) |x| {
                const pixel = self.pixel_slice(base_x + x, base_y + y);
                pixel[0] = fill.b;
                pixel[1] = fill.g;
                pixel[2] = fill.r;
                pixel[3] = fill.a;
            }
        }
    }

    inline fn i2f(i: u8) f32 {
        return @as(f32, @floatFromInt(i)) / 255.0;
    }

    inline fn f2i(f: f32) u8 {
        var v = f;
        v *= 255.0;
        //if (v > 255.0) v = 255.0;
        //if (v < 0.0) v = 0.0;

        return @intFromFloat(v);
    }

    fn blit_gray(self: *Painter, position: Position, size: Size, buffer: []u8) void {
        const base_x: usize = @intCast(position.x);
        const base_y: usize = @intCast(position.y);
        for (0..size.h) |y| {
            for (0..size.w) |x| {
                const pixel = self.pixel_slice(base_x + x, base_y + y);
                const a = i2f(buffer[y * size.w + x]);
                pixel[0] = f2i(i2f(pixel[0]) * (1 - a) + 1.0 * a);
                pixel[1] = f2i(i2f(pixel[1]) * (1 - a) + 1.0 * a);
                pixel[2] = f2i(i2f(pixel[2]) * (1 - a) + 1.0 * a);
                pixel[3] = f2i(i2f(pixel[3]) * (1 - a) + 1.0 * a);
            }
        }
    }

    fn draw_text(self: *Painter, position: Position, text: []const u8, face: FontFace) !void {
        var shaped = face.shape(text);
        defer shaped.destroy();

        std.log.debug("Measured: {}", .{try shaped.measure()});

        var x = position.x;
        var y = position.y;
        for (0..shaped.glyph_info.len) |i| {
            const glyph = try shaped.render_glyph(i);
            std.log.debug("x={}, y={}, w={}, h={}", .{ glyph.x_offset, glyph.y_offset, glyph.width, glyph.height });

            if (glyph.buffer != null) {
                self.blit_gray(
                    Position{ .x = x + glyph.x_offset, .y = y + glyph.y_offset },
                    Size{ .w = glyph.width, .h = glyph.height },
                    glyph.buffer.?,
                );
            }

            x += @divTrunc(glyph.x_advance, 64);
            y += @divTrunc(glyph.y_advance, 64);
        }
    }

    inline fn pixel_slice(self: *Painter, x: usize, y: usize) []u8 {
        const offset = (y * self.width + x) * 4;
        return self.buffer[offset .. offset + 4];
    }
};

pub const TextRenderer = struct {
    ft: c.FT_Library = null,

    fn create() !TextRenderer {
        var ft: c.FT_Library = undefined;
        var err: i32 = 0;
        err = c.FT_Init_FreeType(&ft);
        if (err != 0) {
            return error.FreeTypeError;
        }
        return TextRenderer{ .ft = ft };
    }

    fn createFace(self: *TextRenderer, file: [:0]const u8, idx: u32, size: u32) !FontFace {
        var face: c.FT_Face = undefined;
        var err: i32 = 0;
        err = c.FT_New_Face(self.ft, file, idx, &face);
        if (err != 0) {
            return error.FreeTypeError;
        }
        err = c.FT_Set_Pixel_Sizes(face, 0, size);
        if (err != 0) {
            return error.FreeTypeError;
        }
        const hb_font = c.hb_ft_font_create_referenced(face) orelse unreachable;
        return FontFace{ .hb = hb_font, .ft = face };
    }

    fn destroy(self: *TextRenderer) void {
        _ = c.FT_Done_FreeType(self.ft);
        self.ft = null;
    }
};

pub const FontFace = struct {
    hb: ?*c.hb_font_t,
    ft: c.FT_Face,

    fn shape(self: *const FontFace, text: []const u8) ShapedText {
        const buf = c.hb_buffer_create() orelse unreachable;
        c.hb_buffer_add_utf8(buf, text.ptr, @intCast(text.len), 0, @intCast(text.len));
        c.hb_buffer_guess_segment_properties(buf);
        c.hb_shape(self.hb, buf, null, 0);

        var glyph_count: u32 = undefined;
        const glyph_info = c.hb_buffer_get_glyph_infos(buf, &glyph_count);
        const glyph_pos = c.hb_buffer_get_glyph_positions(buf, &glyph_count);

        return ShapedText{
            .hb = buf,
            .ft = self.ft,
            .glyph_info = glyph_info[0..glyph_count],
            .glyph_pos = glyph_pos[0..glyph_count],
        };
    }

    fn destroy(self: *FontFace) void {
        c.hb_font_destroy(self.hb);
        self.hb = null;
    }
};

pub const ShapedText = struct {
    hb: ?*c.hb_buffer_t,
    ft: c.FT_Face,
    glyph_info: []c.hb_glyph_info_t,
    glyph_pos: []c.hb_glyph_position_t,

    fn render_glyph(self: *ShapedText, idx: usize) !RenderedGlyph {
        var err: i32 = 0;
        err = c.FT_Load_Glyph(self.ft, self.glyph_info[idx].codepoint, c.FT_LOAD_DEFAULT);
        if (err != 0) {
            return error.FreeTypeError;
        }
        err = c.FT_Render_Glyph(self.ft.*.glyph, c.FT_RENDER_MODE_NORMAL);
        if (err != 0) {
            return error.FreeTypeError;
        }

        const pos = self.glyph_pos[idx];
        const glyph = self.ft.*.glyph.*;
        const bitmap = self.ft.*.glyph.*.bitmap;
        const pitch: u32 = @intCast(bitmap.pitch);
        var buffer: ?[]u8 = null;
        if (bitmap.buffer != null) {
            buffer = bitmap.buffer[0 .. bitmap.rows * pitch];
        }
        return RenderedGlyph{
            .x_offset = pos.x_offset + glyph.bitmap_left,
            .y_offset = pos.y_offset - glyph.bitmap_top,
            .x_advance = pos.x_advance,
            .y_advance = pos.y_advance,
            .buffer = buffer,
            .width = bitmap.width,
            .height = bitmap.rows,
            .stride = pitch,
        };
    }

    fn measure(self: *ShapedText) !TextMeasurement {
        var width: i32 = 0;
        for (0..self.glyph_info.len) |idx| {
            const pos = self.glyph_pos[idx];
            width += @divTrunc(pos.x_advance, 64);
        }
        return TextMeasurement{
            .width = @intCast(width),
            .ascent = @intCast(@divTrunc(self.ft.*.ascender, 64)),
            .bounding_h = @intCast(@divTrunc(self.ft.*.ascender - self.ft.*.descender, 64)),
        };
    }

    fn destroy(self: *ShapedText) void {
        c.hb_buffer_destroy(self.hb);
        self.hb = null;
    }
};

pub const TextMeasurement = struct {
    width: u32,
    ascent: u32,
    bounding_h: u32,
};

pub const RenderedGlyph = struct {
    x_offset: i32,
    y_offset: i32,
    x_advance: i32,
    y_advance: i32,
    buffer: ?[]u8,
    width: u32,
    height: u32,
    stride: u32,
};

pub fn main() !void {
    var text = try TextRenderer.create();
    defer text.destroy();

    var face = try text.createFace("/home/yan/.local/share/fonts/Iosevka.ttc", 55, 14);
    defer face.destroy();

    const display = try wl.Display.connect(null);
    const registry = try display.getRegistry();

    var context = RegistryContext{};
    registry.setListener(*RegistryContext, RegistryContext.listener, &context);
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    const shm = context.shm orelse return error.MissingProtocol;
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

    const buffer = blk: {
        const size = 4 * 1920 * 32;
        const fd = try posix.memfd_create("wl-shm", 0);
        try posix.ftruncate(fd, size);
        const data = try posix.mmap(null, size, posix.PROT.READ | posix.PROT.WRITE, .{ .TYPE = .SHARED }, fd, 0);

        var painter = Painter{ .buffer = data, .width = 1920, .height = 40 };
        painter.draw_rect(Position{ .x = 4, .y = 4 }, Size{ .w = 250, .h = 24 }, Color{ .r = 255, .g = 0, .b = 0, .a = 127 });
        try painter.draw_text(Position{ .x = 8, .y = 4 + 3 + 15 }, "I Chose to Be This Way - Bladee", face);

        const pool = try shm.createPool(fd, size);
        defer pool.destroy();

        break :blk try pool.createBuffer(0, 1920, 32, 1920 * 4, .argb8888);
    };
    defer buffer.destroy();

    surface.attach(buffer, 0, 0);
    surface.commit();
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    while (true) {
        if (display.dispatch() != .SUCCESS) return error.DispatchFailed;
    }
}
