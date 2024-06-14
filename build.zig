const std = @import("std");

const Scanner = @import("zig-wayland").Scanner;

pub fn tracy(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    const dep = b.dependency("tracy", .{});

    const lib = b.addStaticLibrary(.{
        .name = "tracy-client",
        .target = target,
        .optimize = .ReleaseFast,
    });
    lib.linkLibC();
    lib.linkLibCpp();
    lib.addIncludePath(dep.path("public"));
    lib.addCSourceFiles(.{
        .root = dep.path(""),
        .files = &[_][]const u8{
            "public/TracyClient.cpp",
        },
    });
    lib.root_module.addCMacro("TRACY_ENABLE", "1");
    lib.root_module.addCMacro("TRACY_MANUAL_LIFETIME", "1");
    lib.root_module.addCMacro("TRACY_DELAYED_INIT", "1");
    lib.root_module.addCMacro("TRACY_NO_SAMPLING", "1");
    return lib;
}

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const scanner = Scanner.create(b, .{});
    scanner.addSystemProtocol("stable/xdg-shell/xdg-shell.xml");
    scanner.addCustomProtocol("/usr/share/wlr-protocols/unstable/wlr-layer-shell-unstable-v1.xml");
    scanner.generate("wl_shm", 1);
    scanner.generate("wl_compositor", 1);
    scanner.generate("wl_output", 4);
    scanner.generate("zwlr_layer_shell_v1", 5);

    const wayland = b.createModule(.{ .root_source_file = scanner.result });

    const registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");
    const vk_gen = b.dependency("vulkan_zig", .{}).artifact("generator");
    const vk_generate_cmd = b.addRunArtifact(vk_gen);
    vk_generate_cmd.addFileArg(registry);

    const vma = b.addStaticLibrary(.{
        .name = "vma",
        .target = target,
        .optimize = optimize,
    });
    vma.linkLibC();
    vma.linkLibCpp();
    vma.addIncludePath(b.dependency("vulkan_memory_allocator", .{}).path("include"));
    vma.addCSourceFile(.{ .file = b.path("src/vma.cpp") });

    const exe = b.addExecutable(.{
        .name = "zshell",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("wayland", wayland);
    exe.root_module.addAnonymousImport("vulkan", .{
        .root_source_file = vk_generate_cmd.addOutputFileArg("vk.zig"),
    });
    exe.addIncludePath(b.dependency("vulkan_memory_allocator", .{}).path("include"));
    exe.linkLibC();
    exe.linkSystemLibrary("wayland-client");
    exe.linkSystemLibrary("freetype2");
    exe.linkSystemLibrary("harfbuzz");
    exe.linkSystemLibrary("vulkan");
    exe.linkLibrary(vma);
    exe.linkLibrary(tracy(b, target));
    exe.addIncludePath(b.dependency("tracy", .{}).path("public"));
    scanner.addCSource(exe);

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
