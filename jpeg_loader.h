#pragma once

#include <cstdint>
#include <string>

enum class ResizeMode {
    Nearest,
    BicubicCatmull,
    BicubicMitchell,
};

struct JpegBenchOptions {
    std::string dir = "imgtest";
    int batch = 32;
    int iters = 50;
    bool resize = false;
    int out_w = 640;
    int out_h = 400;
    ResizeMode resize_mode = ResizeMode::Nearest;
    bool patchify = false;
    int patch_size = 16;
    int simulate_ms = 0;
};

struct JpegBenchStats {
    double total_ms = 0.0;
    double images_per_sec = 0.0;
    double loader_busy_ms = 0.0;
    double loader_wait_ms = 0.0;
    double consumer_busy_ms = 0.0;
    double consumer_wait_ms = 0.0;
    uint64_t checksum = 0;
};

const char* resize_mode_name(ResizeMode mode);
bool parse_resize_mode(const std::string& value, ResizeMode* mode);

bool run_jpeg_loader_bench(const JpegBenchOptions& opt, JpegBenchStats* stats, std::string* error);
