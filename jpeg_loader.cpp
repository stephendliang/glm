#include "jpeg_loader.h"

#include <turbojpeg.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>

namespace fs = std::filesystem;

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
    std::vector<uint8_t> patches;
};

struct BatchBuffer {
    std::vector<Image> images;
};

struct SharedState {
    std::mutex mu;
    std::condition_variable cv;
    bool ready[2] = {false, false};
    bool stop = false;
    std::string error;
    double loader_wait_ms = 0.0;
    double loader_busy_ms = 0.0;
    double consumer_wait_ms = 0.0;
    double consumer_busy_ms = 0.0;
};

const char* resize_mode_name(ResizeMode mode) {
    switch (mode) {
        case ResizeMode::Nearest:
            return "nearest";
        case ResizeMode::BicubicCatmull:
            return "catmull-rom";
        case ResizeMode::BicubicMitchell:
            return "mitchell";
    }
    return "unknown";
}

bool parse_resize_mode(const std::string& value, ResizeMode* mode) {
    if (!mode) {
        return false;
    }
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (v == "nearest") {
        *mode = ResizeMode::Nearest;
        return true;
    }
    if (v == "catmull" || v == "catmull-rom") {
        *mode = ResizeMode::BicubicCatmull;
        return true;
    }
    if (v == "mitchell") {
        *mode = ResizeMode::BicubicMitchell;
        return true;
    }
    return false;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    in.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in) {
        throw std::runtime_error("Failed to read file: " + path);
    }
    return data;
}

bool decode_jpeg(tjhandle handle, const std::string& path, Image* out) {
    auto data = read_file(path);
    int w = 0, h = 0, subsamp = 0, cs = 0;
    if (tjDecompressHeader3(handle, data.data(), static_cast<unsigned long>(data.size()),
                            &w, &h, &subsamp, &cs) != 0) {
        return false;
    }
    out->width = w;
    out->height = h;
    out->pixels.resize(static_cast<size_t>(w) * h * 3);
    int flags = TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE;
    if (tjDecompress2(handle,
                      data.data(),
                      static_cast<unsigned long>(data.size()),
                      out->pixels.data(),
                      w,
                      0,
                      h,
                      TJPF_RGB,
                      flags) != 0) {
        return false;
    }
    return true;
}

void resize_nearest(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        int sy = (y * sh) / dh;
        const uint8_t* src_row = src + static_cast<size_t>(sy) * sw * 3;
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dw * 3;
        for (int x = 0; x < dw; ++x) {
            int sx = (x * sw) / dw;
            const uint8_t* sp = src_row + static_cast<size_t>(sx) * 3;
            uint8_t* dp = dst_row + static_cast<size_t>(x) * 3;
            dp[0] = sp[0];
            dp[1] = sp[1];
            dp[2] = sp[2];
        }
    }
}

float cubic_weight(float x, float b, float c) {
    x = std::fabs(x);
    if (x < 1.0f) {
        return ((12.0f - 9.0f * b - 6.0f * c) * x * x * x
                + (-18.0f + 12.0f * b + 6.0f * c) * x * x
                + (6.0f - 2.0f * b)) / 6.0f;
    }
    if (x < 2.0f) {
        return ((-b - 6.0f * c) * x * x * x
                + (6.0f * b + 30.0f * c) * x * x
                + (-12.0f * b - 48.0f * c) * x
                + (8.0f * b + 24.0f * c)) / 6.0f;
    }
    return 0.0f;
}

void precompute_kernel(int in_size, int out_size, float b, float c,
                       std::vector<std::array<int, 4>>* indices,
                       std::vector<std::array<float, 4>>* weights) {
    indices->resize(out_size);
    weights->resize(out_size);
    float scale = static_cast<float>(in_size) / static_cast<float>(out_size);
    for (int i = 0; i < out_size; ++i) {
        float x = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
        int ix = static_cast<int>(std::floor(x));
        float sum = 0.0f;
        std::array<int, 4> idx{};
        std::array<float, 4> w{};
        for (int k = 0; k < 4; ++k) {
            int sx = ix + (k - 1);
            int clamped = std::min(std::max(sx, 0), in_size - 1);
            float wk = cubic_weight(x - static_cast<float>(sx), b, c);
            idx[k] = clamped;
            w[k] = wk;
            sum += wk;
        }
        if (sum != 0.0f) {
            for (int k = 0; k < 4; ++k) {
                w[k] /= sum;
            }
        }
        (*indices)[i] = idx;
        (*weights)[i] = w;
    }
}

void resize_bicubic(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh,
                    float b, float c) {
    std::vector<std::array<int, 4>> x_idx;
    std::vector<std::array<float, 4>> x_w;
    std::vector<std::array<int, 4>> y_idx;
    std::vector<std::array<float, 4>> y_w;
    precompute_kernel(sw, dw, b, c, &x_idx, &x_w);
    precompute_kernel(sh, dh, b, c, &y_idx, &y_w);

    for (int y = 0; y < dh; ++y) {
        const auto& wy = y_w[y];
        const auto& iy = y_idx[y];
        for (int x = 0; x < dw; ++x) {
            const auto& wx = x_w[x];
            const auto& ix = x_idx[x];
            for (int cidx = 0; cidx < 3; ++cidx) {
                float sum = 0.0f;
                for (int ky = 0; ky < 4; ++ky) {
                    const uint8_t* row = src + static_cast<size_t>(iy[ky]) * sw * 3;
                    float wyv = wy[ky];
                    for (int kx = 0; kx < 4; ++kx) {
                        const uint8_t* sp = row + static_cast<size_t>(ix[kx]) * 3;
                        sum += wyv * wx[kx] * sp[cidx];
                    }
                }
                int val = static_cast<int>(std::lround(sum));
                if (val < 0) {
                    val = 0;
                } else if (val > 255) {
                    val = 255;
                }
                dst[(static_cast<size_t>(y) * dw + x) * 3 + cidx] = static_cast<uint8_t>(val);
            }
        }
    }
}

void resize_image(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, ResizeMode mode) {
    if (mode == ResizeMode::Nearest) {
        resize_nearest(src, sw, sh, dst, dw, dh);
        return;
    }
    if (mode == ResizeMode::BicubicCatmull) {
        resize_bicubic(src, sw, sh, dst, dw, dh, 0.0f, 0.5f);
        return;
    }
    resize_bicubic(src, sw, sh, dst, dw, dh, 1.0f / 3.0f, 1.0f / 3.0f);
}

void resize_and_pad(Image* img, int out_w, int out_h, ResizeMode mode) {
    if (img->width == out_w && img->height == out_h) {
        return;
    }
    float scale_w = static_cast<float>(out_w) / static_cast<float>(img->width);
    float scale_h = static_cast<float>(out_h) / static_cast<float>(img->height);
    float scale = std::min(scale_w, scale_h);
    int rw = std::max(1, static_cast<int>(img->width * scale + 0.5f));
    int rh = std::max(1, static_cast<int>(img->height * scale + 0.5f));

    std::vector<uint8_t> resized(static_cast<size_t>(rw) * rh * 3);
    resize_image(img->pixels.data(), img->width, img->height, resized.data(), rw, rh, mode);

    std::vector<uint8_t> padded(static_cast<size_t>(out_w) * out_h * 3, 0);
    int x0 = (out_w - rw) / 2;
    int y0 = (out_h - rh) / 2;
    for (int y = 0; y < rh; ++y) {
        uint8_t* dst_row = padded.data() + static_cast<size_t>(y + y0) * out_w * 3;
        const uint8_t* src_row = resized.data() + static_cast<size_t>(y) * rw * 3;
        std::copy(src_row, src_row + static_cast<size_t>(rw) * 3,
                  dst_row + static_cast<size_t>(x0) * 3);
    }

    img->width = out_w;
    img->height = out_h;
    img->pixels.swap(padded);
}

bool patchify_rgb(const Image& img, int patch_size, std::vector<uint8_t>* out) {
    if (patch_size <= 0) {
        return false;
    }
    const int channels = 3;
    if (img.width % patch_size != 0 || img.height % patch_size != 0) {
        return false;
    }
    int grid_h = img.height / patch_size;
    int grid_w = img.width / patch_size;
    size_t patch_dim = static_cast<size_t>(patch_size) * patch_size * channels;
    size_t num_patches = static_cast<size_t>(grid_h) * grid_w;
    out->resize(num_patches * patch_dim);
    const uint8_t* src = img.pixels.data();
    for (int ph = 0; ph < grid_h; ++ph) {
        for (int pw = 0; pw < grid_w; ++pw) {
            size_t patch_idx = static_cast<size_t>(ph) * grid_w + pw;
            size_t base = patch_idx * patch_dim;
            for (int y = 0; y < patch_size; ++y) {
                int sy = ph * patch_size + y;
                const uint8_t* row = src + static_cast<size_t>(sy) * img.width * channels;
                for (int x = 0; x < patch_size; ++x) {
                    int sx = pw * patch_size + x;
                    const uint8_t* sp = row + static_cast<size_t>(sx) * channels;
                    size_t offset = base + static_cast<size_t>(y * patch_size + x) * channels;
                    (*out)[offset + 0] = sp[0];
                    (*out)[offset + 1] = sp[1];
                    (*out)[offset + 2] = sp[2];
                }
            }
        }
    }
    return true;
}

std::vector<std::string> list_jpegs(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        auto path = entry.path();
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext == ".jpg" || ext == ".jpeg") {
            files.push_back(path.string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool run_jpeg_loader_bench(const JpegBenchOptions& opt, JpegBenchStats* stats, std::string* error) {
    JpegBenchStats local_stats;
    if (opt.batch <= 0 || opt.iters <= 0) {
        if (error) {
            *error = "batch and iters must be positive.";
        }
        return false;
    }
    if (opt.patchify && opt.patch_size <= 0) {
        if (error) {
            *error = "patch_size must be positive when patchify is enabled.";
        }
        return false;
    }

    std::vector<std::string> files;
    try {
        files = list_jpegs(opt.dir);
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Failed to list directory: ") + e.what();
        }
        return false;
    }
    if (files.empty()) {
        if (error) {
            *error = "No JPEGs found in " + opt.dir;
        }
        return false;
    }

    BatchBuffer buffers[2];
    SharedState state;
    std::atomic<size_t> file_index{0};

    auto loader_thread = std::thread([&]() {
        tjhandle handle = tjInitDecompress();
        if (!handle) {
            std::lock_guard<std::mutex> lock(state.mu);
            state.error = "tjInitDecompress failed";
            state.stop = true;
            state.cv.notify_all();
            return;
        }
        for (int i = 0; i < opt.iters; ++i) {
            int buf = i % 2;
            auto wait_start = std::chrono::steady_clock::now();
            {
                std::unique_lock<std::mutex> lock(state.mu);
                state.cv.wait(lock, [&]() { return state.stop || !state.ready[buf]; });
                state.loader_wait_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wait_start).count();
                if (state.stop) {
                    break;
                }
            }

            auto work_start = std::chrono::steady_clock::now();
            auto& batch = buffers[buf];
            batch.images.clear();
            batch.images.reserve(static_cast<size_t>(opt.batch));

            for (int b = 0; b < opt.batch; ++b) {
                size_t idx = file_index.fetch_add(1) % files.size();
                Image img;
                bool ok = false;
                try {
                    ok = decode_jpeg(handle, files[idx], &img);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(state.mu);
                    state.error = e.what();
                    state.stop = true;
                    break;
                }
                if (!ok) {
                    std::lock_guard<std::mutex> lock(state.mu);
                    state.error = "JPEG decode failed: " + files[idx];
                    state.stop = true;
                    break;
                }
                if (opt.resize) {
                    resize_and_pad(&img, opt.out_w, opt.out_h, opt.resize_mode);
                }
                if (opt.patchify) {
                    if (!patchify_rgb(img, opt.patch_size, &img.patches)) {
                        std::lock_guard<std::mutex> lock(state.mu);
                        state.error = "Patchify failed (image not divisible by patch size): " + files[idx];
                        state.stop = true;
                        break;
                    }
                }
                batch.images.emplace_back(std::move(img));
                if (state.stop) {
                    break;
                }
            }

            state.loader_busy_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - work_start).count();

            {
                std::lock_guard<std::mutex> lock(state.mu);
                state.ready[buf] = true;
            }
            state.cv.notify_all();
            if (state.stop) {
                break;
            }
        }
        tjDestroy(handle);
    });

    auto start = std::chrono::steady_clock::now();
    volatile uint64_t checksum = 0;
    for (int i = 0; i < opt.iters; ++i) {
        int buf = i % 2;
        auto wait_start = std::chrono::steady_clock::now();
        {
            std::unique_lock<std::mutex> lock(state.mu);
            state.cv.wait(lock, [&]() { return state.stop || state.ready[buf]; });
            state.consumer_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wait_start).count();
            if (state.stop) {
                break;
            }
        }

        auto work_start = std::chrono::steady_clock::now();
        const auto& batch = buffers[buf];
        for (const auto& img : batch.images) {
            if (!img.patches.empty()) {
                checksum += img.patches[0];
            } else if (!img.pixels.empty()) {
                checksum += img.pixels[0];
            }
        }
        if (opt.simulate_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(opt.simulate_ms));
        }
        state.consumer_busy_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - work_start).count();

        {
            std::lock_guard<std::mutex> lock(state.mu);
            state.ready[buf] = false;
        }
        state.cv.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(state.mu);
        state.stop = true;
    }
    state.cv.notify_all();
    loader_thread.join();
    auto end = std::chrono::steady_clock::now();

    if (!state.error.empty()) {
        if (error) {
            *error = state.error;
        }
        return false;
    }

    local_stats.total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double images = static_cast<double>(opt.batch) * opt.iters;
    local_stats.images_per_sec = images / (local_stats.total_ms / 1000.0);
    local_stats.loader_busy_ms = state.loader_busy_ms;
    local_stats.loader_wait_ms = state.loader_wait_ms;
    local_stats.consumer_busy_ms = state.consumer_busy_ms;
    local_stats.consumer_wait_ms = state.consumer_wait_ms;
    local_stats.checksum = checksum;

    if (stats) {
        *stats = local_stats;
    }
    return true;
}
