#include "glm_image.h"
#include "glm_types.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <array>

namespace fs = std::filesystem;

// ImageNet normalization constants (same as GLM-4V Python processor)
constexpr float IMAGE_MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float IMAGE_STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};

static std::vector<uint8_t> read_file_bytes(const std::string& path) {
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

bool decode_jpeg(tjhandle handle, const std::string& path,
                 int* out_w, int* out_h, std::vector<uint8_t>* pixels) {
    auto data = read_file_bytes(path);
    int w = 0, h = 0, subsamp = 0, cs = 0;
    if (tjDecompressHeader3(handle, data.data(), static_cast<unsigned long>(data.size()),
                            &w, &h, &subsamp, &cs) != 0) {
        return false;
    }
    *out_w = w;
    *out_h = h;
    pixels->resize(static_cast<size_t>(w) * h * 3);
    int flags = TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE;
    if (tjDecompress2(handle, data.data(), static_cast<unsigned long>(data.size()),
                      pixels->data(), w, 0, h, TJPF_RGB, flags) != 0) {
        return false;
    }
    return true;
}

// Bicubic kernel weight (Catmull-Rom: b=0, c=0.5)
static float cubic_weight(float x) {
    const float b = 0.0f, c = 0.5f;
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

static void precompute_kernel(int in_size, int out_size,
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
            float wk = cubic_weight(x - static_cast<float>(sx));
            idx[k] = clamped;
            w[k] = wk;
            sum += wk;
        }
        if (std::fabs(sum) > 1e-8f) {
            for (int k = 0; k < 4; ++k) {
                w[k] /= sum;
            }
        }
        (*indices)[i] = idx;
        (*weights)[i] = w;
    }
}

static void resize_bicubic(const uint8_t* src, int sw, int sh,
                    uint8_t* dst, int dw, int dh) {
    std::vector<std::array<int, 4>> x_idx, y_idx;
    std::vector<std::array<float, 4>> x_w, y_w;
    precompute_kernel(sw, dw, &x_idx, &x_w);
    precompute_kernel(sh, dh, &y_idx, &y_w);

    for (int y = 0; y < dh; ++y) {
        const auto& wy = y_w[y];
        const auto& iy = y_idx[y];
        for (int x = 0; x < dw; ++x) {
            const auto& wx = x_w[x];
            const auto& ix = x_idx[x];
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                for (int ky = 0; ky < 4; ++ky) {
                    const uint8_t* row = src + static_cast<size_t>(iy[ky]) * sw * 3;
                    float wyv = wy[ky];
                    for (int kx = 0; kx < 4; ++kx) {
                        const uint8_t* sp = row + static_cast<size_t>(ix[kx]) * 3;
                        sum += wyv * wx[kx] * sp[c];
                    }
                }
                int val = static_cast<int>(std::lround(sum));
                val = std::clamp(val, 0, 255);
                dst[(static_cast<size_t>(y) * dw + x) * 3 + c] = static_cast<uint8_t>(val);
            }
        }
    }
}

// Resize with aspect ratio preservation and center padding (black)
void resize_and_pad(const std::vector<uint8_t>& src, int sw, int sh,
                    std::vector<uint8_t>* dst, int target_size) {
    float scale_w = static_cast<float>(target_size) / static_cast<float>(sw);
    float scale_h = static_cast<float>(target_size) / static_cast<float>(sh);
    float scale = std::min(scale_w, scale_h);
    int rw = std::max(1, static_cast<int>(sw * scale + 0.5f));
    int rh = std::max(1, static_cast<int>(sh * scale + 0.5f));

    std::vector<uint8_t> resized(static_cast<size_t>(rw) * rh * 3);
    resize_bicubic(src.data(), sw, sh, resized.data(), rw, rh);

    dst->assign(static_cast<size_t>(target_size) * target_size * 3, 0);
    int x0 = (target_size - rw) / 2;
    int y0 = (target_size - rh) / 2;
    for (int y = 0; y < rh; ++y) {
        uint8_t* dst_row = dst->data() + static_cast<size_t>(y + y0) * target_size * 3;
        const uint8_t* src_row = resized.data() + static_cast<size_t>(y) * rw * 3;
        std::copy(src_row, src_row + static_cast<size_t>(rw) * 3,
                  dst_row + static_cast<size_t>(x0) * 3);
    }
}

// List images recursively (JPEG and PNG)
std::vector<std::string> list_images_recursive(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Convert uint8 RGB image to float16 patches for vision encoder
// Input: [H, W, 3] uint8 where H=W=target_size
// Output: mx::array [num_patches, patch_dim] float32
mx::array normalize_and_patchify(const std::vector<uint8_t>& pixels, int size, int patch_size) {
    int grid = size / patch_size;
    int num_patches = grid * grid;
    int temporal_patch = 2;
    int patch_dim = temporal_patch * patch_size * 3 * patch_size;  // 1176

    // Allocate float32 buffer (matching Python and verification test)
    std::vector<float> patches_f32(num_patches * patch_dim);

    // Patchify: Layout is [temporal, patch_h, channel, patch_w]
    for (int gh = 0; gh < grid; ++gh) {
        for (int gw = 0; gw < grid; ++gw) {
            int patch_idx = gh * grid + gw;
            int base = patch_idx * patch_dim;

            for (int t = 0; t < temporal_patch; ++t) {
                for (int ph = 0; ph < patch_size; ++ph) {
                    for (int c = 0; c < 3; ++c) {
                        for (int pw = 0; pw < patch_size; ++pw) {
                            // Source: [H, W, 3] uint8
                            int sy = gh * patch_size + ph;
                            int sx = gw * patch_size + pw;
                            size_t src_idx = (static_cast<size_t>(sy) * size + sx) * 3 + c;

                            // Dest: [temporal, patch_h, channel, patch_w]
                            size_t dst_idx = base +
                                t * (patch_size * 3 * patch_size) +
                                ph * (3 * patch_size) +
                                c * patch_size +
                                pw;

                            // Normalize: rescale to [0,1] then apply mean/std
                            float val = pixels[src_idx] / 255.0f;
                            val = (val - IMAGE_MEAN[c]) / IMAGE_STD[c];
                            patches_f32[dst_idx] = val;
                        }
                    }
                }
            }
        }
    }

    return mx::array(patches_f32.data(), {num_patches, patch_dim}, mx::float32);
}

// Batch multiple images into single patches array
// Returns [batch * num_patches, patch_dim]
mx::array batch_patches(const std::vector<mx::array>& patches_list) {
    if (patches_list.empty()) return empty_array();
    return mx::concatenate(patches_list, 0);
}
