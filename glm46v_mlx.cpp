#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <optional>
#include <string>
#include <map>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <filesystem>
#include <array>
#include <functional>
#include <turbojpeg.h>
#include "glm_types.h"

namespace fs = std::filesystem;

// Global image config - initialized in main()
static ImageConfig g_img_cfg;

mx::array make_default_position_ids(int B, int T);

mx::array silu(mx::array x) {
    return mx::multiply(x, mx::sigmoid(x));
}

mx::array gelu_erf(mx::array x) {
    const float inv_sqrt2 = 0.70710678118f;
    auto erf_arg = mx::multiply(mx::array(inv_sqrt2), x);
    auto one_plus = mx::add(mx::array(1.0f), mx::erf(erf_arg));
    return mx::multiply(mx::multiply(mx::array(0.5f), x), one_plus);
}

// ============================================================================
// JPEG Loader Infrastructure (from jpeg_loader.cpp)
// ============================================================================

std::vector<uint8_t> read_file_bytes(const std::string& path) {
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
float cubic_weight(float x) {
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

void precompute_kernel(int in_size, int out_size,
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

void resize_bicubic(const uint8_t* src, int sw, int sh,
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

// ============================================================================
// GLM-4V Preprocessing: Normalize and Patchify
// ============================================================================

// Convert uint8 RGB image to float16 patches for vision encoder
// Input: [H, W, 3] uint8 where H=W=target_size
// Output: mx::array [num_patches, PATCH_INPUT_DIM] float32
// Note: PATCH_INPUT_DIM = temporal_patch * patch_h * 3 * patch_w = 2 * 14 * 3 * 14 = 1176
// Layout matches HuggingFace: after permute(0, 2, 4, 6, 3, 5, 1, 7) and view
// Final order: [temporal, patch_h, channel, patch_w]
// ImageNet normalization constants (same as GLM-4V Python processor)
constexpr float IMAGE_MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float IMAGE_STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};

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

// ==================== Weight Loading Functions ====================

// Load vision encoder weights from raw binary file (float16)
// Order must match export_weights_raw_binary() in Python exactly
bool load_vision_weights(VisionWeights* model, const std::string& path) {
    std::cout << "Loading vision weights from: " << path << std::endl;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << path << std::endl;
        return false;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read entire file into buffer
    std::vector<uint16_t> buffer(file_size / sizeof(uint16_t));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        std::cerr << "Error: Failed to read file: " << path << std::endl;
        return false;
    }

    const uint16_t* ptr = buffer.data();

    // Helper to create array and advance pointer
    // Cast to float16_t* so MLX interprets the bits as float16, not uint16
    auto read_array = [&ptr](const mx::Shape& shape) -> mx::array {
        size_t count = 1;
        for (auto d : shape) count *= d;
        mx::array arr(reinterpret_cast<const mlx::core::float16_t*>(ptr), shape, mx::float16);
        ptr += count;
        return arr;
    };

    // 1. patch_embed.weight [VISION_DIM, 2, 14, 14, 3]
    model->patch_embed_weight = read_array({VISION_DIM, 2, PATCH_SIZE, PATCH_SIZE, 3});

    // 2. patch_embed.bias [VISION_DIM]
    model->patch_embed_bias = read_array({VISION_DIM});

    // 3. pos_embed.weight [576, VISION_DIM]
    model->pos_embed_weight = read_array({NUM_PATCHES, VISION_DIM});

    // 4. post_conv_layernorm.weight [VISION_DIM]
    model->post_conv_layernorm.weight = read_array({VISION_DIM});

    // 5-148. Transformer blocks (24 blocks, 9 weights each)
    for (int i = 0; i < NUM_LAYERS; ++i) {
        auto& block = model->blocks[i];

        block.norm1.weight = read_array({VISION_DIM});
        block.norm2.weight = read_array({VISION_DIM});

        block.attn.qkv.weight = read_array({VISION_DIM, 3 * VISION_DIM});
        block.attn.qkv.bias = read_array({3 * VISION_DIM});

        block.attn.proj.weight = read_array({VISION_DIM, VISION_DIM});
        block.attn.proj.bias = read_array({VISION_DIM});

        block.mlp.gate_proj.weight = read_array({VISION_DIM, BLOCK_MLP_DIM});
        block.mlp.up_proj.weight = read_array({VISION_DIM, BLOCK_MLP_DIM});
        block.mlp.down_proj.weight = read_array({BLOCK_MLP_DIM, VISION_DIM});
    }

    // 149. post_layernorm.weight [VISION_DIM]
    model->post_layernorm.weight = read_array({VISION_DIM});

    // 150. downsample.weight [MERGE_DIM, 2, 2, VISION_DIM]
    model->downsample_weight = read_array({MERGE_DIM, SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE, VISION_DIM});

    // 151. downsample.bias [MERGE_DIM]
    model->downsample_bias = read_array({MERGE_DIM});

    // Merger weights
    model->merger.proj.weight = read_array({MERGE_DIM, MERGE_DIM});
    model->merger.proj.bias = read_array({MERGE_DIM});

    model->merger.post_ln.weight = read_array({MERGE_DIM});
    model->merger.post_ln.bias = read_array({MERGE_DIM});

    model->merger.gate_proj.weight = read_array({MERGE_DIM, MERGER_MLP_DIM});
    model->merger.up_proj.weight = read_array({MERGE_DIM, MERGER_MLP_DIM});
    model->merger.down_proj.weight = read_array({MERGER_MLP_DIM, MERGE_DIM});

    // Verify we read the expected amount
    size_t elements_read = ptr - buffer.data();
    if (elements_read != buffer.size()) {
        std::cerr << "Warning: Read " << elements_read << " elements, buffer has " << buffer.size() << std::endl;
    }

    std::cout << "Vision weights loaded successfully!" << std::endl;
    std::cout << "  File size: " << file_size / 1024 / 1024 << " MB (float16)" << std::endl;
    return true;
}

// Load text model weights from raw binary file (float16)
// Order must match export_text_weights_raw() in Python exactly
bool load_text_weights(TextModelWeights* model, const std::string& path) {
    std::cout << "Loading text weights from: " << path << std::endl;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << path << std::endl;
        return false;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint16_t> buffer(file_size / sizeof(uint16_t));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        std::cerr << "Error: Failed to read file: " << path << std::endl;
        return false;
    }

    const uint16_t* ptr = buffer.data();

    auto read_array = [&ptr](const mx::Shape& shape) -> mx::array {
        size_t count = 1;
        for (auto d : shape) count *= d;
        mx::array arr(reinterpret_cast<const mlx::core::float16_t*>(ptr), shape, mx::float16);
        ptr += count;
        return arr;
    };

    // 1. embed_tokens [vocab_size, dim]
    model->embed_tokens = read_array({TEXT_VOCAB_SIZE, TEXT_DIM});

    // 2. Transformer layers
    for (int i = 0; i < TEXT_NUM_LAYERS; ++i) {
        auto& layer = model->layers[i];

        // All 4 layer norms (in order matching Python export)
        layer.input_layernorm.weight = read_array({TEXT_DIM});
        layer.post_self_attn_layernorm.weight = read_array({TEXT_DIM});
        layer.post_attention_layernorm.weight = read_array({TEXT_DIM});
        layer.post_mlp_layernorm.weight = read_array({TEXT_DIM});

        // Self attention
        layer.self_attn.q_proj.weight = read_array({TEXT_DIM, TEXT_DIM});
        layer.self_attn.q_proj.bias = read_array({TEXT_DIM});

        layer.self_attn.k_proj.weight = read_array({TEXT_DIM, TEXT_KV_DIM});
        layer.self_attn.k_proj.bias = read_array({TEXT_KV_DIM});

        layer.self_attn.v_proj.weight = read_array({TEXT_DIM, TEXT_KV_DIM});
        layer.self_attn.v_proj.bias = read_array({TEXT_KV_DIM});

        layer.self_attn.o_proj.weight = read_array({TEXT_DIM, TEXT_DIM});

        // MLP
        layer.mlp.gate_up_proj.weight = read_array({TEXT_DIM, TEXT_GATE_UP_DIM});
        layer.mlp.down_proj.weight = read_array({TEXT_MLP_DIM, TEXT_DIM});
    }

    // 3. Final norm
    model->norm.weight = read_array({TEXT_DIM});

    // 4. LM head
    model->lm_head.weight = read_array({TEXT_DIM, TEXT_VOCAB_SIZE});

    size_t elements_read = ptr - buffer.data();
    if (elements_read != buffer.size()) {
        std::cerr << "Warning: Read " << elements_read << " elements, buffer has " << buffer.size() << std::endl;
    }

    std::cout << "Text weights loaded successfully!" << std::endl;
    std::cout << "  File size: " << file_size / 1024 / 1024 << " MB (float16)" << std::endl;
    return true;
}

mx::array fast_linear(mx::array x, const LinearWeights* w) {
    if (w->bias.size() == 0) {
        return mx::matmul(x, w->weight);
    }
    return mx::addmm(w->bias, x, w->weight);
}

mx::array rms_norm(mx::array x, const RMSNormWeights* w, float eps = 1e-5f) {
    return mx::fast::rms_norm(x, w->weight, eps);
}

mx::array layer_norm(mx::array x, const LayerNormWeights* w, float eps = 1e-5f) {
    return mx::fast::layer_norm(x, w->weight, w->bias, eps);
}

mx::array rotate_half_vision(mx::array x) {
    int dim = x.shape(-1);
    int half = dim / 2;
    auto x1 = mx::slice(x, {0, 0, 0}, {x.shape(0), x.shape(1), half});
    auto x2 = mx::slice(x, {0, 0, half}, {x.shape(0), x.shape(1), dim});
    auto neg_x2 = mx::negative(x2);
    return mx::concatenate({neg_x2, x1}, 2);
}

std::pair<mx::array, mx::array> apply_rotary_pos_emb_vision(mx::array q, mx::array k,
                                                            const mx::array& cos,
                                                            const mx::array& sin) {
    auto q_fp32 = mx::astype(q, mx::float32);
    auto k_fp32 = mx::astype(k, mx::float32);
    auto cos_exp = mx::reshape(cos, {cos.shape(0), 1, cos.shape(1)});
    auto sin_exp = mx::reshape(sin, {sin.shape(0), 1, sin.shape(1)});
    auto q_embed = mx::add(mx::multiply(q_fp32, cos_exp),
                           mx::multiply(rotate_half_vision(q_fp32), sin_exp));
    auto k_embed = mx::add(mx::multiply(k_fp32, cos_exp),
                           mx::multiply(rotate_half_vision(k_fp32), sin_exp));
    return {mx::astype(q_embed, q.dtype()), mx::astype(k_embed, k.dtype())};
}

VisionPositionData build_vision_position_data(const std::vector<GridTHW>& grid_thw) {
    VisionPositionData data;
    std::vector<int> h_coords;
    std::vector<int> w_coords;
    std::vector<int> pos_embed_indices;
    std::vector<int> image_type_h_vec;  // For position embedding interpolation
    std::vector<int> image_type_w_vec;
    int max_grid_size = 0;

    for (const auto& grid : grid_thw) {
        int t = grid.t;
        int h = grid.h;
        int w = grid.w;
        max_grid_size = std::max(max_grid_size, std::max(h, w));

        data.seqlens.push_back(t * h * w);
        int grid_h_blocks = h / SPATIAL_MERGE_SIZE;
        int grid_w_blocks = w / SPATIAL_MERGE_SIZE;
        for (int t_idx = 0; t_idx < t; ++t_idx) {
            for (int h_block = 0; h_block < grid_h_blocks; ++h_block) {
                for (int w_block = 0; w_block < grid_w_blocks; ++w_block) {
                    for (int h_inner = 0; h_inner < SPATIAL_MERGE_SIZE; ++h_inner) {
                        for (int w_inner = 0; w_inner < SPATIAL_MERGE_SIZE; ++w_inner) {
                            int h_idx = h_block * SPATIAL_MERGE_SIZE + h_inner;
                            int w_idx = w_block * SPATIAL_MERGE_SIZE + w_inner;
                            h_coords.push_back(h_idx);
                            w_coords.push_back(w_idx);

                            // Compute scaled coordinates for position embedding interpolation
                            float h_scaled = (h_idx + 0.5f) * PATCH_GRID / (float)h - 0.5f;
                            float w_scaled = (w_idx + 0.5f) * PATCH_GRID / (float)w - 0.5f;
                            int h_nn = std::max(0, std::min(PATCH_GRID - 1, (int)std::round(h_scaled)));
                            int w_nn = std::max(0, std::min(PATCH_GRID - 1, (int)std::round(w_scaled)));
                            pos_embed_indices.push_back(h_nn * PATCH_GRID + w_nn);

                            // Store the h,w coordinates for 2D position embedding lookup
                            image_type_h_vec.push_back(h_nn);
                            image_type_w_vec.push_back(w_nn);
                        }
                    }
                }
            }
        }
    }

    data.total_seq = (int)h_coords.size();

    auto h_coords_arr = mx::array(h_coords.data(), {data.total_seq}, mx::int32);
    auto w_coords_arr = mx::array(w_coords.data(), {data.total_seq}, mx::int32);
    data.pos_embed_indices = mx::array(pos_embed_indices.data(), {data.total_seq}, mx::int32);
    data.image_type_h = mx::array(image_type_h_vec.data(), {data.total_seq}, mx::int32);
    data.image_type_w = mx::array(image_type_w_vec.data(), {data.total_seq}, mx::int32);
    data.raw_h_coords = h_coords_arr;  // Store raw coordinates for interpolation
    data.raw_w_coords = w_coords_arr;

    int rotary_dim = HEAD_DIM / 2;
    auto inv_idx = mx::astype(mx::arange(0, rotary_dim, 2), mx::float32);
    auto inv_freq = mx::divide(
        mx::array(1.0f),
        mx::power(mx::array(VISION_ROPE_THETA),
                  mx::divide(inv_idx, mx::array((float)rotary_dim))));

    auto seq = mx::astype(mx::arange(max_grid_size), mx::float32);
    auto seq_col = mx::reshape(seq, {max_grid_size, 1});
    auto inv_row = mx::reshape(inv_freq, {1, inv_freq.shape(0)});
    auto freqs = mx::multiply(seq_col, inv_row);

    auto h_emb = mx::take(freqs, h_coords_arr, 0);
    auto w_emb = mx::take(freqs, w_coords_arr, 0);
    auto rotary = mx::concatenate({h_emb, w_emb}, 1);
    auto emb = mx::concatenate({rotary, rotary}, 1);
    data.cos = mx::cos(emb);
    data.sin = mx::sin(emb);
    return data;
}

// LLM RoPE rotation: interleaved pattern
// Python: x1 = x[..., 0::2], x2 = x[..., 1::2], return stack((-x2, x1), dim=-1).flatten(-2)
// Result: [a, b, c, d] -> [-b, a, -d, c]
mx::array rotate_half_llm(mx::array x) {
    int B = x.shape(0);
    int H = x.shape(1);
    int T = x.shape(2);
    int dim = x.shape(3);

    // Extract even and odd indices
    // x1 = x[..., 0::2]  (even indices)
    // x2 = x[..., 1::2]  (odd indices)
    auto x1 = mx::slice(x, {0, 0, 0, 0}, {B, H, T, dim}, {1, 1, 1, 2});  // step 2
    auto x2 = mx::slice(x, {0, 0, 0, 1}, {B, H, T, dim}, {1, 1, 1, 2});  // start at 1, step 2

    // stack((-x2, x1), dim=-1).flatten(-2)
    // Reshape to [B, H, T, half, 1] for each, stack, then flatten
    int half = dim / 2;
    auto neg_x2 = mx::negative(x2);  // [B, H, T, half]
    // x1 is also [B, H, T, half]

    // Interleave: stack along new dim, then flatten
    // Result should be [-b, a, -d, c, ...] = [-x2[0], x1[0], -x2[1], x1[1], ...]
    neg_x2 = mx::reshape(neg_x2, {B, H, T, half, 1});
    x1 = mx::reshape(x1, {B, H, T, half, 1});
    auto stacked = mx::concatenate({neg_x2, x1}, 4);  // [B, H, T, half, 2]
    return mx::reshape(stacked, {B, H, T, dim});
}

std::pair<mx::array, mx::array> text_rotary_embeddings(const mx::array& hidden_states,
                                                       const mx::array& position_ids) {
    // Use full head_dim for rotary embedding computation (partial_rotary_factor
    // only affects how much of q/k gets rotary applied, not the embedding size)
    int rotary_dim = TEXT_HEAD_DIM;
    auto inv_idx = mx::astype(mx::arange(0, rotary_dim, 2), mx::float32);
    auto inv_freq = mx::divide(
        mx::array(1.0f),
        mx::power(mx::array(TEXT_ROPE_THETA),
                  mx::divide(inv_idx, mx::array((float)rotary_dim))));

    auto inv_exp = mx::reshape(inv_freq, {1, 1, inv_freq.shape(0), 1});
    int B = position_ids.shape(1);
    inv_exp = mx::broadcast_to(inv_exp, {3, B, inv_freq.shape(0), 1});

    auto pos = mx::astype(position_ids, mx::float32);
    auto pos_exp = mx::reshape(pos, {3, B, 1, pos.shape(2)});
    auto freqs = mx::matmul(inv_exp, pos_exp);
    freqs = mx::transpose(freqs, {0, 1, 3, 2});
    auto emb = mx::concatenate({freqs, freqs}, 3);
    auto cos = mx::cos(emb);
    auto sin = mx::sin(emb);
    return {mx::astype(cos, hidden_states.dtype()), mx::astype(sin, hidden_states.dtype())};
}

std::pair<mx::array, mx::array> apply_multimodal_rotary_pos_emb(
    mx::array q, mx::array k, const mx::array& cos, const mx::array& sin) {
    // mrope_section doubled: [16, 24, 24, 16, 24, 24]
    // Split cos/sin (shape [3, B, T, 128]) into 6 chunks along last dim
    std::vector<int> chunk_sizes = {
        MROPE_SECTION_0, MROPE_SECTION_1, MROPE_SECTION_2,
        MROPE_SECTION_0, MROPE_SECTION_1, MROPE_SECTION_2
    };
    mx::Shape split_indices;
    int cumsum = 0;
    for (int i = 0; i < 5; ++i) {
        cumsum += chunk_sizes[i];
        split_indices.push_back(cumsum);
    }
    // split_indices = {16, 40, 64, 80, 104} splits into 6 chunks

    auto cos_chunks = mx::split(cos, split_indices, 3);
    auto sin_chunks = mx::split(sin, split_indices, 3);

    int B = q.shape(0);
    int T = q.shape(2);

    // For each chunk i, select section (i % 3)
    // chunk 0: select section 0 (temporal) -> m[0] of shape [1, B, T, 16]
    // chunk 1: select section 1 (height) -> m[1] of shape [1, B, T, 24]
    // chunk 2: select section 2 (width) -> m[2] of shape [1, B, T, 24]
    // chunk 3: select section 0 (temporal) -> m[0] of shape [1, B, T, 16]
    // chunk 4: select section 1 (height) -> m[1] of shape [1, B, T, 24]
    // chunk 5: select section 2 (width) -> m[2] of shape [1, B, T, 24]
    std::vector<mx::array> cos_parts, sin_parts;
    for (int i = 0; i < 6; ++i) {
        int sec = i % 3;
        int chunk_dim = chunk_sizes[i];
        // cos_chunks[i] has shape [3, B, T, chunk_dim], select section sec
        auto cos_part = mx::slice(cos_chunks[i], {sec, 0, 0, 0}, {sec + 1, B, T, chunk_dim});
        auto sin_part = mx::slice(sin_chunks[i], {sec, 0, 0, 0}, {sec + 1, B, T, chunk_dim});
        cos_parts.push_back(mx::squeeze(cos_part, std::vector<int>{0}));
        sin_parts.push_back(mx::squeeze(sin_part, std::vector<int>{0}));
    }

    // Concatenate to get [B, T, 128]
    auto cos_cat = mx::concatenate(cos_parts, 2);
    auto sin_cat = mx::concatenate(sin_parts, 2);
    // Reshape to [B, 1, T, 128]
    cos_cat = mx::reshape(cos_cat, {B, 1, T, cos_cat.shape(2)});
    sin_cat = mx::reshape(sin_cat, {B, 1, T, sin_cat.shape(2)});

    // Interleave: take first half and repeat_interleave(2)
    // cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    int half = cos_cat.shape(3) / 2;  // 64
    auto cos_half = mx::slice(cos_cat, {0, 0, 0, 0}, {B, 1, T, half});
    auto sin_half = mx::slice(sin_cat, {0, 0, 0, 0}, {B, 1, T, half});

    // repeat_interleave(2, -1): each element is repeated twice along last dim
    // [a, b, c] -> [a, a, b, b, c, c]
    // We can do this with reshape and tile
    cos_half = mx::reshape(cos_half, {B, 1, T, half, 1});
    sin_half = mx::reshape(sin_half, {B, 1, T, half, 1});
    auto cos_interleaved = mx::tile(cos_half, {1, 1, 1, 1, 2});
    auto sin_interleaved = mx::tile(sin_half, {1, 1, 1, 1, 2});
    cos_interleaved = mx::reshape(cos_interleaved, {B, 1, T, half * 2});
    sin_interleaved = mx::reshape(sin_interleaved, {B, 1, T, half * 2});

    int rotary_dim = cos_interleaved.shape(3);
    auto q_rot = mx::slice(q, {0, 0, 0, 0}, {B, q.shape(1), T, rotary_dim});
    auto q_pass = mx::slice(q, {0, 0, 0, rotary_dim}, {B, q.shape(1), T, q.shape(3)});
    auto k_rot = mx::slice(k, {0, 0, 0, 0}, {B, k.shape(1), T, rotary_dim});
    auto k_pass = mx::slice(k, {0, 0, 0, rotary_dim}, {B, k.shape(1), T, k.shape(3)});

    auto q_embed = mx::add(mx::multiply(q_rot, cos_interleaved), mx::multiply(rotate_half_llm(q_rot), sin_interleaved));
    auto k_embed = mx::add(mx::multiply(k_rot, cos_interleaved), mx::multiply(rotate_half_llm(k_rot), sin_interleaved));
    q_embed = mx::concatenate({q_embed, q_pass}, 3);
    k_embed = mx::concatenate({k_embed, k_pass}, 3);
    return {q_embed, k_embed};
}

// Cached version of attention for KV cache generation with pre-allocated cache
// Uses slice_update for O(1) cache writes instead of O(n) concatenation
// MLX SDPA supports native GQA - no need for repeat_kv expansion
// Returns: attention_output (cache is updated in-place via layer_cache reference)
mx::array text_attention_forward_cached(
    mx::array x,  // [B, T, dim] where T=prompt_len (prefill) or T=1 (decode)
    const TextAttentionWeights* w,
    const std::pair<mx::array, mx::array>& position_embeddings,
    LayerKVCache& layer_cache,  // Mutable reference - updated in place
    int cache_pos               // Write position in cache
) {
    int B = x.shape(0);
    int T = x.shape(1);

    // Project Q, K, V for current tokens
    auto q = fast_linear(x, &w->q_proj);  // [B, T, 4096]
    auto k = fast_linear(x, &w->k_proj);  // [B, T, 256]
    auto v = fast_linear(x, &w->v_proj);  // [B, T, 256]

    // Reshape to multi-head format
    auto q_rs = mx::transpose(mx::reshape(q, {B, T, TEXT_NUM_HEADS, TEXT_HEAD_DIM}), {0, 2, 1, 3});
    auto k_rs = mx::transpose(mx::reshape(k, {B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM}), {0, 2, 1, 3});
    auto v_rs = mx::transpose(mx::reshape(v, {B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM}), {0, 2, 1, 3});
    // Now: q_rs [B, 32, T, 128], k_rs/v_rs [B, 2, T, 128]

    // Apply ROPE to Q and K
    auto rotated = apply_multimodal_rotary_pos_emb(q_rs, k_rs, position_embeddings.first, position_embeddings.second);
    q_rs = rotated.first;
    auto k_rot = rotated.second;

    // Update cache using slice_update (O(T) instead of O(cache_len + T))
    layer_cache.k_cache = mx::slice_update(layer_cache.k_cache, k_rot,
        {0, 0, cache_pos, 0}, {B, TEXT_KV_HEADS, cache_pos + T, TEXT_HEAD_DIM});
    layer_cache.v_cache = mx::slice_update(layer_cache.v_cache, v_rs,
        {0, 0, cache_pos, 0}, {B, TEXT_KV_HEADS, cache_pos + T, TEXT_HEAD_DIM});
    layer_cache.current_len = cache_pos + T;

    // Slice valid portion for attention
    int total_len = layer_cache.current_len;
    auto k_valid = mx::slice(layer_cache.k_cache, {0, 0, 0, 0},
                             {B, TEXT_KV_HEADS, total_len, TEXT_HEAD_DIM});
    auto v_valid = mx::slice(layer_cache.v_cache, {0, 0, 0, 0},
                             {B, TEXT_KV_HEADS, total_len, TEXT_HEAD_DIM});

    // SDPA with native GQA support (no repeat_kv needed!)
    // q_rs: [B, 32, T, 128], k_valid/v_valid: [B, 2, total_len, 128]
    // During decode (T=1), don't use causal mask - single query attending to all keys is correct
    // During prefill (T>1), use causal mask to prevent attending to future tokens
    float scale = 1.0f / std::sqrt((float)TEXT_HEAD_DIM);
    auto out = (T == 1)
        ? mx::fast::scaled_dot_product_attention(q_rs, k_valid, v_valid, scale)
        : mx::fast::scaled_dot_product_attention(q_rs, k_valid, v_valid, scale, "causal");

    // Reshape and project output
    out = mx::transpose(out, {0, 2, 1, 3});
    out = mx::reshape(out, {B, T, TEXT_DIM});
    return fast_linear(out, &w->o_proj);
}

mx::array text_mlp_forward(mx::array x, const TextMLPWeights* w) {
    auto gate_up = fast_linear(x, &w->gate_up_proj);
    auto gate_up_split = mx::split(gate_up, 2, 2);
    auto gate = silu(gate_up_split[0]);
    auto up = gate_up_split[1];
    return fast_linear(mx::multiply(gate, up), &w->down_proj);
}

// Cached version of layer forward for KV cache generation
// Cache is updated in-place via layer_cache reference
mx::array text_layer_forward_cached(
    mx::array x,
    const TextLayerWeights* w,
    const std::pair<mx::array, mx::array>& position_embeddings,
    LayerKVCache& layer_cache,  // Mutable reference - updated in place
    int cache_pos               // Write position in cache
) {
    // Attention path
    auto residual = x;
    x = rms_norm(x, &w->input_layernorm, TEXT_RMS_EPS);
    auto attn_out = text_attention_forward_cached(x, &w->self_attn, position_embeddings, layer_cache, cache_pos);
    x = rms_norm(attn_out, &w->post_self_attn_layernorm, TEXT_RMS_EPS);
    x = mx::add(residual, x);

    // MLP path (unchanged from non-cached version)
    residual = x;
    x = rms_norm(x, &w->post_attention_layernorm, TEXT_RMS_EPS);
    x = text_mlp_forward(x, &w->mlp);
    x = rms_norm(x, &w->post_mlp_layernorm, TEXT_RMS_EPS);

    return mx::add(residual, x);
}

// Compute logits from hidden states using lm_head
mx::array compute_logits(mx::array hidden_states, const TextModelWeights* model) {
    return fast_linear(hidden_states, &model->lm_head);
}

// Sliding window size for repetition penalty (limits memory usage)
constexpr int REP_PENALTY_WINDOW = 64;

// Helper: Apply repetition penalty to logits using sliding window
// logits: [B, vocab_size], generated_tokens: per-batch token history
// Returns logits with penalty applied to recently generated tokens
// Uses a hybrid CPU/GPU approach: builds penalty mask on CPU, applies on GPU
mx::array apply_repetition_penalty_batched(
    mx::array logits,
    const std::vector<std::vector<int32_t>>& generated_tokens,
    float penalty
) {
    int B = logits.shape(0);
    int vocab_size = logits.shape(1);

    // Evaluate logits once to access data
    mx::eval(logits);

    // Copy logits to CPU for modification
    std::vector<float> logits_data(B * vocab_size);
    std::memcpy(logits_data.data(), logits.data<float>(), B * vocab_size * sizeof(float));

    // Apply penalty per batch element using sliding window
    for (int b = 0; b < B && b < static_cast<int>(generated_tokens.size()); ++b) {
        const auto& hist = generated_tokens[b];
        if (hist.empty()) continue;

        // Only consider last REP_PENALTY_WINDOW tokens
        int start = std::max(0, static_cast<int>(hist.size()) - REP_PENALTY_WINDOW);
        for (int i = start; i < static_cast<int>(hist.size()); ++i) {
            int32_t tok = hist[i];
            if (tok >= 0 && tok < vocab_size) {
                float& val = logits_data[b * vocab_size + tok];
                if (val > 0) {
                    val /= penalty;
                } else {
                    val *= penalty;
                }
            }
        }
    }

    return mx::array(logits_data.data(), {B, vocab_size}, mx::float32);
}

// Helper: Apply top-p filtering in a vectorized manner
// scores: [B, vocab_size] - temperature-scaled logits
// Returns scores with tokens beyond top_p cumulative probability masked to -inf
mx::array apply_top_p_batched(mx::array scores, float top_p) {
    int B = scores.shape(0);
    int vocab_size = scores.shape(1);

    // 1. Sort descending by negating then sorting ascending
    auto neg_scores = mx::negative(scores);
    auto sorted_indices = mx::argsort(neg_scores, -1);  // [B, vocab] indices for descending order

    // 2. Gather sorted scores using GPU-native broadcasting (no CPU allocation!)
    // batch_indices = [[0,0,0,...], [1,1,1,...], ...] via arange + broadcast
    auto batch_indices = mx::reshape(mx::arange(B, mx::int32), {B, 1});
    batch_indices = mx::broadcast_to(batch_indices, {B, vocab_size});
    auto linear_sorted = mx::add(
        mx::multiply(batch_indices, mx::array(vocab_size, mx::int32)),
        sorted_indices
    );
    linear_sorted = mx::reshape(linear_sorted, {B * vocab_size});

    auto scores_flat = mx::reshape(scores, {B * vocab_size});
    auto sorted_scores = mx::take(scores_flat, linear_sorted, 0);
    sorted_scores = mx::reshape(sorted_scores, {B, vocab_size});

    // 3. Compute cumulative probabilities
    auto sorted_probs = mx::softmax(sorted_scores, -1);
    auto cumsum = mx::cumsum(sorted_probs, -1);  // [B, vocab]

    // 4. Create mask for tokens beyond top_p (after first token exceeding threshold)
    // Shift cumsum right by 1 to keep the first token that exceeds threshold
    auto zeros_col = mx::zeros({B, 1}, cumsum.dtype());
    auto cumsum_shifted = mx::slice(cumsum, {0, 0}, {B, vocab_size - 1});
    cumsum_shifted = mx::concatenate({zeros_col, cumsum_shifted}, -1);

    auto sorted_mask = mx::greater(cumsum_shifted, mx::array(top_p));  // [B, vocab]

    // 5. Apply mask to sorted scores
    auto neg_inf = mx::array(-std::numeric_limits<float>::infinity());
    auto masked_sorted = mx::where(sorted_mask, neg_inf, sorted_scores);

    // 6. Scatter back to original order using argsort of sorted_indices
    auto unsort_indices = mx::argsort(sorted_indices, -1);  // Inverse permutation
    auto linear_unsort = mx::add(
        mx::multiply(batch_indices, mx::array(vocab_size, mx::int32)),
        unsort_indices
    );
    linear_unsort = mx::reshape(linear_unsort, {B * vocab_size});

    auto masked_flat = mx::reshape(masked_sorted, {B * vocab_size});
    auto result_flat = mx::take(masked_flat, linear_unsort, 0);

    return mx::reshape(result_flat, {B, vocab_size});
}

// Sample tokens for batched generation - returns array of shape [B]
// logits: [B, vocab_size] or [B, T, vocab_size] (last token used)
// OPTIMIZED: Uses fully vectorized MLX operations, single eval() at end
mx::array sample_tokens_batched(
    mx::array logits,
    const GenerationConfig& config,
    const std::vector<std::vector<int32_t>>& generated_tokens,  // Per-batch history
    const std::vector<bool>& finished  // Which sequences are done (optional masking)
) {
    int B = logits.shape(0);

    // 1. Handle 3D logits - take last token for each batch
    if (logits.ndim() == 3) {
        int T = logits.shape(1);
        logits = mx::slice(logits, {0, T-1, 0}, {B, T, logits.shape(2)});
        logits = mx::squeeze(logits, 1);  // [B, vocab_size]
    }

    int vocab_size = logits.shape(1);
    logits = mx::astype(logits, mx::float32);

    // 2. Apply repetition penalty first (needed for both greedy and sampling)
    if (config.repetition_penalty != 1.0f && !generated_tokens.empty()) {
        logits = apply_repetition_penalty_batched(logits, generated_tokens, config.repetition_penalty);
    }

    // 3. Fast path: Greedy decoding (single op!)
    if (!config.do_sample) {
        auto tokens = mx::argmax(logits, -1);  // [B]
        tokens = mx::astype(tokens, mx::int32);  // Ensure int32 for consistency

        // Handle finished sequences
        if (!finished.empty()) {
            std::vector<int32_t> eos_data(B, config.eos_token_ids[0]);
            auto eos_tokens = mx::array(eos_data.data(), {B}, mx::int32);
            std::vector<uint8_t> mask_data(B);
            for (int b = 0; b < B; ++b) mask_data[b] = finished[b] ? 1 : 0;
            auto mask = mx::array(mask_data.data(), {B}, mx::bool_);
            tokens = mx::where(mask, eos_tokens, tokens);
        }

        mx::eval(tokens);
        return tokens;
    }

    // 4. Temperature scaling (sampling path continues here)
    auto scores = mx::divide(logits, mx::array(config.temperature));

    // 5. Top-K filtering (vectorized)
    if (config.top_k > 0 && config.top_k < vocab_size) {
        auto topk_vals = mx::topk(scores, config.top_k, -1);  // [B, k]
        auto thresholds = mx::min(topk_vals, -1, true);       // [B, 1]
        auto mask = mx::less(scores, thresholds);
        auto neg_inf = mx::array(-std::numeric_limits<float>::infinity());
        scores = mx::where(mask, neg_inf, scores);
    }

    // 6. Top-P filtering (vectorized)
    if (config.top_p < 1.0f) {
        scores = apply_top_p_batched(scores, config.top_p);
    }

    // 7. Compute log_softmax and sample from categorical distribution
    // log_softmax(x) = x - log(sum(exp(x))) = x - max - log(sum(exp(x - max)))
    auto max_val = mx::max(scores, -1, true);  // [B, 1]
    auto shifted = mx::subtract(scores, max_val);
    auto log_sum_exp = mx::add(mx::log(mx::sum(mx::exp(shifted), -1, true)), max_val);
    auto log_probs = mx::subtract(scores, log_sum_exp);  // [B, vocab]

    // categorical(logits, num_samples) - samples from last axis, returns [B, num_samples]
    auto sampled = mx::random::categorical(log_probs, 1);  // [B, 1]
    sampled = mx::reshape(sampled, {B});  // [B]
    sampled = mx::astype(sampled, mx::int32);

    // 8. Handle finished sequences
    if (!finished.empty()) {
        std::vector<int32_t> eos_data(B, config.eos_token_ids[0]);
        auto eos_tokens = mx::array(eos_data.data(), {B}, mx::int32);
        std::vector<uint8_t> mask_data(B);
        for (int b = 0; b < B; ++b) mask_data[b] = finished[b] ? 1 : 0;
        auto mask = mx::array(mask_data.data(), {B}, mx::bool_);
        sampled = mx::where(mask, eos_tokens, sampled);
    }

    // 9. Single eval at the end (key optimization!)
    mx::eval(sampled);
    return sampled;
}

// ==================== Batched Generation Functions ====================

// Batched model forward - processes B sequences and updates batched KV cache
// Uses pre-allocated cache with slice_update for O(1) writes
mx::array text_model_forward_embeds_batched(
    mx::array inputs_embeds,         // [B, T, 4096]
    const TextModelWeights* model,
    const mx::array& position_ids,   // [3, B, T]
    BatchedTextKVCache& cache
) {
    auto pos_ids = position_ids.size() > 0 ? position_ids
                   : make_default_position_ids(inputs_embeds.shape(0), inputs_embeds.shape(1));
    auto position_embeddings = text_rotary_embeddings(inputs_embeds, pos_ids);

    int T = inputs_embeds.shape(1);
    int cache_pos = cache.current_len;  // Current write position

    auto x = inputs_embeds;
    for (int i = 0; i < TEXT_NUM_LAYERS; ++i) {
        x = text_layer_forward_cached(
            x, &model->layers[i], position_embeddings,
            cache.layers[i], cache_pos);
    }

    // Update global cache length after all layers processed
    cache.current_len = cache_pos + T;

    return rms_norm(x, &model->norm, TEXT_RMS_EPS);
}

// Check if a token is EOS
bool is_eos_token(int32_t token, const GenerationConfig& config) {
    for (int eos_id : config.eos_token_ids) {
        if (token == eos_id) return true;
    }
    return false;
}

// Batched generation function - generates B sequences in parallel
std::vector<std::vector<int32_t>> generate_batched(
    mx::array inputs_embeds,         // [B, prompt_len, 4096]
    const TextModelWeights* model,
    const mx::array& position_ids,   // [3, B, prompt_len]
    const GenerationConfig& config
) {
    int B = inputs_embeds.shape(0);
    int prompt_len = inputs_embeds.shape(1);

    std::vector<std::vector<int32_t>> generated(B);
    BatchedTextKVCache cache(B);

    // Pre-allocate KV cache for O(1) slice_update writes
    int max_seq = prompt_len + config.max_new_tokens;
    cache.preallocate(B, max_seq);

    // === PREFILL: Process all B sequences at once ===
    auto hidden = text_model_forward_embeds_batched(inputs_embeds, model, position_ids, cache);

    // Compute max position values from prefill for decode position tracking
    mx::eval(position_ids);
    // For batched case, compute max per batch element
    // position_ids shape: [3, B, T] - take max across dims 0 and 2 for each batch
    auto pos_max_t = mx::max(position_ids, {0, 2});  // [B]
    mx::eval(pos_max_t);

    // Initialize next positions for each batch element
    // Single eval, then read all values via pointer (avoids B separate evals)
    std::vector<int> next_positions(B);
    const int32_t* pos_max_ptr = pos_max_t.data<int32_t>();
    for (int b = 0; b < B; ++b) {
        next_positions[b] = pos_max_ptr[b] + 1;
        cache.cache_positions[b] = next_positions[b];
    }

    // Get logits for last token position of each batch
    // hidden: [B, prompt_len, 4096] -> extract [B, 4096] for last position
    auto last_hidden = mx::slice(hidden, {0, prompt_len - 1, 0}, {B, prompt_len, TEXT_DIM});
    // last_hidden is [B, 1, 4096], squeeze to [B, 4096] for logits
    last_hidden = mx::squeeze(last_hidden, 1);

    auto logits = compute_logits(mx::expand_dims(last_hidden, 1), model);  // [B, 1, vocab]
    logits = mx::squeeze(logits, 1);  // [B, vocab]
    mx::eval(logits);

    // Sample first tokens for all B sequences
    auto next_tokens = sample_tokens_batched(logits, config, generated, cache.finished);
    mx::eval(next_tokens);

    // Append first tokens to generated (single pointer read, no per-element eval)
    const int32_t* first_tokens_ptr = next_tokens.data<int32_t>();
    for (int b = 0; b < B; ++b) {
        int32_t tok = first_tokens_ptr[b];
        generated[b].push_back(tok);
        if (is_eos_token(tok, config)) {
            cache.finished[b] = true;
        }
    }

    // === DECODE: Generate all B sequences in parallel ===
    // Pre-allocate position buffer to avoid allocation each step
    // Layout: [t0,t1,...,tB-1, h0,h1,...,hB-1, w0,w1,...,wB-1]
    std::vector<int32_t> decode_pos_data(3 * B);

    for (int step = 0; step < config.max_new_tokens - 1; ++step) {
        // Check if all sequences are done
        if (cache.all_finished()) break;

        // Get embeddings for B tokens
        // next_tokens is [B], we need to look up embeddings
        auto token_arr = mx::reshape(next_tokens, {B, 1});  // [B, 1]
        auto new_embeds = mx::take(model->embed_tokens, token_arr, 0);  // [B, 1, 4096]

        // Update position buffer in-place (reuses pre-allocated vector)
        for (int b = 0; b < B; ++b) {
            decode_pos_data[b] = next_positions[b];          // temporal
            decode_pos_data[B + b] = next_positions[b];      // height
            decode_pos_data[2 * B + b] = next_positions[b];  // width
        }
        auto decode_pos = mx::array(decode_pos_data.data(), {3, B, 1}, mx::int32);

        // Forward all B sequences
        hidden = text_model_forward_embeds_batched(new_embeds, model, decode_pos, cache);

        // Update positions for next step
        for (int b = 0; b < B; ++b) {
            if (!cache.finished[b]) {
                next_positions[b]++;
                cache.cache_positions[b] = next_positions[b];
            }
        }

        // Get logits and sample
        logits = compute_logits(hidden, model);  // [B, 1, vocab]
        logits = mx::squeeze(logits, 1);  // [B, vocab]
        mx::eval(logits);

        next_tokens = sample_tokens_batched(logits, config, generated, cache.finished);
        mx::eval(next_tokens);

        // Append tokens to results (single pointer read, no per-element eval)
        const int32_t* tokens_ptr = next_tokens.data<int32_t>();
        for (int b = 0; b < B; ++b) {
            if (!cache.finished[b]) {
                int32_t tok = tokens_ptr[b];
                generated[b].push_back(tok);
                if (is_eos_token(tok, config)) {
                    cache.finished[b] = true;
                }
            }
        }
    }

    return generated;
}

mx::array vision_mlp_forward(mx::array x, const MLPWeights* w) {
    auto gate = silu(fast_linear(x, &w->gate_proj));
    auto up = fast_linear(x, &w->up_proj);
    return fast_linear(mx::multiply(gate, up), &w->down_proj);
}

mx::array vision_attention_forward(mx::array x, const VisionBlockWeights* w,
                                   const VisionPositionData& pos) {
    int total_seq = x.shape(0);
    auto qkv = fast_linear(x, &w->attn.qkv);
    auto qkv_rs = mx::reshape(qkv, {total_seq, 3, NUM_HEADS, HEAD_DIM});
    auto qkv_split = mx::split(qkv_rs, 3, 1);
    auto q = mx::squeeze(qkv_split[0], std::vector<int>{1});
    auto k = mx::squeeze(qkv_split[1], std::vector<int>{1});
    auto v = mx::squeeze(qkv_split[2], std::vector<int>{1});

    auto rotated = apply_rotary_pos_emb_vision(q, k, pos.cos, pos.sin);
    q = rotated.first;
    k = rotated.second;

    std::vector<mx::array> outputs;
    outputs.reserve(pos.seqlens.size());
    int offset = 0;
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    for (int len : pos.seqlens) {
        auto q_chunk = mx::slice(q, {offset, 0, 0}, {offset + len, NUM_HEADS, HEAD_DIM});
        auto k_chunk = mx::slice(k, {offset, 0, 0}, {offset + len, NUM_HEADS, HEAD_DIM});
        auto v_chunk = mx::slice(v, {offset, 0, 0}, {offset + len, NUM_HEADS, HEAD_DIM});

        q_chunk = mx::transpose(q_chunk, {1, 0, 2});
        k_chunk = mx::transpose(k_chunk, {1, 0, 2});
        v_chunk = mx::transpose(v_chunk, {1, 0, 2});

        q_chunk = mx::reshape(q_chunk, {1, NUM_HEADS, len, HEAD_DIM});
        k_chunk = mx::reshape(k_chunk, {1, NUM_HEADS, len, HEAD_DIM});
        v_chunk = mx::reshape(v_chunk, {1, NUM_HEADS, len, HEAD_DIM});

        auto attn = mx::fast::scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, scale);
        attn = mx::transpose(attn, {0, 2, 1, 3});
        attn = mx::reshape(attn, {len, VISION_DIM});
        outputs.push_back(attn);
        offset += len;
    }

    auto attn_out = mx::concatenate(outputs, 0);
    return fast_linear(attn_out, &w->attn.proj);
}

mx::array vision_block_forward(mx::array x, const VisionBlockWeights* w,
                               const VisionPositionData& pos) {
    auto residual = x;
    auto x_norm = rms_norm(x, &w->norm1, VISION_RMS_EPS);
    auto attn_out = vision_attention_forward(x_norm, w, pos);
    x = mx::add(residual, attn_out);

    residual = x;
    x_norm = rms_norm(x, &w->norm2, VISION_RMS_EPS);
    auto mlp_out = vision_mlp_forward(x_norm, &w->mlp);
    return mx::add(residual, mlp_out);
}

mx::array merger_forward(mx::array x, const MergerWeights* w) {
    x = fast_linear(x, &w->proj);
    x = layer_norm(x, &w->post_ln, VISION_RMS_EPS);
    x = gelu_erf(x);
    auto gate = silu(fast_linear(x, &w->gate_proj));
    auto up = fast_linear(x, &w->up_proj);
    return fast_linear(mx::multiply(gate, up), &w->down_proj);
}

mx::array patch_embed_forward(mx::array patches, const VisionWeights* model) {
    int num_patches = patches.shape(0);
    // Input layout from normalize_and_patchify: [T, H, C, W] = [2, 14, 3, 14]
    auto x = mx::reshape(patches, {num_patches, TEMPORAL_PATCH, PATCH_SIZE, 3, PATCH_SIZE});
    // Transpose to MLX conv3d NHWC format: [N, T, H, W, C]
    x = mx::transpose(x, {0, 1, 2, 4, 3});
    x = mx::conv3d(
        x,
        model->patch_embed_weight,
        {TEMPORAL_PATCH, PATCH_SIZE, PATCH_SIZE},
        {0, 0, 0},
        {1, 1, 1},
        1);
    x = mx::add(x, mx::reshape(model->patch_embed_bias, {1, 1, 1, 1, VISION_DIM}));
    return mx::reshape(x, {num_patches, VISION_DIM});
}

// Bilinear interpolation for position embeddings
// This matches Python's F.grid_sample behavior for non-24x24 grids
mx::array interpolate_position_embeddings(const mx::array& pos_embed_weight,
                                          const mx::array& h_coords,
                                          const mx::array& w_coords,
                                          const mx::array& target_h,
                                          const mx::array& target_w) {
    // pos_embed_weight: [576, 1536] = [24*24, hidden_size]
    // Reshape to [24, 24, hidden_size]
    int orig_size = 24;  // sqrt(576)
    int hidden_size = pos_embed_weight.shape(1);
    auto pos_2d = mx::reshape(pos_embed_weight, {orig_size, orig_size, hidden_size});

    // Match Python's grid_sample behavior:
    // Python: norm = ((coord + 0.5) / target) * 2 - 1
    // grid_sample with align_corners=False maps norm to pixel index:
    //   pixel_idx = (norm + 1) / 2 * orig_size - 0.5
    //             = ((coord + 0.5) / target) * orig_size - 0.5
    auto h_f32 = mx::astype(h_coords, mx::float32);
    auto w_f32 = mx::astype(w_coords, mx::float32);
    auto th_f32 = mx::astype(target_h, mx::float32);
    auto tw_f32 = mx::astype(target_w, mx::float32);

    // Compute pixel positions in [0, orig_size-1] range (matching grid_sample)
    auto h_pos = mx::subtract(
        mx::multiply(mx::divide(mx::add(h_f32, mx::array(0.5f)), th_f32),
                     mx::array(static_cast<float>(orig_size))),
        mx::array(0.5f));
    auto w_pos = mx::subtract(
        mx::multiply(mx::divide(mx::add(w_f32, mx::array(0.5f)), tw_f32),
                     mx::array(static_cast<float>(orig_size))),
        mx::array(0.5f));

    // Clamp to valid range
    auto h_clamped = mx::clip(h_pos, mx::array(0.0f), mx::array(static_cast<float>(orig_size - 1 - 1e-5)));
    auto w_clamped = mx::clip(w_pos, mx::array(0.0f), mx::array(static_cast<float>(orig_size - 1 - 1e-5)));

    // Get integer indices for bilinear interpolation
    auto h0 = mx::floor(h_clamped);
    auto w0 = mx::floor(w_clamped);
    auto h1 = mx::minimum(mx::add(h0, mx::array(1.0f)), mx::array(static_cast<float>(orig_size - 1)));
    auto w1 = mx::minimum(mx::add(w0, mx::array(1.0f)), mx::array(static_cast<float>(orig_size - 1)));

    // Compute interpolation weights
    auto h_frac = mx::subtract(h_clamped, h0);
    auto w_frac = mx::subtract(w_clamped, w0);

    // Convert to int32 for indexing
    auto h0_int = mx::astype(h0, mx::int32);
    auto w0_int = mx::astype(w0, mx::int32);
    auto h1_int = mx::astype(h1, mx::int32);
    auto w1_int = mx::astype(w1, mx::int32);
    mx::eval({h0_int, w0_int, h1_int, w1_int, h_frac, w_frac});

    // Get the 4 corner values for each position
    // Index into flattened pos_embed: idx = h * orig_size + w
    auto idx00 = mx::add(mx::multiply(h0_int, mx::array(orig_size)), w0_int);
    auto idx01 = mx::add(mx::multiply(h0_int, mx::array(orig_size)), w1_int);
    auto idx10 = mx::add(mx::multiply(h1_int, mx::array(orig_size)), w0_int);
    auto idx11 = mx::add(mx::multiply(h1_int, mx::array(orig_size)), w1_int);

    auto v00 = mx::take(pos_embed_weight, idx00, 0);  // [N, hidden_size]
    auto v01 = mx::take(pos_embed_weight, idx01, 0);
    auto v10 = mx::take(pos_embed_weight, idx10, 0);
    auto v11 = mx::take(pos_embed_weight, idx11, 0);

    // Expand weights for broadcasting [N, 1]
    auto h_w = mx::reshape(h_frac, {-1, 1});
    auto w_w = mx::reshape(w_frac, {-1, 1});
    auto one_minus_h = mx::subtract(mx::array(1.0f), h_w);
    auto one_minus_w = mx::subtract(mx::array(1.0f), w_w);

    // Bilinear interpolation
    auto result = mx::add(
        mx::add(
            mx::multiply(mx::multiply(one_minus_h, one_minus_w), mx::astype(v00, mx::float32)),
            mx::multiply(mx::multiply(one_minus_h, w_w), mx::astype(v01, mx::float32))
        ),
        mx::add(
            mx::multiply(mx::multiply(h_w, one_minus_w), mx::astype(v10, mx::float32)),
            mx::multiply(mx::multiply(h_w, w_w), mx::astype(v11, mx::float32))
        )
    );

    return mx::astype(result, pos_embed_weight.dtype());
}

mx::array vision_embeddings_forward(mx::array x, const VisionWeights* model,
                                    const VisionPositionData& pos) {
    // Check if we need interpolation (grid size != 24)
    // For 24x24 grid, use simple lookup; for other sizes, use bilinear interpolation
    bool needs_interpolation = false;
    for (size_t i = 0; i < pos.seqlens.size(); ++i) {
        int seq = pos.seqlens[i];
        // 24x24 grid has 576 patches
        if (seq != 576) {
            needs_interpolation = true;
            break;
        }
    }

    if (!needs_interpolation) {
        // Original fast path for 24x24 grid
        auto pos_embed = mx::take(model->pos_embed_weight, pos.pos_embed_indices, 0);
        return mx::add(x, pos_embed);
    }

    // Use bilinear interpolation for non-24x24 grids
    // Build target_h and target_w arrays from seqlens
    int total_seq = pos.image_type_h.shape(0);
    std::vector<float> target_h_vec(total_seq);
    std::vector<float> target_w_vec(total_seq);

    int offset = 0;
    for (size_t img_idx = 0; img_idx < pos.seqlens.size(); ++img_idx) {
        int seq_len = pos.seqlens[img_idx];
        // Infer grid size from seq_len (assuming t=1)
        int grid_size = static_cast<int>(std::sqrt(seq_len));
        for (int i = 0; i < seq_len; ++i) {
            target_h_vec[offset + i] = static_cast<float>(grid_size);
            target_w_vec[offset + i] = static_cast<float>(grid_size);
        }
        offset += seq_len;
    }

    auto target_h = mx::array(target_h_vec.data(), {total_seq}, mx::float32);
    auto target_w = mx::array(target_w_vec.data(), {total_seq}, mx::float32);

    auto pos_embed = interpolate_position_embeddings(
        model->pos_embed_weight,
        pos.raw_h_coords,  // Raw h coordinates (0 to grid_h-1)
        pos.raw_w_coords,  // Raw w coordinates (0 to grid_w-1)
        target_h,
        target_w
    );
    return mx::add(x, pos_embed);
}

mx::array vision_forward(mx::array patches, const VisionWeights* model,
                         const std::vector<GridTHW>& grid_thw) {
    auto pos = build_vision_position_data(grid_thw);
    auto x = patch_embed_forward(patches, model);

    x = rms_norm(x, &model->post_conv_layernorm, VISION_RMS_EPS);
    x = vision_embeddings_forward(x, model, pos);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        x = vision_block_forward(x, &model->blocks[i], pos);
    }

    x = rms_norm(x, &model->post_layernorm, VISION_RMS_EPS);

    // Reshape for spatial merge: [total_tokens, dim] -> [merge_groups, 2, 2, dim]
    // Each group of SPATIAL_MERGE_SIZE^2 tokens gets merged into one
    int merge_tokens = x.shape(0) / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE);
    x = mx::reshape(x, {merge_tokens, SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE, VISION_DIM});

    // MLX conv2d with NHWC: input [N, H, W, C], weight [O, kH, kW, I], output [N, H', W', O]
    // With kernel=stride=(2,2), this reduces 2x2 spatial to 1x1
    x = mx::conv2d(x, model->downsample_weight,
                   {SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE}, {0, 0}, {1, 1}, 1);
    // Output is [merge_tokens, 1, 1, MERGE_DIM]

    x = mx::add(x, mx::reshape(model->downsample_bias, {1, 1, 1, MERGE_DIM}));

    // Flatten spatial dims: [merge_tokens, 1, 1, MERGE_DIM] -> [merge_tokens, MERGE_DIM]
    x = mx::reshape(x, {merge_tokens, MERGE_DIM});

    return merger_forward(x, &model->merger);
}

mx::array make_default_position_ids(int B, int T) {
    auto pos = mx::astype(mx::arange(T), mx::int32);
    pos = mx::reshape(pos, {1, 1, T});
    return mx::broadcast_to(pos, {3, B, T});
}

RopeIndexResult get_rope_index(const mx::array& input_ids,
                               const std::vector<GridTHW>& image_grid_thw,
                               const std::vector<GridTHW>& video_grid_thw,
                               const mx::array& attention_mask) {
    RopeIndexResult result;
    auto ids_cpu = mx::copy(input_ids, mx::Device(mx::Device::cpu, 0));
    mx::array mask_cpu = empty_array();
    if (attention_mask.size() != 0) {
        mask_cpu = mx::copy(attention_mask, mx::Device(mx::Device::cpu, 0));
    }
    mx::eval(ids_cpu);
    if (mask_cpu.size() != 0) {
        mx::eval(mask_cpu);
    }

    int B = ids_cpu.shape(0);
    int T = ids_cpu.shape(1);
    std::vector<int32_t> position_ids(3 * B * T, 1);
    std::vector<int32_t> rope_deltas;
    rope_deltas.reserve(B);

    const int32_t* ids_ptr = ids_cpu.data<int32_t>();
    const int32_t* mask_ptr = mask_cpu.size() ? mask_cpu.data<int32_t>() : nullptr;

    int image_index = 0;
    int video_index = 0;
    int video_group_index = 0;

    for (int b = 0; b < B; ++b) {
        std::vector<int> active_positions;
        std::vector<int32_t> tokens;
        active_positions.reserve(T);
        tokens.reserve(T);

        for (int t = 0; t < T; ++t) {
            int idx = b * T + t;
            int32_t mask_val = mask_ptr ? mask_ptr[idx] : 1;
            if (mask_val == 1) {
                active_positions.push_back(t);
                tokens.push_back(ids_ptr[idx]);
            }
        }

        enum class TokenType { Image, Video, Text };
        std::vector<TokenType> token_types;
        token_types.reserve(tokens.size());
        bool video_check = false;
        for (auto tok : tokens) {
            if (tok == VIDEO_START_TOKEN_ID) {
                video_check = true;
            } else if (tok == VIDEO_END_TOKEN_ID) {
                video_check = false;
            }

            if (tok == IMAGE_TOKEN_ID && !video_check) {
                token_types.push_back(TokenType::Image);
            } else if (tok == IMAGE_TOKEN_ID && video_check) {
                token_types.push_back(TokenType::Video);
            } else {
                token_types.push_back(TokenType::Text);
            }
        }

        struct Group {
            TokenType type;
            int start;
            int end;
        };
        std::vector<Group> groups;
        if (!token_types.empty()) {
            TokenType current = token_types[0];
            int start = 0;
            for (size_t i = 1; i < token_types.size(); ++i) {
                if (token_types[i] != current) {
                    groups.push_back({current, start, (int)i});
                    current = token_types[i];
                    start = (int)i;
                }
            }
            groups.push_back({current, start, (int)token_types.size()});
        }

        std::vector<int32_t> llm_t;
        std::vector<int32_t> llm_h;
        std::vector<int32_t> llm_w;
        llm_t.reserve(tokens.size());
        llm_h.reserve(tokens.size());
        llm_w.reserve(tokens.size());

        int current_pos = 0;
        int video_frame_num = 1;
        for (const auto& group : groups) {
            if (group.type == TokenType::Image) {
                const auto& grid = image_grid_thw[image_index];
                int llm_grid_t = grid.t;
                int llm_grid_h = grid.h / SPATIAL_MERGE_SIZE;
                int llm_grid_w = grid.w / SPATIAL_MERGE_SIZE;

                for (int t_idx = 0; t_idx < llm_grid_t; ++t_idx) {
                    for (int h_idx = 0; h_idx < llm_grid_h; ++h_idx) {
                        for (int w_idx = 0; w_idx < llm_grid_w; ++w_idx) {
                            llm_t.push_back(t_idx + current_pos);
                            llm_h.push_back(h_idx + current_pos);
                            llm_w.push_back(w_idx + current_pos);
                        }
                    }
                }
                // Update current_pos to be max of generated positions + 1
                // (not total count, since positions are spatially arranged)
                int max_t = current_pos + llm_grid_t - 1;
                int max_h = current_pos + llm_grid_h - 1;
                int max_w = current_pos + llm_grid_w - 1;
                current_pos = std::max({max_t, max_h, max_w}) + 1;
                image_index += 1;
                video_frame_num = 1;
            } else if (group.type == TokenType::Video) {
                const auto& grid = video_grid_thw[video_index];
                int llm_grid_t = video_frame_num;
                int llm_grid_h = grid.h / SPATIAL_MERGE_SIZE;
                int llm_grid_w = grid.w / SPATIAL_MERGE_SIZE;

                for (int t_idx = 0; t_idx < llm_grid_t; ++t_idx) {
                    for (int h_idx = 0; h_idx < llm_grid_h; ++h_idx) {
                        for (int w_idx = 0; w_idx < llm_grid_w; ++w_idx) {
                            llm_t.push_back(t_idx + current_pos);
                            llm_h.push_back(h_idx + current_pos);
                            llm_w.push_back(w_idx + current_pos);
                        }
                    }
                }

                video_group_index += 1;
                if (video_group_index >= grid.t) {
                    video_index += 1;
                    video_group_index = 0;
                }
                // Update current_pos to be max of generated positions + 1
                int max_t = current_pos + llm_grid_t - 1;
                int max_h = current_pos + llm_grid_h - 1;
                int max_w = current_pos + llm_grid_w - 1;
                current_pos = std::max({max_t, max_h, max_w}) + 1;
                video_frame_num += 1;
            } else {
                int text_len = group.end - group.start;
                for (int i = 0; i < text_len; ++i) {
                    int pos = current_pos + i;
                    llm_t.push_back(pos);
                    llm_h.push_back(pos);
                    llm_w.push_back(pos);
                }
                current_pos += text_len;
                video_frame_num = 1;
            }
        }

        int max_pos = 0;
        for (size_t i = 0; i < llm_t.size(); ++i) {
            max_pos = std::max(max_pos, (int)llm_t[i]);
            max_pos = std::max(max_pos, (int)llm_h[i]);
            max_pos = std::max(max_pos, (int)llm_w[i]);
        }
        rope_deltas.push_back((int32_t)(max_pos + 1 - T));

        for (size_t i = 0; i < active_positions.size(); ++i) {
            int pos = active_positions[i];
            int base = b * T + pos;
            position_ids[base] = llm_t[i];
            position_ids[B * T + base] = llm_h[i];
            position_ids[2 * B * T + base] = llm_w[i];
        }
    }

    result.position_ids = mx::array(position_ids.data(), {3, B, T}, mx::int32);
    result.rope_deltas = mx::array(rope_deltas.data(), {B, 1}, mx::int32);
    return result;
}


// ============================================================================
// Shared Utilities for Generation Modes
// ============================================================================

// Parse generation config from environment variables
GenerationConfig parse_generation_config(int max_tokens) {
    GenerationConfig config;
    config.max_new_tokens = max_tokens;

    const char* do_sample_env = std::getenv("GLM_DO_SAMPLE");
    if (do_sample_env) {
        config.do_sample = std::string(do_sample_env) == "1";
    }
    const char* temp_env = std::getenv("GLM_TEMPERATURE");
    if (temp_env) {
        config.temperature = std::stof(temp_env);
    }
    const char* top_k_env = std::getenv("GLM_TOP_K");
    if (top_k_env) {
        config.top_k = std::atoi(top_k_env);
    }
    const char* top_p_env = std::getenv("GLM_TOP_P");
    if (top_p_env) {
        config.top_p = std::stof(top_p_env);
    }
    const char* rep_penalty_env = std::getenv("GLM_REP_PENALTY");
    if (rep_penalty_env) {
        config.repetition_penalty = std::stof(rep_penalty_env);
    }
    const char* seed_env = std::getenv("GLM_SEED");
    if (seed_env) {
        mx::random::seed(static_cast<uint64_t>(std::atoll(seed_env)));
        std::cout << "Random seed: " << seed_env << std::endl;
    }

    return config;
}

// Print generation config summary
void print_generation_config(const GenerationConfig& config) {
    std::cout << "\nSampling config: do_sample=" << (config.do_sample ? "true" : "false")
              << " temp=" << config.temperature
              << " top_k=" << config.top_k
              << " top_p=" << config.top_p
              << " rep_penalty=" << config.repetition_penalty << std::endl;
}

// Load filenames from text file
std::vector<std::string> load_filenames(const std::string& path) {
    std::vector<std::string> filenames;
    std::ifstream file(path);
    if (!file) {
        return filenames;  // Return empty on failure
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) filenames.push_back(line);
    }
    return filenames;
}

// Load prompt tokens from binary file
std::vector<int32_t> load_prompt_tokens(const std::string& path) {
    std::vector<int32_t> tokens;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return tokens;  // Return empty on failure
    }
    file.seekg(0, std::ios::end);
    int num_tokens = file.tellg() / sizeof(int32_t);
    file.seekg(0, std::ios::beg);
    tokens.resize(num_tokens);
    file.read(reinterpret_cast<char*>(tokens.data()), num_tokens * sizeof(int32_t));
    return tokens;
}

// Print token analysis for generated tokens
void print_token_analysis(const std::vector<int32_t>& tokens, bool include_think_tags = true) {
    for (size_t j = 0; j < tokens.size(); ++j) {
        int32_t tok = tokens[j];
        if (tok == 151329) {
            std::cout << "  [" << j << "] EOS" << std::endl;
        } else if (include_think_tags && tok == 151346) {
            std::cout << "  [" << j << "] <think>" << std::endl;
        } else if (include_think_tags && tok == 151347) {
            std::cout << "  [" << j << "] </think>" << std::endl;
        } else if (tok >= 65 && tok <= 90) {
            std::cout << "  [" << j << "] '" << static_cast<char>(tok) << "'" << std::endl;
        } else if (tok >= 97 && tok <= 122) {
            std::cout << "  [" << j << "] '" << static_cast<char>(tok) << "'" << std::endl;
        }
    }
}

// Print generation timing summary
void print_generation_timing(double ms, int total_tokens) {
    double tokens_per_sec = total_tokens / (ms / 1000.0);
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Generated " << total_tokens << " total tokens in "
              << std::fixed << std::setprecision(1) << ms << " ms"
              << " (" << std::setprecision(1) << tokens_per_sec << " tok/s)" << std::endl;
}

// ============================================================================
// Precompute Embeddings: JPEG -> Vision Encoder -> Save to Disk
// ============================================================================

int run_precompute_embeddings(const std::string& input_dir,
                               const std::string& output_dir,
                               const std::string& weights_dir,
                               int batch_size) {
    std::cout << "=== Precompute Vision Embeddings ===" << std::endl;
    std::cout << "Input:  " << input_dir << std::endl;
    std::cout << "Output: " << output_dir << std::endl;
    std::cout << "Config: " << g_img_cfg.image_size << "x" << g_img_cfg.image_size
              << " -> " << g_img_cfg.tokens_per_image << " tokens per image" << std::endl;
    std::cout << "Batch:  " << batch_size << std::endl;

    // Create output directory
    fs::create_directories(output_dir);

    // List all images
    std::vector<std::string> image_files;
    try {
        image_files = list_images_recursive(input_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error listing images: " << e.what() << std::endl;
        return 1;
    }

    if (image_files.empty()) {
        std::cerr << "No images found in " << input_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << image_files.size() << " images\n" << std::endl;

    // Load vision encoder
    std::cout << "Loading vision encoder..." << std::endl;
    VisionWeights vision;
    if (!load_vision_weights(&vision, weights_dir + "/vision_encoder.bin")) {
        std::cerr << "Failed to load vision weights" << std::endl;
        return 1;
    }

    // Create fixed grid_thw for compiled vision forward (fixed batch size for compilation)
    std::vector<GridTHW> fixed_grid_thw;
    for (int i = 0; i < batch_size; ++i) {
        fixed_grid_thw.push_back({1, g_img_cfg.grid, g_img_cfg.grid});
    }

    // Create compiled vision forward ONCE (first call will compile the graph)
    auto compiled_vision = mx::compile([&vision, &fixed_grid_thw](const std::vector<mx::array>& inputs) {
        return std::vector<mx::array>{vision_forward(inputs[0], &vision, fixed_grid_thw)};
    });

    // Double-buffer state
    PrecomputeBuffer buffers[2];
    std::mutex mu;
    std::condition_variable cv;
    bool ready[2] = {false, false};
    bool loader_done = false;
    std::string error;

    // Output files
    std::string embeddings_path = output_dir + "/embeddings.bin";
    std::string filenames_path = output_dir + "/filenames.txt";
    std::ofstream emb_out(embeddings_path, std::ios::binary);
    std::ofstream names_out(filenames_path);

    if (!emb_out || !names_out) {
        std::cerr << "Failed to open output files" << std::endl;
        return 1;
    }

    size_t total_images = image_files.size();
    std::atomic<size_t> images_processed{0};
    std::atomic<size_t> file_idx{0};

    // Loader thread: decode JPEGs and normalize/patchify
    auto loader_thread = std::thread([&]() {
        tjhandle tj = tjInitDecompress();
        if (!tj) {
            std::lock_guard<std::mutex> lock(mu);
            error = "Failed to init turbojpeg";
            loader_done = true;
            cv.notify_all();
            return;
        }

        int buf_idx = 0;
        while (file_idx.load() < total_images) {
            // Wait for buffer to be free
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&]() { return !ready[buf_idx] || !error.empty(); });
                if (!error.empty()) break;
            }

            // Fill buffer with batch
            auto& buf = buffers[buf_idx];
            buf.patches.clear();
            buf.paths.clear();

            for (int b = 0; b < batch_size; ++b) {
                size_t idx = file_idx.fetch_add(1);
                if (idx >= total_images) break;

                const auto& path = image_files[idx];
                std::vector<uint8_t> pixels;
                int w, h;

                // Decode JPEG
                if (!decode_jpeg(tj, path, &w, &h, &pixels)) {
                    std::lock_guard<std::mutex> lock(mu);
                    error = "Failed to decode: " + path;
                    break;
                }

                // Resize and pad to target size
                std::vector<uint8_t> resized;
                resize_and_pad(pixels, w, h, &resized, g_img_cfg.image_size);

                // Normalize and patchify
                auto patches = normalize_and_patchify(resized, g_img_cfg.image_size, 14);
                buf.patches.push_back(patches);
                buf.paths.push_back(path);
            }

            if (!error.empty()) break;

            // Mark buffer as ready
            {
                std::lock_guard<std::mutex> lock(mu);
                ready[buf_idx] = true;
            }
            cv.notify_all();
            buf_idx = 1 - buf_idx;
        }

        tjDestroy(tj);
        {
            std::lock_guard<std::mutex> lock(mu);
            loader_done = true;
        }
        cv.notify_all();
    });

    // Main thread: run vision encoder and save
    auto start_time = std::chrono::steady_clock::now();
    int buf_idx = 0;

    // Async pipeline state: hold previous batch for overlapped I/O
    std::optional<mx::array> prev_embeddings;
    std::vector<std::string> prev_paths;

    while (true) {
        // Wait for buffer to be ready
        {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&]() { return ready[buf_idx] || loader_done || !error.empty(); });
            if (!error.empty()) break;
            if (!ready[buf_idx] && loader_done) break;
            if (!ready[buf_idx]) continue;
        }

        auto& buf = buffers[buf_idx];
        if (buf.patches.empty()) {
            std::lock_guard<std::mutex> lock(mu);
            ready[buf_idx] = false;
            cv.notify_all();
            buf_idx = 1 - buf_idx;
            continue;
        }

        // Batch patches: [B * num_patches, patch_dim]
        auto batched_patches = batch_patches(buf.patches);
        int actual_B = static_cast<int>(buf.patches.size());

        // Pad to fixed batch_size for compiled forward if needed
        mx::array padded_patches = batched_patches;
        if (actual_B < batch_size) {
            // Pad with zeros to reach fixed batch_size
            int patches_per_image = batched_patches.shape(0) / actual_B;
            int total_patches_needed = batch_size * patches_per_image;
            int padding_patches = total_patches_needed - batched_patches.shape(0);
            auto padding = mx::zeros({padding_patches, batched_patches.shape(1)}, batched_patches.dtype());
            padded_patches = mx::concatenate({batched_patches, padding}, 0);
        }

        // Run compiled vision encoder (uses fixed batch_size grid_thw)
        auto result = compiled_vision({padded_patches});
        auto embeddings = result[0];
        // .data<>() implicitly evals - no need for explicit eval calls

        // Slice to actual batch size if we padded
        // embeddings shape: [batch_size * tokens_per_image, TEXT_DIM]
        if (actual_B < batch_size) {
            int actual_tokens = actual_B * g_img_cfg.tokens_per_image;
            embeddings = mx::slice(embeddings, {0, 0}, {actual_tokens, TEXT_DIM});
        }
        int B = actual_B;

        // embeddings shape: [B * tokens_per_image, TEXT_DIM]
        // Reshape to [B, tokens_per_image, TEXT_DIM]
        embeddings = mx::reshape(embeddings, {B, g_img_cfg.tokens_per_image, TEXT_DIM});

        // Start GPU work asynchronously
        auto emb_fp16 = mx::astype(embeddings, mx::float16);
        mx::async_eval({emb_fp16});  // NON-BLOCKING - returns immediately

        // While GPU computes current batch, write previous batch to disk
        if (prev_embeddings) {
            // .data<>() implicitly waits for prev_embeddings to be ready
            const uint16_t* data = prev_embeddings->data<uint16_t>();
            size_t bytes = static_cast<size_t>(prev_embeddings->shape(0)) * prev_embeddings->shape(1)
                         * prev_embeddings->shape(2) * sizeof(uint16_t);
            emb_out.write(reinterpret_cast<const char*>(data), bytes);

            for (const auto& path : prev_paths) {
                names_out << path << "\n";
            }
        }

        // Current becomes previous
        prev_embeddings = emb_fp16;
        prev_paths = buf.paths;

        images_processed += B;

        // Progress
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double rate = images_processed.load() / elapsed;
        std::cout << "\r  Processed " << images_processed.load() << "/" << total_images
                  << " images (" << std::fixed << std::setprecision(1) << rate << " img/s)"
                  << std::flush;

        // Release buffer
        {
            std::lock_guard<std::mutex> lock(mu);
            ready[buf_idx] = false;
        }
        cv.notify_all();
        buf_idx = 1 - buf_idx;
    }

    // Write the last batch
    if (prev_embeddings) {
        const uint16_t* data = prev_embeddings->data<uint16_t>();
        size_t bytes = static_cast<size_t>(prev_embeddings->shape(0)) * prev_embeddings->shape(1)
                     * prev_embeddings->shape(2) * sizeof(uint16_t);
        emb_out.write(reinterpret_cast<const char*>(data), bytes);
        for (const auto& path : prev_paths) {
            names_out << path << "\n";
        }
    }

    loader_thread.join();

    if (!error.empty()) {
        std::cerr << "\nError: " << error << std::endl;
        return 1;
    }

    emb_out.close();
    names_out.close();

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n\n=== Complete ===" << std::endl;
    std::cout << "Processed: " << images_processed.load() << " images" << std::endl;
    std::cout << "Time:      " << std::fixed << std::setprecision(2) << total_time << " seconds" << std::endl;
    std::cout << "Rate:      " << std::setprecision(1) << (images_processed.load() / total_time) << " images/sec" << std::endl;
    std::cout << "Output:" << std::endl;
    std::cout << "  " << embeddings_path << " (" << (fs::file_size(embeddings_path) / 1024 / 1024) << " MB)" << std::endl;
    std::cout << "  " << filenames_path << std::endl;

    return 0;
}

// ============================================================================
// Generate Mode: Dual-Image Comparison
// ============================================================================

int run_generate_mode(const std::string& weights_dir) {
    const char* embeds_dir_env = std::getenv("GLM_EMBEDS_DIR");
    const char* pairs_file_env = std::getenv("GLM_PAIRS_FILE");
    const char* idx_a_env = std::getenv("GLM_IDX_A");
    const char* idx_b_env = std::getenv("GLM_IDX_B");
    const char* batch_size_env = std::getenv("GLM_BATCH_SIZE");
    const char* max_tokens_env = std::getenv("GLM_MAX_TOKENS");
    const char* chunk_env = std::getenv("GLM_CHUNK_SIZE");

    std::string embeds_dir = embeds_dir_env ? embeds_dir_env : "embeddings_out";
    int max_tokens = max_tokens_env ? std::atoi(max_tokens_env) : 100;
    int batch_size = batch_size_env ? std::atoi(batch_size_env) : 4;

    // Chunk size controls actual inference batch size (default: process all at once)
    int chunk_size = batch_size;
    if (chunk_env) {
        chunk_size = std::atoi(chunk_env);
        if (chunk_size <= 0) chunk_size = batch_size;
    }

    std::cout << "=== Dual-Image Comparison Generation (448x448) ===" << std::endl;
    if (chunk_size < batch_size) {
        std::cout << "Chunk size: " << chunk_size << " (processing in chunks)" << std::endl;
    }

    // 1. Load filenames
    std::string filenames_path = embeds_dir + "/filenames.txt";
    auto filenames = load_filenames(filenames_path);
    if (filenames.empty()) {
        std::cerr << "Failed to open " << filenames_path << std::endl;
        return 1;
    }
    int num_images = static_cast<int>(filenames.size());
    std::cout << "Loaded " << num_images << " image filenames" << std::endl;

    if (num_images < 2) {
        std::cerr << "Need at least 2 images for comparison" << std::endl;
        return 1;
    }

    // 2. Determine pairs to compare (unified logic)
    std::vector<std::pair<int, int>> pairs;

    if (pairs_file_env) {
        // Load pairs from file
        int max_pairs = batch_size_env ? std::atoi(batch_size_env) : 64;
        std::ifstream pairs_file(pairs_file_env);
        if (!pairs_file) {
            std::cerr << "Failed to open pairs file: " << pairs_file_env << std::endl;
            return 1;
        }
        std::string pair_line;
        while (std::getline(pairs_file, pair_line) && static_cast<int>(pairs.size()) < max_pairs) {
            size_t comma = pair_line.find(',');
            if (comma != std::string::npos) {
                int idx_a = std::atoi(pair_line.substr(0, comma).c_str());
                int idx_b = std::atoi(pair_line.substr(comma + 1).c_str());
                if (idx_a >= 0 && idx_a < num_images && idx_b >= 0 && idx_b < num_images) {
                    pairs.push_back({idx_a, idx_b});
                }
            }
        }
        pairs_file.close();
        std::cout << "Loaded " << pairs.size() << " pairs from " << pairs_file_env << std::endl;
    } else if (idx_a_env && idx_b_env) {
        // Single explicit pair (both must be set)
        int idx_a = std::atoi(idx_a_env);
        int idx_b = std::atoi(idx_b_env);
        if (idx_a >= 0 && idx_a < num_images && idx_b >= 0 && idx_b < num_images) {
            pairs.push_back({idx_a, idx_b});
            std::cout << "Using explicit pair: A=" << idx_a << ", B=" << idx_b << std::endl;
        } else {
            std::cerr << "Invalid indices: idx_a=" << idx_a << ", idx_b=" << idx_b
                      << " (num_images=" << num_images << ")" << std::endl;
            return 1;
        }
    } else {
        // Generate random pairs (default: 4)
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < batch_size; ++i) {
            int idx_a = std::rand() % num_images;
            int idx_b;
            do { idx_b = std::rand() % num_images; } while (idx_b == idx_a);
            pairs.push_back({idx_a, idx_b});
        }
        std::cout << "Generated " << pairs.size() << " random pairs" << std::endl;
    }

    int B = static_cast<int>(pairs.size());
    if (B == 0) {
        std::cerr << "No valid pairs to process" << std::endl;
        return 1;
    }

    std::cout << "\nComparing " << B << " pair" << (B > 1 ? "s" : "") << ":" << std::endl;
    for (int i = 0; i < B; ++i) {
        std::cout << "  [" << i << "] A=" << pairs[i].first << " (" << filenames[pairs[i].first] << ")"
                  << " vs B=" << pairs[i].second << " (" << filenames[pairs[i].second] << ")" << std::endl;
    }

    // 3. Load embeddings for all pairs
    std::string embeds_path = embeds_dir + "/embeddings.bin";
    std::ifstream embeds_file(embeds_path, std::ios::binary);
    if (!embeds_file) {
        std::cerr << "Failed to open " << embeds_path << std::endl;
        return 1;
    }

    const int tokens_per_image = g_img_cfg.tokens_per_image;
    const size_t bytes_per_image = tokens_per_image * TEXT_DIM * sizeof(uint16_t);

    std::vector<std::vector<uint16_t>> embeds_a(B, std::vector<uint16_t>(tokens_per_image * TEXT_DIM));
    std::vector<std::vector<uint16_t>> embeds_b(B, std::vector<uint16_t>(tokens_per_image * TEXT_DIM));

    for (int i = 0; i < B; ++i) {
        embeds_file.seekg(pairs[i].first * bytes_per_image);
        embeds_file.read(reinterpret_cast<char*>(embeds_a[i].data()), bytes_per_image);
        embeds_file.seekg(pairs[i].second * bytes_per_image);
        embeds_file.read(reinterpret_cast<char*>(embeds_b[i].data()), bytes_per_image);
    }
    embeds_file.close();

    // 4. Load prompt tokens
    std::string tokens_path = weights_dir + "/dueling_prompt_tokens_448.bin";
    auto prompt_tokens = load_prompt_tokens(tokens_path);
    if (prompt_tokens.empty()) {
        std::cerr << "Failed to open " << tokens_path << std::endl;
        return 1;
    }
    int num_tokens = static_cast<int>(prompt_tokens.size());
    std::cout << "Loaded " << num_tokens << " prompt tokens" << std::endl;

    // 5. Load text model
    std::cout << "Loading text model..." << std::endl;
    TextModelWeights text;
    if (!load_text_weights(&text, weights_dir + "/text_model.bin")) {
        std::cerr << "Failed to load text weights" << std::endl;
        return 1;
    }

    // 6. Prepare base embeddings (shared across all pairs)
    auto input_ids = mx::array(prompt_tokens.data(), {1, num_tokens}, mx::int32);
    auto base_embeds = mx::take(text.embed_tokens, input_ids, 0);  // [1, seq_len, dim]

    // Pre-extract prefix, mid, suffix (shared across batch)
    auto prefix = mx::slice(base_embeds, {0, 0, 0}, {1, g_img_cfg.image_a_start, TEXT_DIM});
    auto mid = mx::slice(base_embeds, {0, g_img_cfg.image_a_end, 0}, {1, g_img_cfg.image_b_start, TEXT_DIM});
    auto suffix = mx::slice(base_embeds, {0, g_img_cfg.image_b_end, 0}, {1, num_tokens, TEXT_DIM});

    // 7. Configure generation (before chunking loop)
    auto config = parse_generation_config(max_tokens);
    print_generation_config(config);

    // 8. Chunked batched generation
    int num_chunks = (B + chunk_size - 1) / chunk_size;
    std::cout << "\nGenerating up to " << max_tokens << " tokens for " << B << " pair" << (B > 1 ? "s" : "");
    if (num_chunks > 1) {
        std::cout << " in " << num_chunks << " chunks of " << chunk_size;
    }
    std::cout << "..." << std::endl;

    std::vector<std::vector<int32_t>> all_generated;
    all_generated.reserve(B);

    auto start = std::chrono::high_resolution_clock::now();

    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int chunk_start = chunk_idx * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, B);
        int chunk_B = chunk_end - chunk_start;

        if (num_chunks > 1) {
            std::cout << "\n--- Chunk " << (chunk_idx + 1) << "/" << num_chunks
                      << " (pairs " << chunk_start << "-" << (chunk_end - 1) << ") ---" << std::endl;
        }

        // Build batched sequences for this chunk - single allocation approach
        // 1. Batch all image embeddings into contiguous tensors
        std::vector<uint16_t> img_a_data(chunk_B * tokens_per_image * TEXT_DIM);
        std::vector<uint16_t> img_b_data(chunk_B * tokens_per_image * TEXT_DIM);
        const size_t img_bytes = tokens_per_image * TEXT_DIM * sizeof(uint16_t);
        for (int i = 0; i < chunk_B; ++i) {
            std::memcpy(img_a_data.data() + i * tokens_per_image * TEXT_DIM,
                        embeds_a[chunk_start + i].data(), img_bytes);
            std::memcpy(img_b_data.data() + i * tokens_per_image * TEXT_DIM,
                        embeds_b[chunk_start + i].data(), img_bytes);
        }
        auto batched_img_a = mx::array(reinterpret_cast<const mlx::core::float16_t*>(img_a_data.data()),
                                       {chunk_B, tokens_per_image, TEXT_DIM}, mx::float16);
        auto batched_img_b = mx::array(reinterpret_cast<const mlx::core::float16_t*>(img_b_data.data()),
                                       {chunk_B, tokens_per_image, TEXT_DIM}, mx::float16);

        // 2. Broadcast prefix/mid/suffix to batch dimension [chunk_B, tokens, dim]
        auto prefix_batched = mx::broadcast_to(prefix, {chunk_B, prefix.shape(1), TEXT_DIM});
        auto mid_batched = mx::broadcast_to(mid, {chunk_B, mid.shape(1), TEXT_DIM});
        auto suffix_batched = mx::broadcast_to(suffix, {chunk_B, suffix.shape(1), TEXT_DIM});

        // 3. Single concatenate along sequence axis
        auto inputs_embeds = mx::concatenate({prefix_batched, batched_img_a, mid_batched, batched_img_b, suffix_batched}, 1);

        // Create batched position IDs for this chunk
        // CRITICAL: MLX broadcast_to/tile create views that don't materialize correctly
        // when copied to CPU and read via pointer. Must build from raw data.
        std::vector<int32_t> batched_ids(chunk_B * num_tokens);
        mx::eval(input_ids);
        const int32_t* src_ids = input_ids.data<int32_t>();
        for (int b = 0; b < chunk_B; ++b) {
            std::memcpy(batched_ids.data() + b * num_tokens, src_ids, num_tokens * sizeof(int32_t));
        }
        auto chunk_input_ids = mx::array(batched_ids.data(), {chunk_B, num_tokens}, mx::int32);

        std::vector<GridTHW> grids;
        for (int i = 0; i < chunk_B; ++i) {
            grids.push_back({1, g_img_cfg.grid, g_img_cfg.grid});  // Image A
            grids.push_back({1, g_img_cfg.grid, g_img_cfg.grid});  // Image B
        }
        auto rope = get_rope_index(chunk_input_ids, grids, {}, empty_array());
        auto position_ids = rope.position_ids;

        // Generate for this chunk
        auto chunk_generated = generate_batched(inputs_embeds, &text, position_ids, config);

        // Accumulate results
        for (auto& gen : chunk_generated) {
            all_generated.push_back(std::move(gen));
        }

        if (num_chunks > 1) {
            int chunk_tokens = 0;
            for (int i = chunk_start; i < chunk_end; ++i) {
                chunk_tokens += all_generated[i].size();
            }
            std::cout << "Chunk generated " << chunk_tokens << " tokens" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    int total_tokens = 0;
    for (const auto& gen : all_generated) {
        total_tokens += gen.size();
    }

    // 9. Output results
    print_generation_timing(ms, total_tokens);

    for (int i = 0; i < B; ++i) {
        std::cout << "\n--- Pair " << i << " (A=" << pairs[i].first << " vs B=" << pairs[i].second << ") ---" << std::endl;
        std::cout << "Generated " << all_generated[i].size() << " tokens" << std::endl;
        std::cout << "Token IDs: ";
        for (size_t j = 0; j < all_generated[i].size(); ++j) {
            std::cout << all_generated[i][j];
            if (j < all_generated[i].size() - 1) std::cout << " ";
        }
        std::cout << std::endl;

        print_token_analysis(all_generated[i], true);
    }

    return 0;
}

// ============================================================================
// Describe Mode: Single-Image Description
// ============================================================================

int run_describe_mode(const std::string& weights_dir) {
    const char* embeds_dir_env = std::getenv("GLM_EMBEDS_DIR");
    const char* batch_size_env = std::getenv("GLM_BATCH_SIZE");
    const char* indices_file_env = std::getenv("GLM_INDICES_FILE");
    const char* idx_env = std::getenv("GLM_IDX");
    const char* max_tokens_env = std::getenv("GLM_MAX_TOKENS");
    const char* random_indices_env = std::getenv("GLM_RANDOM_INDICES");
    const char* chunk_env = std::getenv("GLM_CHUNK_SIZE");

    std::string embeds_dir = embeds_dir_env ? embeds_dir_env : "embeddings_out";
    int batch_size = batch_size_env ? std::atoi(batch_size_env) : 64;
    int max_tokens = max_tokens_env ? std::atoi(max_tokens_env) : 100;
    bool random_indices = random_indices_env && std::string(random_indices_env) == "1";

    // Chunk size controls actual inference batch size (default: process all at once)
    int chunk_size = batch_size;
    if (chunk_env) {
        chunk_size = std::atoi(chunk_env);
        if (chunk_size <= 0) chunk_size = batch_size;
    }

    std::cout << "=== Image Description Mode (448x448) ===" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    if (chunk_size < batch_size) {
        std::cout << "Chunk size: " << chunk_size << " (processing in chunks)" << std::endl;
    }

    // 1. Load filenames
    std::string filenames_path = embeds_dir + "/filenames.txt";
    auto filenames = load_filenames(filenames_path);
    if (filenames.empty()) {
        std::cerr << "Failed to open " << filenames_path << std::endl;
        return 1;
    }
    int num_images = static_cast<int>(filenames.size());
    std::cout << "Loaded " << num_images << " image filenames" << std::endl;

    if (num_images == 0) {
        std::cerr << "No images found in embeddings directory" << std::endl;
        return 1;
    }

    // 2. Determine indices to process
    // Priority: GLM_INDICES_FILE > GLM_IDX > GLM_RANDOM_INDICES > first N images
    std::vector<int> indices;
    if (indices_file_env) {
        // Load indices from file
        std::ifstream indices_file(indices_file_env);
        if (!indices_file) {
            std::cerr << "Failed to open indices file: " << indices_file_env << std::endl;
            return 1;
        }
        std::string idx_line;
        while (std::getline(indices_file, idx_line) && static_cast<int>(indices.size()) < batch_size) {
            if (!idx_line.empty()) {
                int idx = std::atoi(idx_line.c_str());
                if (idx >= 0 && idx < num_images) {
                    indices.push_back(idx);
                }
            }
        }
        indices_file.close();
        std::cout << "Loaded " << indices.size() << " indices from " << indices_file_env << std::endl;
    } else if (idx_env) {
        // Single explicit index
        int idx = std::atoi(idx_env);
        if (idx >= 0 && idx < num_images) {
            indices.push_back(idx);
            std::cout << "Using single image index: " << idx << std::endl;
        } else {
            std::cerr << "GLM_IDX=" << idx << " is out of range [0, " << num_images - 1 << "]" << std::endl;
            return 1;
        }
    } else if (random_indices) {
        // Generate random indices
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < batch_size && i < num_images; ++i) {
            indices.push_back(std::rand() % num_images);
        }
        std::cout << "Generated " << indices.size() << " random indices" << std::endl;
    } else {
        // Use first B images
        for (int i = 0; i < batch_size && i < num_images; ++i) {
            indices.push_back(i);
        }
        std::cout << "Using first " << indices.size() << " images" << std::endl;
    }

    int B = static_cast<int>(indices.size());
    if (B == 0) {
        std::cerr << "No valid indices to process" << std::endl;
        return 1;
    }
    std::cout << "\nProcessing " << B << " images:" << std::endl;
    for (int i = 0; i < std::min(B, 10); ++i) {
        std::cout << "  [" << i << "] idx=" << indices[i] << " (" << filenames[indices[i]] << ")" << std::endl;
    }
    if (B > 10) {
        std::cout << "  ... and " << (B - 10) << " more" << std::endl;
    }

    // 3. Load embeddings for all images
    std::string embeds_path = embeds_dir + "/embeddings.bin";
    std::ifstream embeds_file(embeds_path, std::ios::binary);
    if (!embeds_file) {
        std::cerr << "Failed to open " << embeds_path << std::endl;
        return 1;
    }

    const int tokens_per_image = g_img_cfg.tokens_per_image;
    const size_t bytes_per_image = tokens_per_image * TEXT_DIM * sizeof(uint16_t);

    std::vector<std::vector<uint16_t>> embeds(B, std::vector<uint16_t>(tokens_per_image * TEXT_DIM));

    for (int i = 0; i < B; ++i) {
        embeds_file.seekg(indices[i] * bytes_per_image);
        embeds_file.read(reinterpret_cast<char*>(embeds[i].data()), bytes_per_image);
    }
    embeds_file.close();
    std::cout << "Loaded embeddings for " << B << " images" << std::endl;

    // 4. Load single-image prompt tokens
    std::string tokens_path = weights_dir + "/single_image_prompt_448.bin";
    auto prompt_tokens = load_prompt_tokens(tokens_path);
    if (prompt_tokens.empty()) {
        std::cerr << "Failed to open " << tokens_path << std::endl;
        return 1;
    }
    int num_tokens = static_cast<int>(prompt_tokens.size());
    std::cout << "Loaded " << num_tokens << " prompt tokens (single-image)" << std::endl;

    // 5. Load text model
    std::cout << "Loading text model..." << std::endl;
    TextModelWeights text;
    if (!load_text_weights(&text, weights_dir + "/text_model.bin")) {
        std::cerr << "Failed to load text weights" << std::endl;
        return 1;
    }

    // 6. Build batched input embeddings
    // Single image layout: [prefix:5][image:tokens][suffix:~9]
    const int image_start = g_img_cfg.image_start;
    const int image_end = g_img_cfg.image_end;

    auto input_ids = mx::array(prompt_tokens.data(), {1, num_tokens}, mx::int32);
    auto base_embeds = mx::take(text.embed_tokens, input_ids, 0);  // [1, seq_len, dim]

    // Pre-extract prefix and suffix (shared across batch)
    auto prefix = mx::slice(base_embeds, {0, 0, 0}, {1, image_start, TEXT_DIM});
    auto suffix = mx::slice(base_embeds, {0, image_end, 0}, {1, num_tokens, TEXT_DIM});

    // 7. Configure generation (before chunking loop)
    auto config = parse_generation_config(max_tokens);
    print_generation_config(config);

    // 8. Chunked batched generation
    int num_chunks = (B + chunk_size - 1) / chunk_size;
    std::cout << "\nGenerating up to " << max_tokens << " tokens for " << B << " images";
    if (num_chunks > 1) {
        std::cout << " in " << num_chunks << " chunks of " << chunk_size;
    }
    std::cout << "..." << std::endl;

    std::vector<std::vector<int32_t>> all_generated;
    all_generated.reserve(B);

    auto start = std::chrono::high_resolution_clock::now();

    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int chunk_start = chunk_idx * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, B);
        int chunk_B = chunk_end - chunk_start;

        if (num_chunks > 1) {
            std::cout << "\n--- Chunk " << (chunk_idx + 1) << "/" << num_chunks
                      << " (images " << chunk_start << "-" << (chunk_end - 1) << ") ---" << std::endl;
        }

        // Build batched sequences for this chunk - single allocation approach
        // 1. Batch all image embeddings into contiguous tensor
        std::vector<uint16_t> img_data(chunk_B * tokens_per_image * TEXT_DIM);
        const size_t img_bytes = tokens_per_image * TEXT_DIM * sizeof(uint16_t);
        for (int i = 0; i < chunk_B; ++i) {
            std::memcpy(img_data.data() + i * tokens_per_image * TEXT_DIM,
                        embeds[chunk_start + i].data(), img_bytes);
        }
        auto batched_img = mx::array(reinterpret_cast<const mlx::core::float16_t*>(img_data.data()),
                                     {chunk_B, tokens_per_image, TEXT_DIM}, mx::float16);

        // 2. Broadcast prefix/suffix to batch dimension [chunk_B, tokens, dim]
        auto prefix_batched = mx::broadcast_to(prefix, {chunk_B, prefix.shape(1), TEXT_DIM});
        auto suffix_batched = mx::broadcast_to(suffix, {chunk_B, suffix.shape(1), TEXT_DIM});

        // 3. Single concatenate along sequence axis
        auto inputs_embeds = mx::concatenate({prefix_batched, batched_img, suffix_batched}, 1);

        // Create batched position IDs for this chunk
        // CRITICAL: MLX broadcast_to/tile create views that don't materialize correctly
        // when copied to CPU and read via pointer. Must build from raw data.
        std::vector<int32_t> batched_ids(chunk_B * num_tokens);
        mx::eval(input_ids);
        const int32_t* src_ids = input_ids.data<int32_t>();
        for (int b = 0; b < chunk_B; ++b) {
            std::memcpy(batched_ids.data() + b * num_tokens, src_ids, num_tokens * sizeof(int32_t));
        }
        auto chunk_input_ids = mx::array(batched_ids.data(), {chunk_B, num_tokens}, mx::int32);

        std::vector<GridTHW> grids;
        for (int i = 0; i < chunk_B; ++i) {
            grids.push_back({1, g_img_cfg.grid, g_img_cfg.grid});  // One grid per image
        }
        auto rope = get_rope_index(chunk_input_ids, grids, {}, empty_array());
        auto position_ids = rope.position_ids;

        // Generate for this chunk
        auto chunk_generated = generate_batched(inputs_embeds, &text, position_ids, config);

        // Accumulate results
        for (auto& gen : chunk_generated) {
            all_generated.push_back(std::move(gen));
        }

        if (num_chunks > 1) {
            int chunk_tokens = 0;
            for (int i = chunk_start; i < chunk_end; ++i) {
                chunk_tokens += all_generated[i].size();
            }
            std::cout << "Chunk generated " << chunk_tokens << " tokens" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    int total_tokens = 0;
    for (const auto& gen : all_generated) {
        total_tokens += gen.size();
    }

    print_generation_timing(ms, total_tokens);

    // Print results for each image
    for (int i = 0; i < B; ++i) {
        std::cout << "\n--- Image " << i << " (idx=" << indices[i] << ": " << filenames[indices[i]] << ") ---" << std::endl;
        std::cout << "Generated " << all_generated[i].size() << " tokens" << std::endl;
        std::cout << "Token IDs: ";
        for (size_t j = 0; j < all_generated[i].size(); ++j) {
            std::cout << all_generated[i][j];
            if (j < all_generated[i].size() - 1) std::cout << " ";
        }
        std::cout << std::endl;

        print_token_analysis(all_generated[i], false);
    }

    return 0;
}

// ============================================================================
// Print Usage Information
// ============================================================================

void print_usage() {
    std::cout << "GLM-4V MLX Inference Engine\n\n";
    std::cout << "Usage: Set one of the following environment variables:\n\n";
    std::cout << "Modes:\n";
    std::cout << "  GLM_PRECOMPUTE=1         Precompute vision embeddings from images\n";
    std::cout << "  GLM_GENERATE=1           Dual-image comparison (single or batched)\n";
    std::cout << "  GLM_DESCRIBE=1           Single-image description (single or batched)\n\n";
    std::cout << "GLM_GENERATE options:\n";
    std::cout << "  GLM_IDX_A=N GLM_IDX_B=M  Single pair (both required for single mode)\n";
    std::cout << "  GLM_PAIRS_FILE=path      Load pairs from file (idx_a,idx_b per line)\n";
    std::cout << "  GLM_BATCH_SIZE=N         Number of random pairs (default: 4)\n";
    std::cout << "  GLM_CHUNK_SIZE=N         Inference batch size (default: BATCH_SIZE)\n\n";
    std::cout << "GLM_DESCRIBE options:\n";
    std::cout << "  GLM_IDX=N                Single explicit image index\n";
    std::cout << "  GLM_INDICES_FILE=path    Load indices from file (one per line)\n";
    std::cout << "  GLM_RANDOM_INDICES=1     Use random indices instead of first N\n";
    std::cout << "  GLM_BATCH_SIZE=N         Number of images to process (default: 64)\n";
    std::cout << "  GLM_CHUNK_SIZE=N         Inference batch size (default: BATCH_SIZE)\n\n";
    std::cout << "Common options:\n";
    std::cout << "  GLM_IMAGE_SIZE=N         Image size in pixels (default: 448, also: 336)\n";
    std::cout << "  GLM_WEIGHTS_DIR=path     Model weights directory (default: vision_weights)\n";
    std::cout << "  GLM_EMBEDS_DIR=path      Embeddings directory (default: embeddings_out)\n";
    std::cout << "  GLM_MAX_TOKENS=N         Max tokens to generate (default: 100)\n\n";
    std::cout << "Sampling options:\n";
    std::cout << "  GLM_DO_SAMPLE=1          Enable sampling (default: greedy)\n";
    std::cout << "  GLM_TEMPERATURE=0.8      Temperature for sampling\n";
    std::cout << "  GLM_TOP_K=50             Top-K sampling\n";
    std::cout << "  GLM_TOP_P=0.9            Top-P (nucleus) sampling\n";
    std::cout << "  GLM_REP_PENALTY=1.1      Repetition penalty\n";
    std::cout << "  GLM_SEED=N               Random seed\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Precompute embeddings\n";
    std::cout << "  GLM_PRECOMPUTE=1 GLM_INPUT_DIR=images GLM_OUTPUT_DIR=embeds ./glm46v_mlx\n\n";
    std::cout << "  # Default: 4 random pairs\n";
    std::cout << "  GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds ./glm46v_mlx\n\n";
    std::cout << "  # Single explicit pair\n";
    std::cout << "  GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds GLM_IDX_A=0 GLM_IDX_B=1 ./glm46v_mlx\n\n";
    std::cout << "  # Custom batch size (8 random pairs)\n";
    std::cout << "  GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds GLM_BATCH_SIZE=8 ./glm46v_mlx\n\n";
    std::cout << "  # Load pairs from file\n";
    std::cout << "  GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds GLM_PAIRS_FILE=pairs.txt ./glm46v_mlx\n\n";
    std::cout << "  # Process 64 pairs in chunks of 8 (lower peak memory)\n";
    std::cout << "  GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds GLM_BATCH_SIZE=64 GLM_CHUNK_SIZE=8 ./glm46v_mlx\n\n";
    std::cout << "  # Describe first 64 images\n";
    std::cout << "  GLM_DESCRIBE=1 GLM_EMBEDS_DIR=embeds ./glm46v_mlx\n\n";
    std::cout << "  # Describe single image by index\n";
    std::cout << "  GLM_DESCRIBE=1 GLM_EMBEDS_DIR=embeds GLM_IDX=5 ./glm46v_mlx\n\n";
    std::cout << "  # Process 512 images in chunks of 128 (lower peak memory)\n";
    std::cout << "  GLM_DESCRIBE=1 GLM_EMBEDS_DIR=embeds GLM_BATCH_SIZE=512 GLM_CHUNK_SIZE=128 ./glm46v_mlx\n";
}

int main(int argc, char* argv[]) {
    auto gpu = mx::Device(mx::Device::gpu, 0);
    if (mx::is_available(gpu)) {
        mx::set_default_device(gpu);
    } else {
        mx::set_default_device(mx::Device(mx::Device::cpu, 0));
        std::cout << "GPU not available; using CPU." << std::endl;
    }

    // Enable MLX compilation for optimized graph execution
    mx::enable_compile();
    mx::set_compile_mode(mx::CompileMode::enabled);

    const char* weights_dir_env = std::getenv("GLM_WEIGHTS_DIR");

    // Initialize image config from GLM_IMAGE_SIZE (default: 448)
    const char* image_size_env = std::getenv("GLM_IMAGE_SIZE");
    int image_size = image_size_env ? std::atoi(image_size_env) : 448;
    g_img_cfg = ImageConfig::from_size(image_size);
    std::cout << "Image configuration:" << std::endl;
    g_img_cfg.print();

    // Mode: Precompute embeddings
    const char* precompute_env = std::getenv("GLM_PRECOMPUTE");
    if (precompute_env && std::string(precompute_env) == "1") {
        std::string weights_dir = weights_dir_env ? weights_dir_env : "vision_weights";
        const char* input_dir_env = std::getenv("GLM_INPUT_DIR");
        const char* output_dir_env = std::getenv("GLM_OUTPUT_DIR");
        const char* batch_env = std::getenv("GLM_BATCH_SIZE");

        std::string input_dir = input_dir_env ? input_dir_env : ".";
        std::string output_dir = output_dir_env ? output_dir_env : "embeddings_out";
        int batch_size = batch_env ? std::atoi(batch_env) : 8;

        return run_precompute_embeddings(input_dir, output_dir, weights_dir, batch_size);
    }

    // Mode: Dual-image comparison
    const char* generate_env = std::getenv("GLM_GENERATE");
    if (generate_env && std::string(generate_env) == "1") {
        std::string weights_dir = weights_dir_env ? weights_dir_env : "vision_weights";
        return run_generate_mode(weights_dir);
    }

    // Mode: Single-image description
    const char* describe_env = std::getenv("GLM_DESCRIBE");
    if (describe_env && std::string(describe_env) == "1") {
        std::string weights_dir = weights_dir_env ? weights_dir_env : "vision_weights";
        return run_describe_mode(weights_dir);
    }

    // No mode specified - print usage
    print_usage();
    return 0;
}
