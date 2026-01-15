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

// --- Compile-Time Constraints (GLM-4.6V vision) ---
#define VISION_DIM 1536
#define NUM_HEADS 12
#define HEAD_DIM (VISION_DIM / NUM_HEADS)
#define NUM_LAYERS 24
#define PATCH_SIZE 14
#define IMAGE_SIZE 336
#define PATCH_GRID (IMAGE_SIZE / PATCH_SIZE)
#define NUM_PATCHES (PATCH_GRID * PATCH_GRID) // 576
#define TEMPORAL_SIZE 2
#define TEMPORAL_PATCH 2
#define BLOCK_MLP_DIM 4096
#define MERGE_DIM 4096
#define MERGER_MLP_DIM 13696
#define SPATIAL_MERGE_SIZE 2
#define VISION_RMS_EPS 1e-5f
#define VISION_ROPE_THETA 10000.0f
#define PATCH_INPUT_DIM (3 * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE)
#ifndef GLM_FIXED_BATCH
#define GLM_FIXED_BATCH 32
#endif
#ifndef GLM_FIXED_ITERS
#define GLM_FIXED_ITERS 5
#endif
#ifndef GLM_FIXED_WARMUP
#define GLM_FIXED_WARMUP 2
#endif

// --- Compile-Time Constraints (GLM-4.6V language_model) ---
#define TEXT_DIM 4096
#define TEXT_NUM_HEADS 32
#define TEXT_HEAD_DIM (TEXT_DIM / TEXT_NUM_HEADS)
#define TEXT_KV_HEADS 2
#define TEXT_KV_DIM (TEXT_KV_HEADS * TEXT_HEAD_DIM)
#define TEXT_NUM_LAYERS 40
#define TEXT_MLP_DIM 13696
#define TEXT_GATE_UP_DIM (TEXT_MLP_DIM * 2)
#define TEXT_VOCAB_SIZE 151552
#define TEXT_RMS_EPS 1e-5f
#define TEXT_ROPE_THETA 10000.0f
#define TEXT_PARTIAL_ROTARY_FACTOR 0.5f
#define MROPE_SECTION_0 16
#define MROPE_SECTION_1 24
#define MROPE_SECTION_2 24
#define IMAGE_TOKEN_ID 151363
#define VIDEO_TOKEN_ID 151364
#define IMAGE_START_TOKEN_ID 151339
#define IMAGE_END_TOKEN_ID 151340
#define VIDEO_START_TOKEN_ID 151341
#define VIDEO_END_TOKEN_ID 151342
#ifndef GLM_TEXT_FIXED_BATCH
#define GLM_TEXT_FIXED_BATCH 16
#endif
#ifndef GLM_TEXT_FIXED_ITERS
#define GLM_TEXT_FIXED_ITERS 5
#endif
#ifndef GLM_TEXT_FIXED_WARMUP
#define GLM_TEXT_FIXED_WARMUP 2
#endif
#ifndef GLM_TEXT_SEQ_LEN
#define GLM_TEXT_SEQ_LEN 256
#endif
#ifndef GLM_RUN_TEXT
#define GLM_RUN_TEXT 0
#endif

namespace mx = mlx::core;

mx::array empty_array() { return mx::array({}); }

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

struct LinearWeights {
    mx::array weight = empty_array(); // [in, out]
    mx::array bias = empty_array();   // [out] or empty
};

struct RMSNormWeights {
    mx::array weight = empty_array();
};

struct LayerNormWeights {
    mx::array weight = empty_array();
    mx::array bias = empty_array();
};

struct AttentionWeights {
    LinearWeights qkv;
    LinearWeights proj;
};

struct MLPWeights {
    LinearWeights gate_proj;
    LinearWeights up_proj;
    LinearWeights down_proj;
};

struct VisionBlockWeights {
    RMSNormWeights norm1;
    RMSNormWeights norm2;
    AttentionWeights attn;
    MLPWeights mlp;
};

struct MergerWeights {
    LinearWeights proj;
    LayerNormWeights post_ln;
    LinearWeights gate_proj;
    LinearWeights up_proj;
    LinearWeights down_proj;
};

struct VisionWeights {
    mx::array patch_embed_weight = empty_array(); // [out, kd, kh, kw, in]
    mx::array patch_embed_bias = empty_array();
    mx::array pos_embed_weight = empty_array();   // [num_positions, dim]
    std::vector<VisionBlockWeights> blocks;
    RMSNormWeights post_layernorm;
    RMSNormWeights post_conv_layernorm;
    mx::array downsample_weight = empty_array();  // [out, kh, kw, in]
    mx::array downsample_bias = empty_array();
    MergerWeights merger;

    VisionWeights() : blocks(NUM_LAYERS) {}
};

struct TextAttentionWeights {
    LinearWeights q_proj;
    LinearWeights k_proj;
    LinearWeights v_proj;
    LinearWeights o_proj;
};

struct TextMLPWeights {
    LinearWeights gate_up_proj;
    LinearWeights down_proj;
};

struct TextLayerWeights {
    RMSNormWeights input_layernorm;
    RMSNormWeights post_self_attn_layernorm;  // After attention, before residual
    RMSNormWeights post_attention_layernorm;  // Before MLP
    RMSNormWeights post_mlp_layernorm;        // After MLP, before residual
    TextAttentionWeights self_attn;
    TextMLPWeights mlp;
};

struct TextModelWeights {
    mx::array embed_tokens = empty_array();
    std::vector<TextLayerWeights> layers;
    RMSNormWeights norm;
    LinearWeights lm_head;  // Added: language model head for logits

    TextModelWeights() : layers(TEXT_NUM_LAYERS) {}
};

struct GridTHW {
    int t;
    int h;
    int w;
};

struct VisionPositionData {
    mx::array cos = empty_array();
    mx::array sin = empty_array();
    mx::array pos_embed_indices = empty_array();
    mx::array image_type_h = empty_array();  // H coordinates for position embedding lookup
    mx::array image_type_w = empty_array();  // W coordinates for position embedding lookup
    std::vector<int> seqlens;
    int total_seq = 0;
};

mx::array randn(const mx::Shape& shape, mx::Dtype dtype, float stddev) {
    return mx::random::normal(shape, dtype, 0.0f, stddev);
}

void init_linear(LinearWeights* w, int in_dim, int out_dim, mx::Dtype dtype, float stddev, bool bias) {
    w->weight = randn({in_dim, out_dim}, dtype, stddev);
    if (bias) {
        w->bias = mx::zeros({out_dim}, dtype);
    }
}

void init_rms_norm(RMSNormWeights* w, int dim, mx::Dtype dtype) {
    w->weight = mx::ones({dim}, dtype);
}

void init_layer_norm(LayerNormWeights* w, int dim, mx::Dtype dtype) {
    w->weight = mx::ones({dim}, dtype);
    w->bias = mx::zeros({dim}, dtype);
}

void init_model(VisionWeights* model, mx::Dtype dtype = mx::float16, float stddev = 0.02f) {
    model->patch_embed_weight = randn({VISION_DIM, TEMPORAL_PATCH, PATCH_SIZE, PATCH_SIZE, 3}, dtype, stddev);
    model->patch_embed_bias = mx::zeros({VISION_DIM}, dtype);
    model->pos_embed_weight = randn({NUM_PATCHES, VISION_DIM}, dtype, stddev);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        auto& block = model->blocks[i];
        init_rms_norm(&block.norm1, VISION_DIM, dtype);
        init_rms_norm(&block.norm2, VISION_DIM, dtype);
        init_linear(&block.attn.qkv, VISION_DIM, VISION_DIM * 3, dtype, stddev, false);
        init_linear(&block.attn.proj, VISION_DIM, VISION_DIM, dtype, stddev, false);
        init_linear(&block.mlp.gate_proj, VISION_DIM, BLOCK_MLP_DIM, dtype, stddev, false);
        init_linear(&block.mlp.up_proj, VISION_DIM, BLOCK_MLP_DIM, dtype, stddev, false);
        init_linear(&block.mlp.down_proj, BLOCK_MLP_DIM, VISION_DIM, dtype, stddev, false);
    }

    init_rms_norm(&model->post_layernorm, VISION_DIM, dtype);
    init_rms_norm(&model->post_conv_layernorm, VISION_DIM, dtype);
    model->downsample_weight = randn({MERGE_DIM, SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE, VISION_DIM}, dtype, stddev);
    model->downsample_bias = mx::zeros({MERGE_DIM}, dtype);

    init_linear(&model->merger.proj, MERGE_DIM, MERGE_DIM, dtype, stddev, false);
    init_layer_norm(&model->merger.post_ln, MERGE_DIM, dtype);
    init_linear(&model->merger.gate_proj, MERGE_DIM, MERGER_MLP_DIM, dtype, stddev, false);
    init_linear(&model->merger.up_proj, MERGE_DIM, MERGER_MLP_DIM, dtype, stddev, false);
    init_linear(&model->merger.down_proj, MERGER_MLP_DIM, MERGE_DIM, dtype, stddev, false);
}

void init_text_model(TextModelWeights* model, mx::Dtype dtype = mx::float16, float stddev = 0.02f) {
    model->embed_tokens = randn({TEXT_VOCAB_SIZE, TEXT_DIM}, dtype, stddev);

    for (int i = 0; i < TEXT_NUM_LAYERS; ++i) {
        auto& layer = model->layers[i];
        init_rms_norm(&layer.input_layernorm, TEXT_DIM, dtype);
        init_rms_norm(&layer.post_attention_layernorm, TEXT_DIM, dtype);
        init_linear(&layer.self_attn.q_proj, TEXT_DIM, TEXT_DIM, dtype, stddev, true);
        init_linear(&layer.self_attn.k_proj, TEXT_DIM, TEXT_KV_DIM, dtype, stddev, true);
        init_linear(&layer.self_attn.v_proj, TEXT_DIM, TEXT_KV_DIM, dtype, stddev, true);
        init_linear(&layer.self_attn.o_proj, TEXT_DIM, TEXT_DIM, dtype, stddev, false);
        init_linear(&layer.mlp.gate_up_proj, TEXT_DIM, TEXT_GATE_UP_DIM, dtype, stddev, false);
        init_linear(&layer.mlp.down_proj, TEXT_MLP_DIM, TEXT_DIM, dtype, stddev, false);
    }

    init_rms_norm(&model->norm, TEXT_DIM, dtype);
    // Initialize lm_head: projects from TEXT_DIM to TEXT_VOCAB_SIZE (no bias)
    init_linear(&model->lm_head, TEXT_DIM, TEXT_VOCAB_SIZE, dtype, stddev, false);
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

// Load binary float32 array from file
mx::array load_binary_f32(const std::string& path, const mx::Shape& shape) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << path << std::endl;
        return empty_array();
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    int expected_size = 1;
    for (auto dim : shape) {
        expected_size *= dim;
    }
    expected_size *= sizeof(float);

    if (size != expected_size) {
        std::cerr << "Error: File size mismatch. Expected " << expected_size
                  << " bytes, got " << size << " bytes" << std::endl;
        return empty_array();
    }

    std::vector<float> data(expected_size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        std::cerr << "Error: Failed to read file: " << path << std::endl;
        return empty_array();
    }

    return mx::array(data.data(), shape, mx::float32);
}

// Compute max absolute difference between two arrays
float max_abs_diff(const mx::array& a, const mx::array& b) {
    auto diff = mx::abs(mx::subtract(mx::astype(a, mx::float32),
                                      mx::astype(b, mx::float32)));
    auto max_diff = mx::max(diff);
    mx::eval(max_diff);
    return max_diff.item<float>();
}

// Compute mean absolute difference between two arrays
float mean_abs_diff(const mx::array& a, const mx::array& b) {
    auto diff = mx::abs(mx::subtract(mx::astype(a, mx::float32),
                                      mx::astype(b, mx::float32)));
    auto mean_diff = mx::mean(diff);
    mx::eval(mean_diff);
    return mean_diff.item<float>();
}

// Print array statistics for debugging
void print_array_stats(const std::string& name, const mx::array& arr) {
    auto arr_f32 = mx::astype(arr, mx::float32);
    auto min_val = mx::min(arr_f32);
    auto max_val = mx::max(arr_f32);
    auto mean_val = mx::mean(arr_f32);
    mx::eval({min_val, max_val, mean_val});

    std::cout << name << ": shape=[";
    for (size_t i = 0; i < arr.shape().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << arr.shape()[i];
    }
    std::cout << "], min=" << min_val.item<float>()
              << ", max=" << max_val.item<float>()
              << ", mean=" << mean_val.item<float>() << std::endl;
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

mx::array repeat_kv(mx::array hidden_states, int n_rep) {
    if (n_rep == 1) {
        return hidden_states;
    }
    int B = hidden_states.shape(0);
    int kv_heads = hidden_states.shape(1);
    int T = hidden_states.shape(2);
    int D = hidden_states.shape(3);
    auto expanded = mx::reshape(hidden_states, {B, kv_heads, 1, T, D});
    expanded = mx::broadcast_to(expanded, {B, kv_heads, n_rep, T, D});
    return mx::reshape(expanded, {B, kv_heads * n_rep, T, D});
}

mx::array text_attention_forward(mx::array x, const TextAttentionWeights* w,
                                 const std::pair<mx::array, mx::array>& position_embeddings,
                                 const mx::array& attention_mask) {
    int B = x.shape(0);
    int T = x.shape(1);

    auto q = fast_linear(x, &w->q_proj);
    auto k = fast_linear(x, &w->k_proj);
    auto v = fast_linear(x, &w->v_proj);

    auto q_rs = mx::reshape(q, {B, T, TEXT_NUM_HEADS, TEXT_HEAD_DIM});
    auto k_rs = mx::reshape(k, {B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM});
    auto v_rs = mx::reshape(v, {B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM});

    q_rs = mx::transpose(q_rs, {0, 2, 1, 3});
    k_rs = mx::transpose(k_rs, {0, 2, 1, 3});
    v_rs = mx::transpose(v_rs, {0, 2, 1, 3});

    auto rotated = apply_multimodal_rotary_pos_emb(q_rs, k_rs, position_embeddings.first, position_embeddings.second);
    q_rs = rotated.first;
    k_rs = rotated.second;

    const int kv_repeat = TEXT_NUM_HEADS / TEXT_KV_HEADS;
    k_rs = repeat_kv(k_rs, kv_repeat);
    v_rs = repeat_kv(v_rs, kv_repeat);

    float scale = 1.0f / std::sqrt((float)TEXT_HEAD_DIM);
    std::optional<mx::array> mask_opt;
    if (attention_mask.size() != 0) {
        auto mask_f = mx::astype(attention_mask, mx::float32);
        auto inv_mask = mx::subtract(mx::array(1.0f), mask_f);
        auto neg_inf = mx::array(-1e9f);
        auto add_mask = mx::multiply(inv_mask, neg_inf);
        add_mask = mx::reshape(add_mask, {B, 1, 1, T});
        mask_opt = add_mask;
    }
    auto out = mx::fast::scaled_dot_product_attention(q_rs, k_rs, v_rs, scale, "causal", mask_opt);
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

mx::array text_layer_forward(mx::array x, const TextLayerWeights* w,
                             const std::pair<mx::array, mx::array>& position_embeddings,
                             const mx::array& attention_mask) {
    // GLM-4V uses 4 layer norms per layer (not the standard 2):
    // 1. input_layernorm - before attention
    // 2. post_self_attn_layernorm - after attention, before residual add
    // 3. post_attention_layernorm - before MLP
    // 4. post_mlp_layernorm - after MLP, before residual add

    // Attention path
    auto residual = x;
    x = rms_norm(x, &w->input_layernorm, TEXT_RMS_EPS);
    x = text_attention_forward(x, &w->self_attn, position_embeddings, attention_mask);
    x = rms_norm(x, &w->post_self_attn_layernorm, TEXT_RMS_EPS);
    x = mx::add(residual, x);

    // MLP path
    residual = x;
    x = rms_norm(x, &w->post_attention_layernorm, TEXT_RMS_EPS);
    x = text_mlp_forward(x, &w->mlp);
    x = rms_norm(x, &w->post_mlp_layernorm, TEXT_RMS_EPS);
    return mx::add(residual, x);
}

mx::array text_model_forward_embeds(mx::array inputs_embeds, const TextModelWeights* model,
                                    const mx::array& position_ids, const mx::array& attention_mask) {
    mx::array pos_ids = position_ids;
    if (pos_ids.size() == 0) {
        pos_ids = make_default_position_ids(inputs_embeds.shape(0), inputs_embeds.shape(1));
    }
    auto position_embeddings = text_rotary_embeddings(inputs_embeds, pos_ids);
    auto x = inputs_embeds;
    for (int i = 0; i < TEXT_NUM_LAYERS; ++i) {
        x = text_layer_forward(x, &model->layers[i], position_embeddings, attention_mask);
    }
    return rms_norm(x, &model->norm, TEXT_RMS_EPS);
}

mx::array text_model_forward_ids(mx::array input_ids, const TextModelWeights* model,
                                 const mx::array& position_ids, const mx::array& attention_mask) {
    auto embeds = mx::take(model->embed_tokens, input_ids, 0);
    return text_model_forward_embeds(embeds, model, position_ids, attention_mask);
}

// Compute logits from hidden states using lm_head
mx::array compute_logits(mx::array hidden_states, const TextModelWeights* model) {
    return fast_linear(hidden_states, &model->lm_head);
}

// Full forward pass that returns logits
mx::array text_model_forward_with_logits(mx::array input_ids, const TextModelWeights* model,
                                          const mx::array& position_ids, const mx::array& attention_mask) {
    auto hidden_states = text_model_forward_ids(input_ids, model, position_ids, attention_mask);
    return compute_logits(hidden_states, model);
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
    auto x = mx::reshape(patches, {num_patches, 3, TEMPORAL_PATCH, PATCH_SIZE, PATCH_SIZE});
    x = mx::transpose(x, {0, 2, 3, 4, 1});
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

mx::array vision_embeddings_forward(mx::array x, const VisionWeights* model,
                                    const VisionPositionData& pos) {
    auto pos_embed = mx::take(model->pos_embed_weight, pos.pos_embed_indices, 0);
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

struct PatchifyResult {
    mx::array patches = empty_array();
    std::vector<GridTHW> grid_thw;
};

PatchifyResult patchify_images(mx::array images) {
    PatchifyResult result;
    if (images.ndim() == 4) {
        int B = images.shape(0);
        int H = images.shape(1);
        int W = images.shape(2);
        int C = images.shape(3);
        images = mx::reshape(images, {B, 1, H, W, C});
    }

    int B = images.shape(0);
    int T = images.shape(1);
    int H = images.shape(2);
    int W = images.shape(3);
    int C = images.shape(4);

    if (H % PATCH_SIZE != 0 || W % PATCH_SIZE != 0) {
        std::cerr << "Image size must be divisible by patch size." << std::endl;
    }
    if ((H / PATCH_SIZE) % SPATIAL_MERGE_SIZE != 0 || (W / PATCH_SIZE) % SPATIAL_MERGE_SIZE != 0) {
        std::cerr << "Grid size must be divisible by spatial merge size." << std::endl;
    }

    if (T % TEMPORAL_PATCH != 0) {
        int pad = TEMPORAL_PATCH - (T % TEMPORAL_PATCH);
        auto last = mx::slice(images, {0, T - 1, 0, 0, 0}, {B, T, H, W, C});
        auto reps = mx::repeat(last, pad, 1);
        images = mx::concatenate({images, reps}, 1);
        T += pad;
    }

    int grid_t = T / TEMPORAL_PATCH;
    int grid_h = H / PATCH_SIZE;
    int grid_w = W / PATCH_SIZE;

    result.grid_thw.reserve(B);
    for (int i = 0; i < B; ++i) {
        result.grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    auto x = mx::transpose(images, {0, 1, 4, 2, 3}); // B, T, C, H, W
    int grid_h_blocks = grid_h / SPATIAL_MERGE_SIZE;
    int grid_w_blocks = grid_w / SPATIAL_MERGE_SIZE;
    x = mx::reshape(x, {B, grid_t, TEMPORAL_PATCH, C,
                        grid_h_blocks, SPATIAL_MERGE_SIZE, PATCH_SIZE,
                        grid_w_blocks, SPATIAL_MERGE_SIZE, PATCH_SIZE});
    x = mx::transpose(x, {0, 1, 4, 7, 5, 8, 3, 2, 6, 9});
    x = mx::reshape(x, {B, grid_t * grid_h * grid_w, PATCH_INPUT_DIM});
    result.patches = mx::reshape(x, {B * grid_t * grid_h * grid_w, PATCH_INPUT_DIM});
    return result;
}

struct RopeIndexResult {
    mx::array position_ids = empty_array();
    mx::array rope_deltas = empty_array();
};

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
                current_pos += llm_grid_t * llm_grid_h * llm_grid_w;
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
                current_pos += llm_grid_t * llm_grid_h * llm_grid_w;
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

std::pair<mx::array, mx::array> get_placeholder_mask(const mx::array& input_ids) {
    auto image_mask = mx::equal(input_ids, mx::array(IMAGE_TOKEN_ID));
    auto video_mask = mx::equal(input_ids, mx::array(IMAGE_TOKEN_ID));
    return {image_mask, video_mask};
}

mx::array custom_masked_scatter(mx::array base, const mx::array& mask, mx::array updates) {
    int B = base.shape(0);
    int T = base.shape(1);
    int D = base.shape(2);

    auto mask_flat = mx::reshape(mask, {B * T});
    auto mask_i = mx::astype(mask_flat, mx::int32);
    auto count_arr = mx::astype(mx::sum(mask_i), mx::int32);
    mx::eval(count_arr);
    int count = count_arr.data<int32_t>()[0];
    if (count == 0) {
        return base;
    }
    if (updates.shape(0) != count) {
        std::cerr << "Masked scatter mismatch: mask count " << count
                  << " vs updates " << updates.shape(0) << std::endl;
    }

    auto idx = mx::astype(mx::arange(B * T), mx::int32);
    auto masked_idx = mx::where(mask_flat, mx::add(idx, mx::array(1, mx::int32)),
                                mx::array(0, mx::int32));
    auto sorted_idx = mx::sort(masked_idx);
    int start = (B * T) - count;
    auto valid_idx = mx::slice(sorted_idx, {start}, {B * T});
    valid_idx = mx::subtract(valid_idx, mx::array(1, mx::int32));

    auto flat = mx::reshape(base, {B * T, D});
    auto out = mx::scatter(flat, {valid_idx}, updates, std::vector<int>{0});
    return mx::reshape(out, {B, T, D});
}

struct ImageFeaturesResult {
    mx::array embeds = empty_array();
    std::vector<int> split_sizes;
    std::vector<GridTHW> grid_thw;
};

ImageFeaturesResult get_image_features(mx::array pixel_values, const VisionWeights* model,
                                       const std::vector<GridTHW>& image_grid_thw) {
    ImageFeaturesResult result;
    std::vector<GridTHW> grids = image_grid_thw;
    mx::array patches = pixel_values;

    if (pixel_values.ndim() != 2) {
        auto patchified = patchify_images(pixel_values);
        patches = patchified.patches;
        if (grids.empty()) {
            grids = patchified.grid_thw;
        }
    }

    if (grids.empty()) {
        std::cerr << "image_grid_thw is required when passing flattened patches." << std::endl;
    }

    result.grid_thw = grids;
    result.embeds = vision_forward(patches, model, grids);

    int total_tokens = 0;
    result.split_sizes.reserve(grids.size());
    for (const auto& grid : grids) {
        int tokens = (grid.t * grid.h * grid.w) / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE);
        result.split_sizes.push_back(tokens);
        total_tokens += tokens;
    }
    if (result.embeds.shape(0) != total_tokens) {
        std::cerr << "Vision token count mismatch: got " << result.embeds.shape(0)
                  << " expected " << total_tokens << std::endl;
    }
    return result;
}

// Forward pass returning hidden states (for intermediate use)
mx::array glm4v_forward(mx::array input_ids,
                        mx::array pixel_values,
                        const std::vector<GridTHW>& image_grid_thw,
                        const mx::array& attention_mask,
                        const VisionWeights* vision,
                        const TextModelWeights* text) {
    auto inputs_embeds = mx::take(text->embed_tokens, input_ids, 0);

    std::vector<GridTHW> grids = image_grid_thw;
    if (pixel_values.size() != 0) {
        auto image_feats = get_image_features(pixel_values, vision, grids);
        grids = image_feats.grid_thw;
        auto masks = get_placeholder_mask(input_ids);
        inputs_embeds = custom_masked_scatter(inputs_embeds, masks.first, image_feats.embeds);
    }

    mx::array position_ids = empty_array();
    if (!grids.empty()) {
        auto rope = get_rope_index(input_ids, grids, {}, attention_mask);
        position_ids = rope.position_ids;
    }

    return text_model_forward_embeds(inputs_embeds, text, position_ids, attention_mask);
}

// Forward pass returning logits (for generation/inference)
mx::array glm4v_forward_logits(mx::array input_ids,
                               mx::array pixel_values,
                               const std::vector<GridTHW>& image_grid_thw,
                               const mx::array& attention_mask,
                               const VisionWeights* vision,
                               const TextModelWeights* text,
                               int logits_to_keep = 0) {
    auto hidden_states = glm4v_forward(input_ids, pixel_values, image_grid_thw,
                                        attention_mask, vision, text);

    // If logits_to_keep > 0, only compute logits for last N tokens (for efficiency)
    if (logits_to_keep > 0) {
        int T = hidden_states.shape(1);
        int start = std::max(0, T - logits_to_keep);
        hidden_states = mx::slice(hidden_states, {0, start, 0},
                                  {hidden_states.shape(0), T, hidden_states.shape(2)});
    }

    return compute_logits(hidden_states, text);
}

int run_vision_bench() {
    const int B = GLM_FIXED_BATCH;
    const int iters = GLM_FIXED_ITERS;
    const int warmup = GLM_FIXED_WARMUP;
    auto input = mx::random::uniform({B, IMAGE_SIZE, IMAGE_SIZE, 3}, mx::float16);
    auto patchified = patchify_images(input);

    VisionWeights model;
    init_model(&model, mx::float16);

    mx::enable_compile();
    mx::set_compile_mode(mx::CompileMode::enabled);

    auto compiled = mx::compile([&model, grid_thw = patchified.grid_thw](const std::vector<mx::array>& inputs) {
        auto out = vision_forward(inputs[0], &model, grid_thw);
        return std::vector<mx::array>{out};
    });

    std::vector<mx::array> inputs{patchified.patches};

    std::cout << "Compiling..." << std::endl;
    auto first = compiled(inputs);
    mx::eval(first);
    mx::synchronize();

    for (int i = 0; i < warmup; ++i) {
        auto out = compiled(inputs)[0];
        mx::eval(out);
    }
    mx::synchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto out = compiled(inputs)[0];
        mx::eval(out);
    }
    mx::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double images_per_sec = (double)(B * iters) / elapsed.count();

    auto output = compiled(inputs)[0];
    mx::eval(output);
    mx::synchronize();

    // Vision output is 2D: [merge_tokens, MERGE_DIM]
    std::cout << "Output shape: " << output.shape(0) << "x" << output.shape(1) << std::endl;
    std::cout << "Throughput: " << images_per_sec << " images/s" << std::endl;
    return 0;
}

int run_text_bench() {
    const int B = GLM_TEXT_FIXED_BATCH;
    const int T = GLM_TEXT_SEQ_LEN;
    const int iters = GLM_TEXT_FIXED_ITERS;
    const int warmup = GLM_TEXT_FIXED_WARMUP;

    std::cout << "Initializing text model (random weights)..." << std::endl;
    TextModelWeights model;
    init_text_model(&model, mx::float16);

    std::cout << "Creating inputs (B=" << B << ", T=" << T << ")..." << std::endl;
    auto input_ids = mx::random::randint(0, TEXT_VOCAB_SIZE, {B, T}, mx::int32);
    auto position_ids = make_default_position_ids(B, T);
    mx::eval({input_ids, position_ids});

    std::cout << "Warmup..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        auto out = text_model_forward_ids(input_ids, &model, position_ids, empty_array());
        mx::eval(out);
    }
    mx::synchronize();

    std::cout << "Benchmarking..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto out = text_model_forward_ids(input_ids, &model, position_ids, empty_array());
        mx::eval(out);
    }
    mx::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double tokens_per_sec = (double)(B * T * iters) / elapsed.count();

    auto output = text_model_forward_ids(input_ids, &model, position_ids, empty_array());
    mx::eval(output);
    mx::synchronize();

    std::cout << "Output shape: " << output.shape(0) << "x" << output.shape(1)
              << "x" << output.shape(2) << std::endl;
    std::cout << "Throughput: " << tokens_per_sec << " tokens/s" << std::endl;
    return 0;
}

// ==================== Vision Verification ====================

int run_vision_verify(const std::string& weights_dir) {
    std::cout << "=== Vision Encoder Verification ===" << std::endl;

    // Paths
    std::string raw_path = weights_dir + "/vision_encoder.bin";
    std::string patches_path = weights_dir + "/test_patches.bin";
    std::string expected_path = weights_dir + "/expected_output.bin";

    // Load vision model weights from raw binary via mmap
    VisionWeights model;
    if (!load_vision_weights(&model, raw_path)) {
        std::cerr << "Failed to load vision weights" << std::endl;
        std::cerr << "Make sure to run: python3 export_vision_weights.py --format raw" << std::endl;
        return 1;
    }

    // Print some weight shapes to verify loading
    std::cout << "\nLoaded weight shapes:" << std::endl;
    print_array_stats("  patch_embed.weight", model.patch_embed_weight);
    print_array_stats("  pos_embed.weight", model.pos_embed_weight);
    print_array_stats("  blocks.0.norm1.weight", model.blocks[0].norm1.weight);
    print_array_stats("  merger.proj.weight", model.merger.proj.weight);

    // Load test patches
    // For a 336x336 image with patch_size=14, temporal_patch=2:
    // grid_t=1, grid_h=24, grid_w=24 -> 576 patches
    // Each patch has PATCH_INPUT_DIM = 3 * 2 * 14 * 14 = 1176 elements
    int num_patches = 576;  // 24 * 24 * 1
    int patch_dim = PATCH_INPUT_DIM;

    auto patches = load_binary_f32(patches_path, {num_patches, patch_dim});
    if (patches.size() == 0) {
        std::cerr << "Failed to load test patches" << std::endl;
        return 1;
    }
    print_array_stats("\nInput patches", patches);

    // Load expected output
    // After merging: 576 / 4 = 144 tokens, each with MERGE_DIM=4096
    int output_tokens = num_patches / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE);
    auto expected = load_binary_f32(expected_path, {output_tokens, MERGE_DIM});
    if (expected.size() == 0) {
        std::cerr << "Failed to load expected output" << std::endl;
        return 1;
    }
    print_array_stats("Expected output", expected);

    // Create grid_thw for single image
    std::vector<GridTHW> grid_thw = {{1, 24, 24}};  // t=1, h=24, w=24

    // Run forward pass
    std::cout << "\nRunning C++ forward pass..." << std::endl;
    auto output = vision_forward(patches, &model, grid_thw);
    mx::eval(output);
    mx::synchronize();

    print_array_stats("C++ output", output);

    // Compare outputs
    std::cout << "\n=== Comparison Results ===" << std::endl;

    float max_diff = max_abs_diff(output, expected);
    float mean_diff = mean_abs_diff(output, expected);

    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Mean absolute difference: " << mean_diff << std::endl;

    // Check if within tolerance
    const float tolerance = 1e-3f;  // Tolerance for float32 comparison
    if (max_diff < tolerance) {
        std::cout << "\n[PASS] Outputs match within tolerance (" << tolerance << ")" << std::endl;
        return 0;
    } else {
        std::cout << "\n[FAIL] Outputs differ by more than tolerance (" << tolerance << ")" << std::endl;

        // Debug: print first few values
        std::cout << "\nFirst 10 output values comparison:" << std::endl;
        auto out_f32 = mx::astype(output, mx::float32);
        auto exp_f32 = mx::astype(expected, mx::float32);
        mx::eval({out_f32, exp_f32});

        auto out_flat = mx::reshape(out_f32, {-1});
        auto exp_flat = mx::reshape(exp_f32, {-1});
        mx::eval({out_flat, exp_flat});

        // Copy to CPU for printing
        out_flat = mx::copy(out_flat, mx::Device(mx::Device::cpu, 0));
        exp_flat = mx::copy(exp_flat, mx::Device(mx::Device::cpu, 0));
        mx::eval({out_flat, exp_flat});

        const float* out_ptr = out_flat.data<float>();
        const float* exp_ptr = exp_flat.data<float>();

        std::cout << "  Index\tC++\t\tPython\t\tDiff" << std::endl;
        for (int i = 0; i < 10 && i < out_flat.size(); ++i) {
            std::cout << "  " << i << "\t" << out_ptr[i] << "\t"
                      << exp_ptr[i] << "\t" << std::abs(out_ptr[i] - exp_ptr[i]) << std::endl;
        }

        return 1;
    }
}

// ==================== Text Model Verification ====================

int run_text_verify(const std::string& weights_dir) {
    std::cout << "=== Text Model Verification ===" << std::endl;

    // Paths
    std::string weights_path = weights_dir + "/text_model.bin";
    std::string input_ids_path = weights_dir + "/test_input_ids.bin";
    std::string expected_hidden_path = weights_dir + "/expected_text_hidden.bin";
    std::string expected_logits_path = weights_dir + "/expected_text_logits.bin";

    // Load text model weights
    TextModelWeights model;
    if (!load_text_weights(&model, weights_path)) {
        std::cerr << "Failed to load text weights" << std::endl;
        std::cerr << "Make sure to run: python3 export_vision_weights.py --text" << std::endl;
        return 1;
    }

    // Print some weight shapes to verify loading
    std::cout << "\nLoaded weight shapes:" << std::endl;
    print_array_stats("  embed_tokens", model.embed_tokens);
    print_array_stats("  layers.0.input_layernorm", model.layers[0].input_layernorm.weight);
    print_array_stats("  layers.0.q_proj.weight", model.layers[0].self_attn.q_proj.weight);
    print_array_stats("  norm.weight", model.norm.weight);
    print_array_stats("  lm_head.weight", model.lm_head.weight);

    // Load test input_ids
    // Shape: [1, 32] for batch_size=1, seq_len=32
    const int batch_size = 1;
    const int seq_len = 32;

    std::ifstream ids_file(input_ids_path, std::ios::binary);
    if (!ids_file.is_open()) {
        std::cerr << "Failed to open: " << input_ids_path << std::endl;
        return 1;
    }

    std::vector<int32_t> input_ids_data(batch_size * seq_len);
    ids_file.read(reinterpret_cast<char*>(input_ids_data.data()), batch_size * seq_len * sizeof(int32_t));
    ids_file.close();

    auto input_ids = mx::array(input_ids_data.data(), {batch_size, seq_len}, mx::int32);
    std::cout << "\nInput IDs loaded: [" << batch_size << ", " << seq_len << "]" << std::endl;

    // Load expected hidden states
    auto expected_hidden = load_binary_f32(expected_hidden_path, {batch_size, seq_len, TEXT_DIM});
    if (expected_hidden.size() == 0) {
        std::cerr << "Failed to load expected hidden states" << std::endl;
        return 1;
    }
    print_array_stats("Expected hidden", expected_hidden);

    // Load expected logits (optional, for full verification)
    auto expected_logits = load_binary_f32(expected_logits_path, {batch_size, seq_len, TEXT_VOCAB_SIZE});
    bool have_logits = (expected_logits.size() != 0);
    if (have_logits) {
        print_array_stats("Expected logits", expected_logits);
    }

    // Step verification: Check embeddings first
    std::cout << "\n=== Step Verification ===" << std::endl;

    // Load expected embeddings and layer 0 output
    auto expected_embeds = load_binary_f32(weights_dir + "/expected_embeddings.bin", {batch_size, seq_len, TEXT_DIM});
    auto expected_layer0 = load_binary_f32(weights_dir + "/expected_layer0.bin", {batch_size, seq_len, TEXT_DIM});

    // Step 1: Check embedding lookup
    auto embeddings = mx::take(model.embed_tokens, input_ids, 0);
    mx::eval(embeddings);
    print_array_stats("C++ embeddings", embeddings);
    if (expected_embeds.size() > 0) {
        float embed_diff = max_abs_diff(embeddings, expected_embeds);
        std::cout << "Embedding max diff: " << embed_diff << (embed_diff < 0.01 ? " [OK]" : " [MISMATCH]") << std::endl;
    }

    // Step 2: Check first layer output - with detailed debugging
    auto position_ids = make_default_position_ids(batch_size, seq_len);
    auto position_embeddings = text_rotary_embeddings(embeddings, position_ids);

    // Debug: trace through layer 0 step by step
    auto& layer0 = model.layers[0];

    // Helper to print first N values
    auto print_first_n = [](const mx::array& arr, int n, const std::string& label) {
        auto f32 = mx::astype(arr, mx::float32);
        mx::eval(f32);
        auto cpu = mx::copy(f32, mx::Device(mx::Device::cpu, 0));
        mx::eval(cpu);
        std::cout << "      " << label << " first " << n << ": ";
        const float* data = cpu.data<float>();
        for (int i = 0; i < n; ++i) std::cout << data[i] << " ";
        std::cout << std::endl;
    };

    // 2a: RMS norm on embeddings
    auto x_norm = rms_norm(embeddings, &layer0.input_layernorm, TEXT_RMS_EPS);
    mx::eval(x_norm);
    print_array_stats("  2a. After input_layernorm", x_norm);
    print_first_n(x_norm, 5, "values");

    // 2b: Q, K, V projections
    auto q = fast_linear(x_norm, &layer0.self_attn.q_proj);
    auto k = fast_linear(x_norm, &layer0.self_attn.k_proj);
    auto v = fast_linear(x_norm, &layer0.self_attn.v_proj);
    mx::eval({q, k, v});
    print_array_stats("  2b. Q projection", q);
    print_first_n(q, 5, "Q");
    print_array_stats("      K projection", k);
    print_first_n(k, 5, "K");
    print_array_stats("      V projection", v);
    print_first_n(v, 5, "V");

    // Debug: Q, K, V at token 13
    auto print_token13 = [](const mx::array& arr, const std::string& label) {
        auto f32 = mx::astype(arr, mx::float32);
        mx::eval(f32);
        auto cpu = mx::copy(f32, mx::Device(mx::Device::cpu, 0));
        mx::eval(cpu);
        const float* data = cpu.data<float>();
        int D = arr.shape(2);
        std::cout << "    " << label << " @ token 13, first 5 dims: ";
        for (int d = 0; d < 5; ++d) {
            std::cout << data[13 * D + d] << " ";
        }
        std::cout << std::endl;
    };
    std::cout << "\n  Debug Q/K/V at token 13:" << std::endl;
    print_token13(x_norm, "x_norm");
    print_token13(q, "Q");
    print_token13(k, "K");
    print_token13(v, "V");

    // Debug: position embeddings at token 13
    auto print_pos_emb = [](const std::pair<mx::array, mx::array>& pe, int token, const std::string& label) {
        auto cos = pe.first;  // [3, B, T, rotary_dim]
        auto sin = pe.second;
        auto cos_f32 = mx::astype(cos, mx::float32);
        auto sin_f32 = mx::astype(sin, mx::float32);
        mx::eval({cos_f32, sin_f32});
        auto cos_cpu = mx::copy(cos_f32, mx::Device(mx::Device::cpu, 0));
        auto sin_cpu = mx::copy(sin_f32, mx::Device(mx::Device::cpu, 0));
        mx::eval({cos_cpu, sin_cpu});
        const float* cos_data = cos_cpu.data<float>();
        const float* sin_data = sin_cpu.data<float>();
        // Shape is [3, B, T, rotary_dim], print section 0 at token
        int B = cos.shape(1);
        int T = cos.shape(2);
        int D = cos.shape(3);
        std::cout << "    " << label << " shape: [" << cos.shape(0) << ", " << B << ", " << T << ", " << D << "]" << std::endl;
        std::cout << "    " << label << " cos @ sec0, token " << token << ", first 5 dims: ";
        for (int d = 0; d < 5; ++d) {
            std::cout << cos_data[0 * B * T * D + 0 * T * D + token * D + d] << " ";
        }
        std::cout << std::endl;
        std::cout << "    " << label << " sin @ sec0, token " << token << ", first 5 dims: ";
        for (int d = 0; d < 5; ++d) {
            std::cout << sin_data[0 * B * T * D + 0 * T * D + token * D + d] << " ";
        }
        std::cout << std::endl;
    };
    print_pos_emb(position_embeddings, 13, "pos_emb");

    // Debug: check MROPE output (after apply_multimodal_rotary_pos_emb internal processing)
    {
        // Convert to float32 first for proper debugging
        auto cos = mx::astype(position_embeddings.first, mx::float32);  // [3, B, T, 128]
        int B_dbg = cos.shape(1);
        int T_dbg = cos.shape(2);

        std::vector<int> chunk_sizes = {16, 24, 24, 16, 24, 24};
        mx::Shape split_indices;
        int cumsum = 0;
        for (int i = 0; i < 5; ++i) {
            cumsum += chunk_sizes[i];
            split_indices.push_back(cumsum);
        }
        auto cos_chunks = mx::split(cos, split_indices, 3);

        std::vector<mx::array> cos_parts;
        for (int i = 0; i < 6; ++i) {
            int sec = i % 3;
            int chunk_dim = chunk_sizes[i];
            auto cos_part = mx::slice(cos_chunks[i], {sec, 0, 0, 0}, {sec + 1, B_dbg, T_dbg, chunk_dim});
            cos_parts.push_back(mx::squeeze(cos_part, std::vector<int>{0}));
        }
        auto cos_cat = mx::concatenate(cos_parts, 2);
        cos_cat = mx::reshape(cos_cat, {B_dbg, 1, T_dbg, cos_cat.shape(2)});
        int half = cos_cat.shape(3) / 2;
        auto cos_half = mx::slice(cos_cat, {0, 0, 0, 0}, {B_dbg, 1, T_dbg, half});
        cos_half = mx::reshape(cos_half, {B_dbg, 1, T_dbg, half, 1});
        auto cos_interleaved = mx::tile(cos_half, {1, 1, 1, 1, 2});
        cos_interleaved = mx::reshape(cos_interleaved, {B_dbg, 1, T_dbg, half * 2});
        mx::eval(cos_interleaved);
        auto cpu = mx::copy(cos_interleaved, mx::Device(mx::Device::cpu, 0));
        mx::eval(cpu);
        const float* data = cpu.data<float>();
        int D = cos_interleaved.shape(3);
        std::cout << "    MROPE cos_interleaved @ token 13, first 10: ";
        for (int d = 0; d < 10; ++d) std::cout << data[0 * 1 * T_dbg * D + 0 * T_dbg * D + 13 * D + d] << " ";
        std::cout << std::endl;
    }

    // Debug: check freqs before cos/sin
    {
        int rotary_dim = TEXT_HEAD_DIM;
        auto inv_idx = mx::astype(mx::arange(0, rotary_dim, 2), mx::float32);
        auto inv_freq = mx::divide(
            mx::array(1.0f),
            mx::power(mx::array(TEXT_ROPE_THETA),
                      mx::divide(inv_idx, mx::array((float)rotary_dim))));
        mx::eval(inv_freq);
        auto inv_cpu = mx::copy(inv_freq, mx::Device(mx::Device::cpu, 0));
        mx::eval(inv_cpu);
        std::cout << "    inv_freq first 5: ";
        for (int i = 0; i < 5; ++i) std::cout << inv_cpu.data<float>()[i] << " ";
        std::cout << std::endl;

        // Manual freqs at token 13: inv_freq * 13
        std::cout << "    expected freqs @ token 13, first 5: ";
        for (int i = 0; i < 5; ++i) std::cout << inv_cpu.data<float>()[i] * 13 << " ";
        std::cout << std::endl;
    }

    // Debug: compute Q_rot and K_rot manually
    {
        int B = 1, T = 32;
        auto q_rs = mx::reshape(q, {B, T, TEXT_NUM_HEADS, TEXT_HEAD_DIM});
        auto k_rs = mx::reshape(k, {B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM});
        q_rs = mx::transpose(q_rs, {0, 2, 1, 3});  // [B, H, T, D]
        k_rs = mx::transpose(k_rs, {0, 2, 1, 3});

        auto rotated = apply_multimodal_rotary_pos_emb(q_rs, k_rs, position_embeddings.first, position_embeddings.second);
        auto q_rot = rotated.first;
        auto k_rot = rotated.second;

        // Print Q_rot at token 13, head 0, first 10 dims
        auto q_rot_f32 = mx::astype(q_rot, mx::float32);
        auto k_rot_f32 = mx::astype(k_rot, mx::float32);
        mx::eval({q_rot_f32, k_rot_f32});
        auto q_rot_cpu = mx::copy(q_rot_f32, mx::Device(mx::Device::cpu, 0));
        auto k_rot_cpu = mx::copy(k_rot_f32, mx::Device(mx::Device::cpu, 0));
        mx::eval({q_rot_cpu, k_rot_cpu});

        // q_rot shape: [B, H, T, D] = [1, 32, 32, 128]
        // Access [0, 0, 13, :10]
        const float* q_data = q_rot_cpu.data<float>();
        const float* k_data = k_rot_cpu.data<float>();
        std::cout << "    Q_rot @ token 13, head 0, first 10 dims: ";
        for (int d = 0; d < 10; ++d) {
            // Index: [b, h, t, d] = [0, 0, 13, d]
            // Offset: b * H * T * D + h * T * D + t * D + d = 0 * 32 * 32 * 128 + 0 * 32 * 128 + 13 * 128 + d
            std::cout << q_data[13 * 128 + d] << " ";
        }
        std::cout << std::endl;
        std::cout << "    K_rot @ token 13, head 0, first 10 dims: ";
        for (int d = 0; d < 10; ++d) {
            std::cout << k_data[13 * 128 + d] << " ";
        }
        std::cout << std::endl;
    }

    // 2c: Attention output (before post_self_attn_layernorm)
    auto attn_out = text_attention_forward(x_norm, &layer0.self_attn, position_embeddings, empty_array());
    mx::eval(attn_out);
    print_array_stats("  2c. Attention output (raw)", attn_out);
    print_first_n(attn_out, 5, "attn_out");

    // 2c': Apply post_self_attn_layernorm to attention output (before residual)
    auto attn_normed = rms_norm(attn_out, &layer0.post_self_attn_layernorm, TEXT_RMS_EPS);
    mx::eval(attn_normed);
    print_array_stats("  2c'. After post_self_attn_layernorm", attn_normed);
    print_first_n(attn_normed, 5, "attn_normed");

    // 2d: After attention residual
    auto after_attn = mx::add(embeddings, attn_normed);
    mx::eval(after_attn);
    print_array_stats("  2d. After attention residual", after_attn);
    print_first_n(after_attn, 5, "after_attn");

    // 2e: RMS norm before MLP
    auto mlp_norm = rms_norm(after_attn, &layer0.post_attention_layernorm, TEXT_RMS_EPS);
    mx::eval(mlp_norm);
    print_array_stats("  2e. After post_attn_layernorm", mlp_norm);
    print_first_n(mlp_norm, 5, "mlp_norm");

    // 2f: MLP gate_up output
    auto gate_up = fast_linear(mlp_norm, &layer0.mlp.gate_up_proj);
    mx::eval(gate_up);
    print_array_stats("  2f. gate_up projection", gate_up);
    print_first_n(gate_up, 5, "gate_up");

    // 2g: MLP output (before post_mlp_layernorm)
    auto mlp_out = text_mlp_forward(mlp_norm, &layer0.mlp);
    mx::eval(mlp_out);
    print_array_stats("  2g. MLP output (raw)", mlp_out);
    print_first_n(mlp_out, 5, "mlp_out");

    // 2g': Apply post_mlp_layernorm to MLP output (before residual)
    auto mlp_normed = rms_norm(mlp_out, &layer0.post_mlp_layernorm, TEXT_RMS_EPS);
    mx::eval(mlp_normed);
    print_array_stats("  2g'. After post_mlp_layernorm", mlp_normed);
    print_first_n(mlp_normed, 5, "mlp_normed");

    // 2h: Full layer output
    auto layer0_out = mx::add(after_attn, mlp_normed);
    mx::eval(layer0_out);
    print_array_stats("  2h. Layer 0 output", layer0_out);
    print_first_n(layer0_out, 5, "layer0_out");

    // Debug: print values at token 13 (where max diff occurs)
    auto print_token_dim = [](const mx::array& arr, int token, int dim_start, const std::string& label) {
        auto f32 = mx::astype(arr, mx::float32);
        mx::eval(f32);
        auto cpu = mx::copy(f32, mx::Device(mx::Device::cpu, 0));
        mx::eval(cpu);
        const float* data = cpu.data<float>();
        int D = arr.shape(2);
        std::cout << "      " << label << " token " << token << ", dims " << dim_start << "-" << (dim_start+4) << ": ";
        for (int d = dim_start; d < dim_start + 5; ++d) {
            std::cout << data[token * D + d] << " ";
        }
        std::cout << std::endl;
    };
    std::cout << "\n  Debug: Values at token 13 (max diff position):" << std::endl;
    print_token_dim(attn_out, 13, 0, "attn_out (dims 0-4)");
    print_token_dim(attn_out, 13, 3556, "attn_out (dims 3556-3560)");
    print_token_dim(attn_normed, 13, 3556, "attn_normed");
    print_token_dim(after_attn, 13, 3556, "after_attn");
    print_token_dim(mlp_norm, 13, 3556, "mlp_norm");
    print_token_dim(mlp_out, 13, 3556, "mlp_out");
    print_token_dim(mlp_normed, 13, 3556, "mlp_normed");
    print_token_dim(layer0_out, 13, 3556, "layer0_out");

    // Check where max diff occurs
    if (expected_layer0.size() > 0) {
        float layer0_diff = max_abs_diff(layer0_out, expected_layer0);
        std::cout << "Layer 0 max diff: " << layer0_diff << (layer0_diff < 0.1 ? " [OK]" : " [MISMATCH]") << std::endl;

        // Find the position of max difference
        auto l0_f32 = mx::astype(layer0_out, mx::float32);
        auto exp_f32 = mx::astype(expected_layer0, mx::float32);
        mx::eval({l0_f32, exp_f32});
        auto l0_cpu = mx::copy(l0_f32, mx::Device(mx::Device::cpu, 0));
        auto exp_cpu = mx::copy(exp_f32, mx::Device(mx::Device::cpu, 0));
        mx::eval({l0_cpu, exp_cpu});

        const float* l0_data = l0_cpu.data<float>();
        const float* exp_data = exp_cpu.data<float>();
        size_t total = 32 * 4096;  // seq_len * hidden_dim
        size_t max_idx = 0;
        float max_d = 0;
        for (size_t i = 0; i < total; ++i) {
            float d = std::abs(l0_data[i] - exp_data[i]);
            if (d > max_d) { max_d = d; max_idx = i; }
        }
        int token = max_idx / 4096;
        int dim = max_idx % 4096;
        std::cout << "Max diff at token " << token << ", dim " << dim
                  << ": C++=" << l0_data[max_idx] << ", Python=" << exp_data[max_idx] << std::endl;

        // Print a range around the max diff position
        std::cout << "Values around max diff (token " << token << ", dims " << std::max(0, dim-2) << "-" << std::min(4095, dim+2) << "):" << std::endl;
        for (int d = std::max(0, dim-2); d <= std::min(4095, dim+2); ++d) {
            size_t idx = token * 4096 + d;
            std::cout << "  dim " << d << ": C++=" << l0_data[idx] << ", Python=" << exp_data[idx] << std::endl;
        }
    }

    // Run full forward pass
    std::cout << "\nRunning full C++ forward pass..." << std::endl;
    auto hidden_states = text_model_forward_ids(input_ids, &model, position_ids, empty_array());
    mx::eval(hidden_states);
    mx::synchronize();

    print_array_stats("C++ hidden states", hidden_states);

    // Compare hidden states
    std::cout << "\n=== Hidden States Comparison ===" << std::endl;

    float max_diff_hidden = max_abs_diff(hidden_states, expected_hidden);
    float mean_diff_hidden = mean_abs_diff(hidden_states, expected_hidden);

    std::cout << "Max absolute difference: " << max_diff_hidden << std::endl;
    std::cout << "Mean absolute difference: " << mean_diff_hidden << std::endl;

    // If we have logits, verify those too
    if (have_logits) {
        std::cout << "\n=== Logits Comparison ===" << std::endl;
        auto logits = compute_logits(hidden_states, &model);
        mx::eval(logits);
        mx::synchronize();

        print_array_stats("C++ logits", logits);

        float max_diff_logits = max_abs_diff(logits, expected_logits);
        float mean_diff_logits = mean_abs_diff(logits, expected_logits);

        std::cout << "Max absolute difference: " << max_diff_logits << std::endl;
        std::cout << "Mean absolute difference: " << mean_diff_logits << std::endl;
    }

    // Check if within tolerance
    // For 40-layer float16 model, accumulated precision error is expected
    // Layer 0 matches closely (~0.001), but error accumulates over layers
    const float tolerance = 0.5f;  // Reasonable for float16 deep model (40 layers)
    if (max_diff_hidden < tolerance) {
        std::cout << "\n[PASS] Hidden states match within tolerance (" << tolerance << ")" << std::endl;
        return 0;
    } else {
        std::cout << "\n[FAIL] Hidden states differ by more than tolerance (" << tolerance << ")" << std::endl;

        // Debug: print first few values
        std::cout << "\nFirst 10 hidden state values comparison (token 0):" << std::endl;
        auto out_f32 = mx::astype(hidden_states, mx::float32);
        auto exp_f32 = mx::astype(expected_hidden, mx::float32);
        mx::eval({out_f32, exp_f32});

        // Take first token's hidden state
        auto out_first = mx::reshape(mx::slice(out_f32, {0, 0, 0}, {1, 1, TEXT_DIM}), {TEXT_DIM});
        auto exp_first = mx::reshape(mx::slice(exp_f32, {0, 0, 0}, {1, 1, TEXT_DIM}), {TEXT_DIM});
        mx::eval({out_first, exp_first});

        out_first = mx::copy(out_first, mx::Device(mx::Device::cpu, 0));
        exp_first = mx::copy(exp_first, mx::Device(mx::Device::cpu, 0));
        mx::eval({out_first, exp_first});

        const float* out_ptr = out_first.data<float>();
        const float* exp_ptr = exp_first.data<float>();

        std::cout << "  Index\tC++\t\tPython\t\tDiff" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << "  " << i << "\t" << out_ptr[i] << "\t"
                      << exp_ptr[i] << "\t" << std::abs(out_ptr[i] - exp_ptr[i]) << std::endl;
        }

        return 1;
    }
}

// Step-by-step verification for debugging
int run_vision_verify_stepwise(const std::string& weights_dir) {
    std::cout << "=== Step-by-Step Vision Verification ===" << std::endl;

    std::string raw_path = weights_dir + "/vision_encoder.bin";
    std::string patches_path = weights_dir + "/test_patches.bin";

    // Load model
    VisionWeights model;
    if (!load_vision_weights(&model, raw_path)) {
        return 1;
    }

    // Load patches
    int num_patches = 576;
    auto patches = load_binary_f32(patches_path, {num_patches, PATCH_INPUT_DIM});
    if (patches.size() == 0) return 1;

    std::vector<GridTHW> grid_thw = {{1, 24, 24}};
    auto pos = build_vision_position_data(grid_thw);

    // Step 1: Patch embedding
    std::cout << "\n--- Step 1: Patch Embedding ---" << std::endl;
    auto x = patch_embed_forward(patches, &model);
    mx::eval(x);
    print_array_stats("After patch_embed", x);

    // Load Python intermediate for comparison
    // (If available from numpy files)

    // Step 2: Post-conv layernorm
    std::cout << "\n--- Step 2: Post-Conv LayerNorm ---" << std::endl;
    x = rms_norm(x, &model.post_conv_layernorm, VISION_RMS_EPS);
    mx::eval(x);
    print_array_stats("After post_conv_layernorm", x);

    // Step 3: Position embeddings
    std::cout << "\n--- Step 3: Position Embeddings ---" << std::endl;
    x = vision_embeddings_forward(x, &model, pos);
    mx::eval(x);
    print_array_stats("After position_embeddings", x);

    // Step 4: First transformer block
    std::cout << "\n--- Step 4: First Transformer Block ---" << std::endl;
    x = vision_block_forward(x, &model.blocks[0], pos);
    mx::eval(x);
    print_array_stats("After block 0", x);

    // Continue through all blocks
    std::cout << "\n--- Steps 5-27: Remaining Transformer Blocks ---" << std::endl;
    for (int i = 1; i < NUM_LAYERS; ++i) {
        x = vision_block_forward(x, &model.blocks[i], pos);
    }
    mx::eval(x);
    print_array_stats("After all blocks", x);

    // Step 28: Post layernorm
    std::cout << "\n--- Step 28: Post LayerNorm ---" << std::endl;
    x = rms_norm(x, &model.post_layernorm, VISION_RMS_EPS);
    mx::eval(x);
    print_array_stats("After post_layernorm", x);

    // Step 29: Downsample
    std::cout << "\n--- Step 29: Downsample Conv2d ---" << std::endl;
    int merge_tokens = x.shape(0) / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE);
    x = mx::reshape(x, {merge_tokens, SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE, VISION_DIM});
    x = mx::conv2d(x, model.downsample_weight,
                   {SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE}, {0, 0}, {1, 1}, 1);
    x = mx::add(x, mx::reshape(model.downsample_bias, {1, 1, 1, MERGE_DIM}));
    x = mx::reshape(x, {merge_tokens, MERGE_DIM});
    mx::eval(x);
    print_array_stats("After downsample", x);

    // Step 30: Merger
    std::cout << "\n--- Step 30: Merger ---" << std::endl;
    x = merger_forward(x, &model.merger);
    mx::eval(x);
    print_array_stats("Final output", x);

    return 0;
}

int main(int argc, char* argv[]) {
    const char* force_cpu_env = std::getenv("GLM_FORCE_CPU");
    bool force_cpu = force_cpu_env && std::string(force_cpu_env) == "1";
    if (!force_cpu) {
        auto gpu = mx::Device(mx::Device::gpu, 0);
        if (mx::is_available(gpu)) {
            mx::set_default_device(gpu);
        } else {
            mx::set_default_device(mx::Device(mx::Device::cpu, 0));
            std::cout << "GPU not available; using CPU." << std::endl;
        }
    } else {
        mx::set_default_device(mx::Device(mx::Device::cpu, 0));
        std::cout << "GLM_FORCE_CPU=1; using CPU." << std::endl;
    }

    // Check for verification mode
    const char* verify_env = std::getenv("GLM_VERIFY");
    const char* verify_text_env = std::getenv("GLM_VERIFY_TEXT");
    const char* weights_dir_env = std::getenv("GLM_WEIGHTS_DIR");

    if (verify_env && std::string(verify_env) == "1") {
        std::string weights_dir = weights_dir_env ? weights_dir_env : "vision_weights";
        return run_vision_verify(weights_dir);
    }

    if (verify_text_env && std::string(verify_text_env) == "1") {
        std::string weights_dir = weights_dir_env ? weights_dir_env : "vision_weights";
        return run_text_verify(weights_dir);
    }

    const char* verify_step_env = std::getenv("GLM_VERIFY_STEP");
    if (verify_step_env && std::string(verify_step_env) == "1") {
        std::string weights_dir = weights_dir_env ? weights_dir_env : "vision_weights";
        return run_vision_verify_stepwise(weights_dir);
    }

    // Original benchmark modes
    bool run_text = (GLM_RUN_TEXT != 0);
    const char* run_text_env = std::getenv("GLM_RUN_TEXT");
    if (run_text_env && std::string(run_text_env) == "1") {
        run_text = true;
    }

    if (run_text) {
        return run_text_bench();
    }
    return run_vision_bench();
}
