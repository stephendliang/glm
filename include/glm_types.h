#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <cstdio>

namespace mx = mlx::core;

// Architecture Constants (GLM-4.6V Vision)
#define VISION_DIM 1536
#define NUM_HEADS 12
#define HEAD_DIM (VISION_DIM / NUM_HEADS)
#define NUM_LAYERS 24
#define PATCH_SIZE 14
#define IMAGE_SIZE 336
#define PATCH_GRID (IMAGE_SIZE / PATCH_SIZE)
#define NUM_PATCHES (PATCH_GRID * PATCH_GRID) // 576
#define TEMPORAL_PATCH 2
#define BLOCK_MLP_DIM 4096
#define MERGE_DIM 4096
#define MERGER_MLP_DIM 13696
#define SPATIAL_MERGE_SIZE 2
#define VISION_RMS_EPS 1e-5f
#define VISION_ROPE_THETA 10000.0f

// Architecture Constants (GLM-4.6V Language Model)
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
#define IMAGE_START_TOKEN_ID 151339
#define IMAGE_END_TOKEN_ID 151340
#define VIDEO_START_TOKEN_ID 151341
#define VIDEO_END_TOKEN_ID 151342

// Helper Function
inline mx::array empty_array() { return mx::array({}); }

// Configuration Structs

/*
    Runtime image configuration - all dimensions derived from GLM_IMAGE_SIZE env var.
    grid = image_size / 14 (patch size is always 14)
    tokens_per_image = (grid / 2)^2 (2x2 spatial merge)
    patch_dim = 3 * 14 * 14 = 588
*/
struct ImageConfig {
    int image_size;
    int grid;
    int tokens_per_image;
    int patch_dim;

    // Single-image offsets
    int image_start;
    int image_end;

    // Dual-image offsets
    int image_a_start;
    int image_a_end;
    int image_b_start;
    int image_b_end;

    static ImageConfig from_size(int size) {
        ImageConfig c;
        c.image_size = size;
        c.grid = size / 14;  // PATCH_SIZE = 14
        c.tokens_per_image = (c.grid / 2) * (c.grid / 2);  // 2x2 spatial merge
        c.patch_dim = 3 * 14 * 14;  // 588

        c.image_start = 5;  // Fixed prompt structure
        c.image_end = 5 + c.tokens_per_image;

        c.image_a_start = 5;
        c.image_a_end = 5 + c.tokens_per_image;
        c.image_b_start = c.image_a_end + 2;  // 2 separator tokens
        c.image_b_end = c.image_b_start + c.tokens_per_image;

        return c;
    }

    void print() const {
        printf("ImageConfig: size=%d grid=%d tokens=%d patch_dim=%d\n",
               image_size, grid, tokens_per_image, patch_dim);
        printf("  Single: start=%d end=%d\n", image_start, image_end);
        printf("  Dual: A=[%d,%d) B=[%d,%d)\n",
               image_a_start, image_a_end, image_b_start, image_b_end);
    }
};

// Generation configuration (defaults match HuggingFace)
struct GenerationConfig {
    int max_new_tokens = 100;
    std::vector<int> eos_token_ids = {151329, 151336, 151338, 151348};  // HF has 4 EOS tokens
    float temperature = 0.8f;     // Match Python default
    bool do_sample = true;        // Match Python default (NOT greedy!)
    int top_k = 50;               // Relaxed from 2 to reduce repetition
    float top_p = 0.6f;           // Match Python default
    float repetition_penalty = 1.1f;  // 1.1 = optimal (breaks loops without hurting quality)
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
    mx::array image_type_h = empty_array();  // Scaled H coordinates (nearest-neighbor indices)
    mx::array image_type_w = empty_array();  // Scaled W coordinates (nearest-neighbor indices)
    mx::array raw_h_coords = empty_array();  // Raw H coordinates (0 to grid_h-1)
    mx::array raw_w_coords = empty_array();  // Raw W coordinates (0 to grid_w-1)
    std::vector<int> seqlens;
    int total_seq = 0;
};

// Base Weight Structs
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

// Vision Model Weights
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

// Text Model Weights
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

// KV Cache Structs

// Per-layer KV cache with pre-allocation support
struct LayerKVCache {
    mx::array k_cache = empty_array();  // Pre-allocated: [B, kv_heads=2, max_seq, head_dim=128]
    mx::array v_cache = empty_array();  // Pre-allocated: [B, kv_heads=2, max_seq, head_dim=128]
    int current_len = 0;  // Valid entries in cache
    int max_len = 0;      // Allocated capacity

    bool is_empty() const { return current_len == 0; }
};

// Batched KV cache with per-batch position tracking for parallel generation
struct BatchedTextKVCache {
    std::vector<LayerKVCache> layers;
    std::vector<int> cache_positions;  // [B] positions, one per batch element
    std::vector<bool> finished;        // [B] EOS flags for each sequence
    int max_seq_len = 0;               // Allocated capacity
    int current_len = 0;               // Global position tracker (sequence length in cache)

    BatchedTextKVCache(int batch_size)
        : layers(TEXT_NUM_LAYERS),
          cache_positions(batch_size, 0),
          finished(batch_size, false) {}

    // Pre-allocate cache to max_len for O(1) slice_update writes
    void preallocate(int B, int max_len) {
        max_seq_len = max_len;
        for (auto& l : layers) {
            l.k_cache = mx::zeros({B, TEXT_KV_HEADS, max_len, TEXT_HEAD_DIM}, mx::float16);
            l.v_cache = mx::zeros({B, TEXT_KV_HEADS, max_len, TEXT_HEAD_DIM}, mx::float16);
            l.max_len = max_len;
            l.current_len = 0;
        }
        current_len = 0;
    }

    void clear() {
        for (auto& l : layers) {
            l.k_cache = empty_array();
            l.v_cache = empty_array();
            l.current_len = 0;
            l.max_len = 0;
        }
        std::fill(cache_positions.begin(), cache_positions.end(), 0);
        std::fill(finished.begin(), finished.end(), false);
        max_seq_len = 0;
        current_len = 0;
    }

    int total_length() const {
        return current_len;
    }

    int batch_size() const {
        return static_cast<int>(cache_positions.size());
    }

    bool all_finished() const {
        for (bool f : finished) {
            if (!f) return false;
        }
        return true;
    }

    int num_finished() const {
        int count = 0;
        for (bool f : finished) {
            if (f) count++;
        }
        return count;
    }
};

// Helper Structs
struct RopeIndexResult {
    mx::array position_ids = empty_array();
    mx::array rope_deltas = empty_array();
};

struct PrecomputeBuffer {
    std::vector<mx::array> patches;     // Patches for each image
    std::vector<std::string> paths;     // File paths
};
