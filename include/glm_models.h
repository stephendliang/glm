#pragma once

#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include "glm_types.h"
#include <vector>
#include <string>

namespace mx = mlx::core;

// Activation functions
mx::array silu(mx::array x);

// Basic operations
mx::array fast_linear(mx::array x, const LinearWeights* w);
mx::array rms_norm(mx::array x, const RMSNormWeights* w, float eps = 1e-5f);

// Weight loading
bool load_vision_weights(VisionWeights* model, const std::string& path);
bool load_text_weights(TextModelWeights* model, const std::string& path);

// Vision model forward
mx::array vision_forward(mx::array patches, const VisionWeights* model,
                         const std::vector<GridTHW>& grid_thw);

// Batched generation
std::vector<std::vector<int32_t>> generate_batched(
    mx::array inputs_embeds,
    const TextModelWeights* model,
    const mx::array& position_ids,
    const GenerationConfig& config);

// RoPE index computation
RopeIndexResult get_rope_index(const mx::array& input_ids,
                               const std::vector<GridTHW>& image_grid_thw,
                               const std::vector<GridTHW>& video_grid_thw,
                               const mx::array& attention_mask);
