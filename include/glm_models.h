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
mx::array gelu_erf(mx::array x);

// Basic operations
mx::array fast_linear(mx::array x, const LinearWeights* w);
mx::array rms_norm(mx::array x, const RMSNormWeights* w, float eps = 1e-5f);
mx::array layer_norm(mx::array x, const LayerNormWeights* w, float eps = 1e-5f);

// Weight loading
bool load_vision_weights(VisionWeights* model, const std::string& path);
bool load_text_weights(TextModelWeights* model, const std::string& path);

// Vision RoPE
mx::array rotate_half_vision(mx::array x);
std::pair<mx::array, mx::array> apply_rotary_pos_emb_vision(mx::array q, mx::array k,
                                                            const mx::array& cos,
                                                            const mx::array& sin);
VisionPositionData build_vision_position_data(const std::vector<GridTHW>& grid_thw);

// Text RoPE
mx::array rotate_half_llm(mx::array x);
std::pair<mx::array, mx::array> text_rotary_embeddings(const mx::array& hidden_states,
                                                       const mx::array& position_ids);
std::pair<mx::array, mx::array> apply_multimodal_rotary_pos_emb(
    mx::array q, mx::array k, const mx::array& cos, const mx::array& sin);

// Vision model forward
mx::array vision_mlp_forward(mx::array x, const MLPWeights* w);
mx::array vision_attention_forward(mx::array x, const VisionBlockWeights* w,
                                   const VisionPositionData& pos);
mx::array vision_block_forward(mx::array x, const VisionBlockWeights* w,
                               const VisionPositionData& pos);
mx::array merger_forward(mx::array x, const MergerWeights* w);
mx::array patch_embed_forward(mx::array patches, const VisionWeights* model);
mx::array interpolate_position_embeddings(const mx::array& pos_embed_weight,
                                          const mx::array& h_coords,
                                          const mx::array& w_coords,
                                          const mx::array& target_h,
                                          const mx::array& target_w);
mx::array vision_embeddings_forward(mx::array x, const VisionWeights* model,
                                    const VisionPositionData& pos);
mx::array vision_forward(mx::array patches, const VisionWeights* model,
                         const std::vector<GridTHW>& grid_thw);

// Text model forward
mx::array text_attention_forward_cached(
    mx::array x,
    const TextAttentionWeights* w,
    const std::pair<mx::array, mx::array>& position_embeddings,
    LayerKVCache& layer_cache,
    int cache_pos);
mx::array text_mlp_forward(mx::array x, const TextMLPWeights* w);
mx::array text_layer_forward_cached(
    mx::array x,
    const TextLayerWeights* w,
    const std::pair<mx::array, mx::array>& position_embeddings,
    LayerKVCache& layer_cache,
    int cache_pos);
mx::array compute_logits(mx::array hidden_states, const TextModelWeights* model);

// Sampling
mx::array apply_repetition_penalty_batched(
    mx::array logits,
    const std::vector<std::vector<int32_t>>& generated_tokens,
    float penalty);
mx::array apply_top_p_batched(mx::array scores, float top_p);
mx::array sample_tokens_batched(
    mx::array logits,
    const GenerationConfig& config,
    const std::vector<std::vector<int32_t>>& generated_tokens,
    const std::vector<bool>& finished);

// Batched generation
mx::array make_default_position_ids(int B, int T);
mx::array text_model_forward_embeds_batched(
    mx::array inputs_embeds,
    const TextModelWeights* model,
    const mx::array& position_ids,
    BatchedTextKVCache& cache);
bool is_eos_token(int32_t token, const GenerationConfig& config);
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
