#include "glm_models.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// Vision Weight Loading

bool load_vision_weights(VisionWeights* model, const std::string& path) {
    std::cout << "Loading vision weights from: " << path << std::endl;

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

    model->patch_embed_weight = read_array({VISION_DIM, 2, PATCH_SIZE, PATCH_SIZE, 3});
    model->patch_embed_bias = read_array({VISION_DIM});
    model->pos_embed_weight = read_array({NUM_PATCHES, VISION_DIM});
    model->post_conv_layernorm.weight = read_array({VISION_DIM});

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

    model->post_layernorm.weight = read_array({VISION_DIM});
    model->downsample_weight = read_array({MERGE_DIM, SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE, VISION_DIM});
    model->downsample_bias = read_array({MERGE_DIM});

    model->merger.proj.weight = read_array({MERGE_DIM, MERGE_DIM});
    model->merger.proj.bias = read_array({MERGE_DIM});
    model->merger.post_ln.weight = read_array({MERGE_DIM});
    model->merger.post_ln.bias = read_array({MERGE_DIM});
    model->merger.gate_proj.weight = read_array({MERGE_DIM, MERGER_MLP_DIM});
    model->merger.up_proj.weight = read_array({MERGE_DIM, MERGER_MLP_DIM});
    model->merger.down_proj.weight = read_array({MERGER_MLP_DIM, MERGE_DIM});

    size_t elements_read = ptr - buffer.data();
    if (elements_read != buffer.size()) {
        std::cerr << "Warning: Read " << elements_read << " elements, buffer has " << buffer.size() << std::endl;
    }

    std::cout << "Vision weights loaded successfully!" << std::endl;
    std::cout << "  File size: " << file_size / 1024 / 1024 << " MB (float16)" << std::endl;
    return true;
}

// Vision RoPE Functions

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
    std::vector<int> image_type_h_vec;
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

                            float h_scaled = (h_idx + 0.5f) * PATCH_GRID / (float)h - 0.5f;
                            float w_scaled = (w_idx + 0.5f) * PATCH_GRID / (float)w - 0.5f;
                            int h_nn = std::max(0, std::min(PATCH_GRID - 1, (int)std::round(h_scaled)));
                            int w_nn = std::max(0, std::min(PATCH_GRID - 1, (int)std::round(w_scaled)));
                            pos_embed_indices.push_back(h_nn * PATCH_GRID + w_nn);
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
    data.raw_h_coords = h_coords_arr;
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

// Vision Model Forward Functions

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
    auto x = mx::reshape(patches, {num_patches, TEMPORAL_PATCH, PATCH_SIZE, 3, PATCH_SIZE});
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

mx::array interpolate_position_embeddings(const mx::array& pos_embed_weight,
                                          const mx::array& h_coords,
                                          const mx::array& w_coords,
                                          const mx::array& target_h,
                                          const mx::array& target_w) {
    int orig_size = 24;
    int hidden_size = pos_embed_weight.shape(1);
    auto pos_2d = mx::reshape(pos_embed_weight, {orig_size, orig_size, hidden_size});

    auto h_f32 = mx::astype(h_coords, mx::float32);
    auto w_f32 = mx::astype(w_coords, mx::float32);
    auto th_f32 = mx::astype(target_h, mx::float32);
    auto tw_f32 = mx::astype(target_w, mx::float32);

    auto h_pos = mx::subtract(
        mx::multiply(mx::divide(mx::add(h_f32, mx::array(0.5f)), th_f32),
                     mx::array(static_cast<float>(orig_size))),
        mx::array(0.5f));
    auto w_pos = mx::subtract(
        mx::multiply(mx::divide(mx::add(w_f32, mx::array(0.5f)), tw_f32),
                     mx::array(static_cast<float>(orig_size))),
        mx::array(0.5f));

    auto h_clamped = mx::clip(h_pos, mx::array(0.0f), mx::array(static_cast<float>(orig_size - 1 - 1e-5)));
    auto w_clamped = mx::clip(w_pos, mx::array(0.0f), mx::array(static_cast<float>(orig_size - 1 - 1e-5)));

    auto h0 = mx::floor(h_clamped);
    auto w0 = mx::floor(w_clamped);
    auto h1 = mx::minimum(mx::add(h0, mx::array(1.0f)), mx::array(static_cast<float>(orig_size - 1)));
    auto w1 = mx::minimum(mx::add(w0, mx::array(1.0f)), mx::array(static_cast<float>(orig_size - 1)));

    auto h_frac = mx::subtract(h_clamped, h0);
    auto w_frac = mx::subtract(w_clamped, w0);

    auto h0_int = mx::astype(h0, mx::int32);
    auto w0_int = mx::astype(w0, mx::int32);
    auto h1_int = mx::astype(h1, mx::int32);
    auto w1_int = mx::astype(w1, mx::int32);
    mx::eval({h0_int, w0_int, h1_int, w1_int, h_frac, w_frac});

    auto idx00 = mx::add(mx::multiply(h0_int, mx::array(orig_size)), w0_int);
    auto idx01 = mx::add(mx::multiply(h0_int, mx::array(orig_size)), w1_int);
    auto idx10 = mx::add(mx::multiply(h1_int, mx::array(orig_size)), w0_int);
    auto idx11 = mx::add(mx::multiply(h1_int, mx::array(orig_size)), w1_int);

    auto v00 = mx::take(pos_embed_weight, idx00, 0);
    auto v01 = mx::take(pos_embed_weight, idx01, 0);
    auto v10 = mx::take(pos_embed_weight, idx10, 0);
    auto v11 = mx::take(pos_embed_weight, idx11, 0);

    auto h_w = mx::reshape(h_frac, {-1, 1});
    auto w_w = mx::reshape(w_frac, {-1, 1});
    auto one_minus_h = mx::subtract(mx::array(1.0f), h_w);
    auto one_minus_w = mx::subtract(mx::array(1.0f), w_w);

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
    bool needs_interpolation = false;
    for (size_t i = 0; i < pos.seqlens.size(); ++i) {
        int seq = pos.seqlens[i];
        if (seq != 576) {
            needs_interpolation = true;
            break;
        }
    }

    if (!needs_interpolation) {
        auto pos_embed = mx::take(model->pos_embed_weight, pos.pos_embed_indices, 0);
        return mx::add(x, pos_embed);
    }

    int total_seq = pos.image_type_h.shape(0);
    std::vector<float> target_h_vec(total_seq);
    std::vector<float> target_w_vec(total_seq);

    int offset = 0;
    for (size_t img_idx = 0; img_idx < pos.seqlens.size(); ++img_idx) {
        int seq_len = pos.seqlens[img_idx];
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
        pos.raw_h_coords,
        pos.raw_w_coords,
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

    int merge_tokens = x.shape(0) / (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE);
    x = mx::reshape(x, {merge_tokens, SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE, VISION_DIM});

    x = mx::conv2d(x, model->downsample_weight,
                   {SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE}, {0, 0}, {1, 1}, 1);

    x = mx::add(x, mx::reshape(model->downsample_bias, {1, 1, 1, MERGE_DIM}));
    x = mx::reshape(x, {merge_tokens, MERGE_DIM});

    return merger_forward(x, &model->merger);
}
