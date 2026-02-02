#include "glm_models.h"
#include <fstream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>

// Sliding window size for repetition penalty (limits memory usage)
constexpr int REP_PENALTY_WINDOW = 64;

// Text Weight Loading

bool load_text_weights(TextModelWeights* model, const std::string& path) {
    printf("Loading text weights from: %s\n", path.c_str());

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path.c_str());
        return false;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint16_t> buffer(file_size / sizeof(uint16_t));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        fprintf(stderr, "Error: Failed to read file: %s\n", path.c_str());
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

    model->embed_tokens = read_array({TEXT_VOCAB_SIZE, TEXT_DIM});

    for (int i = 0; i < TEXT_NUM_LAYERS; ++i) {
        auto& layer = model->layers[i];
        layer.input_layernorm.weight = read_array({TEXT_DIM});
        layer.post_self_attn_layernorm.weight = read_array({TEXT_DIM});
        layer.post_attention_layernorm.weight = read_array({TEXT_DIM});
        layer.post_mlp_layernorm.weight = read_array({TEXT_DIM});
        layer.self_attn.q_proj.weight = read_array({TEXT_DIM, TEXT_DIM});
        layer.self_attn.q_proj.bias = read_array({TEXT_DIM});
        layer.self_attn.k_proj.weight = read_array({TEXT_DIM, TEXT_KV_DIM});
        layer.self_attn.k_proj.bias = read_array({TEXT_KV_DIM});
        layer.self_attn.v_proj.weight = read_array({TEXT_DIM, TEXT_KV_DIM});
        layer.self_attn.v_proj.bias = read_array({TEXT_KV_DIM});
        layer.self_attn.o_proj.weight = read_array({TEXT_DIM, TEXT_DIM});
        layer.mlp.gate_up_proj.weight = read_array({TEXT_DIM, TEXT_GATE_UP_DIM});
        layer.mlp.down_proj.weight = read_array({TEXT_MLP_DIM, TEXT_DIM});
    }

    model->norm.weight = read_array({TEXT_DIM});
    model->lm_head.weight = read_array({TEXT_DIM, TEXT_VOCAB_SIZE});

    size_t elements_read = ptr - buffer.data();
    if (elements_read != buffer.size()) {
        fprintf(stderr, "Warning: Read %zu elements, buffer has %zu\n", elements_read, buffer.size());
    }

    puts("Text weights loaded successfully!");
    printf("  File size: %zu MB (float16)\n", file_size / 1024 / 1024);
    return true;
}

// Text RoPE Functions

mx::array rotate_half_llm(mx::array x) {
    int B = x.shape(0);
    int H = x.shape(1);
    int T = x.shape(2);
    int dim = x.shape(3);

    auto x1 = mx::slice(x, {0, 0, 0, 0}, {B, H, T, dim}, {1, 1, 1, 2});
    auto x2 = mx::slice(x, {0, 0, 0, 1}, {B, H, T, dim}, {1, 1, 1, 2});

    int half = dim / 2;
    auto neg_x2 = mx::negative(x2);

    neg_x2 = mx::reshape(neg_x2, {B, H, T, half, 1});
    x1 = mx::reshape(x1, {B, H, T, half, 1});
    auto stacked = mx::concatenate({neg_x2, x1}, 4);
    return mx::reshape(stacked, {B, H, T, dim});
}

std::pair<mx::array, mx::array> text_rotary_embeddings(const mx::array& hidden_states,
                                                       const mx::array& position_ids) {
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

    auto cos_chunks = mx::split(cos, split_indices, 3);
    auto sin_chunks = mx::split(sin, split_indices, 3);

    int B = q.shape(0);
    int T = q.shape(2);

    std::vector<mx::array> cos_parts, sin_parts;
    for (int i = 0; i < 6; ++i) {
        int sec = i % 3;
        int chunk_dim = chunk_sizes[i];
        auto cos_part = mx::slice(cos_chunks[i], {sec, 0, 0, 0}, {sec + 1, B, T, chunk_dim});
        auto sin_part = mx::slice(sin_chunks[i], {sec, 0, 0, 0}, {sec + 1, B, T, chunk_dim});
        cos_parts.push_back(mx::squeeze(cos_part, std::vector<int>{0}));
        sin_parts.push_back(mx::squeeze(sin_part, std::vector<int>{0}));
    }

    auto cos_cat = mx::concatenate(cos_parts, 2);
    auto sin_cat = mx::concatenate(sin_parts, 2);
    cos_cat = mx::reshape(cos_cat, {B, 1, T, cos_cat.shape(2)});
    sin_cat = mx::reshape(sin_cat, {B, 1, T, sin_cat.shape(2)});

    int half = cos_cat.shape(3) / 2;
    auto cos_half = mx::slice(cos_cat, {0, 0, 0, 0}, {B, 1, T, half});
    auto sin_half = mx::slice(sin_cat, {0, 0, 0, 0}, {B, 1, T, half});

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

// Text Model Forward Functions

mx::array text_attention_forward_cached(
    mx::array x,
    const TextAttentionWeights* w,
    const std::pair<mx::array, mx::array>& position_embeddings,
    LayerKVCache& layer_cache,
    int cache_pos
) {
    int B = x.shape(0);
    int T = x.shape(1);

    auto q = fast_linear(x, &w->q_proj);
    auto k = fast_linear(x, &w->k_proj);
    auto v = fast_linear(x, &w->v_proj);

    auto q_rs = mx::transpose(mx::reshape(q, {B, T, TEXT_NUM_HEADS, TEXT_HEAD_DIM}), {0, 2, 1, 3});
    auto k_rs = mx::transpose(mx::reshape(k, {B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM}), {0, 2, 1, 3});
    auto v_rs = mx::transpose(mx::reshape(v, {B, T, TEXT_KV_HEADS, TEXT_HEAD_DIM}), {0, 2, 1, 3});

    auto rotated = apply_multimodal_rotary_pos_emb(q_rs, k_rs, position_embeddings.first, position_embeddings.second);
    q_rs = rotated.first;
    auto k_rot = rotated.second;

    layer_cache.k_cache = mx::slice_update(layer_cache.k_cache, k_rot,
        {0, 0, cache_pos, 0}, {B, TEXT_KV_HEADS, cache_pos + T, TEXT_HEAD_DIM});
    layer_cache.v_cache = mx::slice_update(layer_cache.v_cache, v_rs,
        {0, 0, cache_pos, 0}, {B, TEXT_KV_HEADS, cache_pos + T, TEXT_HEAD_DIM});
    layer_cache.current_len = cache_pos + T;

    int total_len = layer_cache.current_len;
    auto k_valid = mx::slice(layer_cache.k_cache, {0, 0, 0, 0},
                             {B, TEXT_KV_HEADS, total_len, TEXT_HEAD_DIM});
    auto v_valid = mx::slice(layer_cache.v_cache, {0, 0, 0, 0},
                             {B, TEXT_KV_HEADS, total_len, TEXT_HEAD_DIM});

    float scale = 1.0f / std::sqrt((float)TEXT_HEAD_DIM);
    auto out = (T == 1)
        ? mx::fast::scaled_dot_product_attention(q_rs, k_valid, v_valid, scale)
        : mx::fast::scaled_dot_product_attention(q_rs, k_valid, v_valid, scale, "causal");

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

mx::array text_layer_forward_cached(
    mx::array x,
    const TextLayerWeights* w,
    const std::pair<mx::array, mx::array>& position_embeddings,
    LayerKVCache& layer_cache,
    int cache_pos
) {
    auto residual = x;
    x = rms_norm(x, &w->input_layernorm, TEXT_RMS_EPS);
    auto attn_out = text_attention_forward_cached(x, &w->self_attn, position_embeddings, layer_cache, cache_pos);
    x = rms_norm(attn_out, &w->post_self_attn_layernorm, TEXT_RMS_EPS);
    x = mx::add(residual, x);

    residual = x;
    x = rms_norm(x, &w->post_attention_layernorm, TEXT_RMS_EPS);
    x = text_mlp_forward(x, &w->mlp);
    x = rms_norm(x, &w->post_mlp_layernorm, TEXT_RMS_EPS);

    return mx::add(residual, x);
}

mx::array compute_logits(mx::array hidden_states, const TextModelWeights* model) {
    return fast_linear(hidden_states, &model->lm_head);
}

// Sampling Functions

mx::array apply_repetition_penalty_batched(
    mx::array logits,
    const std::vector<std::vector<int32_t>>& generated_tokens,
    float penalty
) {
    int B = logits.shape(0);
    int vocab_size = logits.shape(1);

    std::vector<int32_t> linear_indices;
    linear_indices.reserve(B * REP_PENALTY_WINDOW);

    for (int b = 0; b < B && b < static_cast<int>(generated_tokens.size()); ++b) {
        const auto& hist = generated_tokens[b];
        if (hist.empty()) continue;

        int start = std::max(0, static_cast<int>(hist.size()) - REP_PENALTY_WINDOW);
        for (int i = start; i < static_cast<int>(hist.size()); ++i) {
            int32_t tok = hist[i];
            if (tok >= 0 && tok < vocab_size) {
                linear_indices.push_back(b * vocab_size + tok);
            }
        }
    }

    if (linear_indices.empty()) {
        return logits;
    }

    int N = static_cast<int>(linear_indices.size());

    auto idx = mx::array(linear_indices.data(), {N}, mx::int32);
    auto logits_flat = mx::reshape(logits, {B * vocab_size});
    auto vals = mx::take(logits_flat, idx, 0);

    auto is_positive = mx::greater(vals, mx::array(0.0f, vals.dtype()));
    auto divisor = mx::where(is_positive,
                             mx::array(penalty, vals.dtype()),
                             mx::array(1.0f / penalty, vals.dtype()));
    auto penalized = mx::divide(vals, divisor);

    auto delta = mx::subtract(penalized, vals);

    auto logits_2d = mx::reshape(logits_flat, {B * vocab_size, 1});
    auto delta_3d = mx::reshape(delta, {N, 1, 1});

    auto result_2d = mx::scatter_add(logits_2d, idx, delta_3d, 0);
    auto result_flat = mx::squeeze(result_2d, 1);

    return mx::reshape(result_flat, {B, vocab_size});
}

mx::array apply_top_p_batched(mx::array scores, float top_p) {
    int B = scores.shape(0);
    int vocab_size = scores.shape(1);

    auto neg_scores = mx::negative(scores);
    auto sorted_indices = mx::argsort(neg_scores, -1);

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

    auto sorted_probs = mx::softmax(sorted_scores, -1);
    auto cumsum = mx::cumsum(sorted_probs, -1);

    auto zeros_col = mx::zeros({B, 1}, cumsum.dtype());
    auto cumsum_shifted = mx::slice(cumsum, {0, 0}, {B, vocab_size - 1});
    cumsum_shifted = mx::concatenate({zeros_col, cumsum_shifted}, -1);

    auto sorted_mask = mx::greater(cumsum_shifted, mx::array(top_p));

    auto neg_inf = mx::array(-std::numeric_limits<float>::infinity());
    auto masked_sorted = mx::where(sorted_mask, neg_inf, sorted_scores);

    auto unsort_indices = mx::argsort(sorted_indices, -1);
    auto linear_unsort = mx::add(
        mx::multiply(batch_indices, mx::array(vocab_size, mx::int32)),
        unsort_indices
    );
    linear_unsort = mx::reshape(linear_unsort, {B * vocab_size});

    auto masked_flat = mx::reshape(masked_sorted, {B * vocab_size});
    auto result_flat = mx::take(masked_flat, linear_unsort, 0);

    return mx::reshape(result_flat, {B, vocab_size});
}

mx::array sample_tokens_batched(
    mx::array logits,
    const GenerationConfig& config,
    const std::vector<std::vector<int32_t>>& generated_tokens,
    const std::vector<bool>& finished
) {
    int B = logits.shape(0);

    if (logits.ndim() == 3) {
        int T = logits.shape(1);
        logits = mx::slice(logits, {0, T-1, 0}, {B, T, logits.shape(2)});
        logits = mx::squeeze(logits, 1);
    }

    int vocab_size = logits.shape(1);
    logits = mx::astype(logits, mx::float32);

    if (config.repetition_penalty != 1.0f && !generated_tokens.empty()) {
        logits = apply_repetition_penalty_batched(logits, generated_tokens, config.repetition_penalty);
    }

    if (!config.do_sample) {
        auto tokens = mx::argmax(logits, -1);
        tokens = mx::astype(tokens, mx::int32);

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

    auto scores = mx::divide(logits, mx::array(config.temperature));

    if (config.top_k > 0 && config.top_k < vocab_size) {
        auto topk_vals = mx::topk(scores, config.top_k, -1);
        auto thresholds = mx::min(topk_vals, -1, true);
        auto mask = mx::less(scores, thresholds);
        auto neg_inf = mx::array(-std::numeric_limits<float>::infinity());
        scores = mx::where(mask, neg_inf, scores);
    }

    if (config.top_p < 1.0f) {
        scores = apply_top_p_batched(scores, config.top_p);
    }

    auto max_val = mx::max(scores, -1, true);
    auto shifted = mx::subtract(scores, max_val);
    auto log_sum_exp = mx::add(mx::log(mx::sum(mx::exp(shifted), -1, true)), max_val);
    auto log_probs = mx::subtract(scores, log_sum_exp);

    auto sampled = mx::random::categorical(log_probs, 1);
    sampled = mx::reshape(sampled, {B});
    sampled = mx::astype(sampled, mx::int32);

    if (!finished.empty()) {
        std::vector<int32_t> eos_data(B, config.eos_token_ids[0]);
        auto eos_tokens = mx::array(eos_data.data(), {B}, mx::int32);
        std::vector<uint8_t> mask_data(B);
        for (int b = 0; b < B; ++b) mask_data[b] = finished[b] ? 1 : 0;
        auto mask = mx::array(mask_data.data(), {B}, mx::bool_);
        sampled = mx::where(mask, eos_tokens, sampled);
    }

    mx::eval(sampled);
    return sampled;
}

// Batched Generation Functions

mx::array make_default_position_ids(int B, int T) {
    auto pos = mx::astype(mx::arange(T), mx::int32);
    pos = mx::reshape(pos, {1, 1, T});
    return mx::broadcast_to(pos, {3, B, T});
}

mx::array text_model_forward_embeds_batched(
    mx::array inputs_embeds,
    const TextModelWeights* model,
    const mx::array& position_ids,
    BatchedTextKVCache& cache
) {
    auto pos_ids = position_ids.size() > 0 ? position_ids
                   : make_default_position_ids(inputs_embeds.shape(0), inputs_embeds.shape(1));
    auto position_embeddings = text_rotary_embeddings(inputs_embeds, pos_ids);

    int T = inputs_embeds.shape(1);
    int cache_pos = cache.current_len;

    auto x = inputs_embeds;
    for (int i = 0; i < TEXT_NUM_LAYERS; ++i) {
        x = text_layer_forward_cached(
            x, &model->layers[i], position_embeddings,
            cache.layers[i], cache_pos);
    }

    cache.current_len = cache_pos + T;

    return rms_norm(x, &model->norm, TEXT_RMS_EPS);
}

bool is_eos_token(int32_t token, const GenerationConfig& config) {
    for (int eos_id : config.eos_token_ids) {
        if (token == eos_id) return true;
    }
    return false;
}

std::vector<std::vector<int32_t>> generate_batched(
    mx::array inputs_embeds,
    const TextModelWeights* model,
    const mx::array& position_ids,
    const GenerationConfig& config
) {
    int B = inputs_embeds.shape(0);
    int prompt_len = inputs_embeds.shape(1);

    std::vector<std::vector<int32_t>> generated(B);
    BatchedTextKVCache cache(B);

    int max_seq = prompt_len + config.max_new_tokens;
    cache.preallocate(B, max_seq);

    auto hidden = text_model_forward_embeds_batched(inputs_embeds, model, position_ids, cache);

    mx::eval(position_ids);
    auto pos_max_t = mx::max(position_ids, {0, 2});
    mx::eval(pos_max_t);

    std::vector<int> next_positions(B);
    const int32_t* pos_max_ptr = pos_max_t.data<int32_t>();
    for (int b = 0; b < B; ++b) {
        next_positions[b] = pos_max_ptr[b] + 1;
        cache.cache_positions[b] = next_positions[b];
    }

    auto last_hidden = mx::slice(hidden, {0, prompt_len - 1, 0}, {B, prompt_len, TEXT_DIM});
    last_hidden = mx::squeeze(last_hidden, 1);

    auto logits = compute_logits(mx::expand_dims(last_hidden, 1), model);
    logits = mx::squeeze(logits, 1);
    mx::eval(logits);

    auto next_tokens = sample_tokens_batched(logits, config, generated, cache.finished);
    mx::eval(next_tokens);

    const int32_t* first_tokens_ptr = next_tokens.data<int32_t>();
    for (int b = 0; b < B; ++b) {
        int32_t tok = first_tokens_ptr[b];
        generated[b].push_back(tok);
        if (is_eos_token(tok, config)) {
            cache.finished[b] = true;
        }
    }

    std::vector<int32_t> decode_pos_data(3 * B);

    for (int step = 0; step < config.max_new_tokens - 1; ++step) {
        if (cache.all_finished()) break;

        auto token_arr = mx::reshape(next_tokens, {B, 1});
        auto new_embeds = mx::take(model->embed_tokens, token_arr, 0);

        for (int b = 0; b < B; ++b) {
            decode_pos_data[b] = next_positions[b];
            decode_pos_data[B + b] = next_positions[b];
            decode_pos_data[2 * B + b] = next_positions[b];
        }
        auto decode_pos = mx::array(decode_pos_data.data(), {3, B, 1}, mx::int32);

        hidden = text_model_forward_embeds_batched(new_embeds, model, decode_pos, cache);

        for (int b = 0; b < B; ++b) {
            if (!cache.finished[b]) {
                next_positions[b]++;
                cache.cache_positions[b] = next_positions[b];
            }
        }

        logits = compute_logits(hidden, model);
        logits = mx::squeeze(logits, 1);
        mx::eval(logits);

        next_tokens = sample_tokens_batched(logits, config, generated, cache.finished);
        mx::eval(next_tokens);

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

// RoPE Index Computation

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
