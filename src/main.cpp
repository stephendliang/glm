#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <filesystem>
#include <functional>
#include <cstring>
#include "glm_types.h"
#include "glm_image.h"
#include "glm_models.h"

namespace fs = std::filesystem;

// Global image config - initialized in main()
static ImageConfig g_img_cfg;

// Shared Utilities for Generation Modes

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
        printf("Random seed: %s\n", seed_env);
    }

    return config;
}

// Print generation config summary
void print_generation_config(const GenerationConfig& config) {
    printf("\nSampling config: do_sample=%s temp=%.2f top_k=%d top_p=%.2f rep_penalty=%.2f\n",
           config.do_sample ? "true" : "false", config.temperature,
           config.top_k, config.top_p, config.repetition_penalty);
}

// Load filenames from text file
std::vector<std::string> load_filenames(const std::string& path) {
    std::vector<std::string> filenames;
    std::ifstream file(path);
    if (!file) {
        return filenames;
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
        return tokens;
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
            printf("  [%zu] EOS\n", j);
        } else if (include_think_tags && tok == 151346) {
            printf("  [%zu] <think>\n", j);
        } else if (include_think_tags && tok == 151347) {
            printf("  [%zu] </think>\n", j);
        } else if (tok >= 65 && tok <= 90) {
            printf("  [%zu] '%c'\n", j, static_cast<char>(tok));
        } else if (tok >= 97 && tok <= 122) {
            printf("  [%zu] '%c'\n", j, static_cast<char>(tok));
        }
    }
}

// Print generation timing summary
void print_generation_timing(double ms, int total_tokens) {
    double tokens_per_sec = total_tokens / (ms / 1000.0);
    printf("\n=== Results ===\nGenerated %d total tokens in %.1f ms (%.1f tok/s)\n",
           total_tokens, ms, tokens_per_sec);
}

// Precompute Embeddings: JPEG -> Vision Encoder -> Save to Disk

int run_precompute_embeddings(const std::string& input_dir,
                               const std::string& output_dir,
                               const std::string& weights_dir,
                               int batch_size) {
    puts("=== Precompute Vision Embeddings ===");
    printf("Input:  %s\n", input_dir.c_str());
    printf("Output: %s\n", output_dir.c_str());
    printf("Config: %dx%d -> %d tokens per image\n",
           g_img_cfg.image_size, g_img_cfg.image_size, g_img_cfg.tokens_per_image);
    printf("Batch:  %d\n", batch_size);

    fs::create_directories(output_dir);

    std::vector<std::string> image_files;
    try {
        image_files = list_images_recursive(input_dir);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error listing images: %s\n", e.what());
        return 1;
    }

    if (image_files.empty()) {
        fprintf(stderr, "No images found in %s\n", input_dir.c_str());
        return 1;
    }
    printf("Found %zu images\n\n", image_files.size());

    puts("Loading vision encoder...");
    VisionWeights vision;
    if (!load_vision_weights(&vision, weights_dir + "/vision_encoder.bin")) {
        fputs("Failed to load vision weights\n", stderr);
        return 1;
    }

    std::vector<GridTHW> fixed_grid_thw;
    for (int i = 0; i < batch_size; ++i) {
        fixed_grid_thw.push_back({1, g_img_cfg.grid, g_img_cfg.grid});
    }

    auto compiled_vision = mx::compile([&vision, &fixed_grid_thw](const std::vector<mx::array>& inputs) {
        return std::vector<mx::array>{vision_forward(inputs[0], &vision, fixed_grid_thw)};
    });

    PrecomputeBuffer buffers[2];
    std::mutex mu;
    std::condition_variable cv;
    bool ready[2] = {false, false};
    bool loader_done = false;
    std::string error;

    std::string embeddings_path = output_dir + "/embeddings.bin";
    std::string filenames_path = output_dir + "/filenames.txt";
    std::ofstream emb_out(embeddings_path, std::ios::binary);
    std::ofstream names_out(filenames_path);

    if (!emb_out || !names_out) {
        fputs("Failed to open output files\n", stderr);
        return 1;
    }

    size_t total_images = image_files.size();
    std::atomic<size_t> images_processed{0};
    std::atomic<size_t> file_idx{0};

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
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&]() { return !ready[buf_idx] || !error.empty(); });
                if (!error.empty()) break;
            }

            auto& buf = buffers[buf_idx];
            buf.patches.clear();
            buf.paths.clear();

            for (int b = 0; b < batch_size; ++b) {
                size_t idx = file_idx.fetch_add(1);
                if (idx >= total_images) break;

                const auto& path = image_files[idx];
                std::vector<uint8_t> pixels;
                int w, h;

                if (!decode_jpeg(tj, path, &w, &h, &pixels)) {
                    std::lock_guard<std::mutex> lock(mu);
                    error = "Failed to decode: " + path;
                    break;
                }

                std::vector<uint8_t> resized;
                resize_and_pad(pixels, w, h, &resized, g_img_cfg.image_size);

                auto patches = normalize_and_patchify(resized, g_img_cfg.image_size, 14);
                buf.patches.push_back(patches);
                buf.paths.push_back(path);
            }

            if (!error.empty()) break;

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

    auto start_time = std::chrono::steady_clock::now();
    int buf_idx = 0;

    std::optional<mx::array> prev_embeddings;
    std::vector<std::string> prev_paths;

    while (true) {
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

        auto batched_patches = batch_patches(buf.patches);
        int actual_B = static_cast<int>(buf.patches.size());

        mx::array padded_patches = batched_patches;
        if (actual_B < batch_size) {
            int patches_per_image = batched_patches.shape(0) / actual_B;
            int total_patches_needed = batch_size * patches_per_image;
            int padding_patches = total_patches_needed - batched_patches.shape(0);
            auto padding = mx::zeros({padding_patches, batched_patches.shape(1)}, batched_patches.dtype());
            padded_patches = mx::concatenate({batched_patches, padding}, 0);
        }

        auto result = compiled_vision({padded_patches});
        auto embeddings = result[0];

        if (actual_B < batch_size) {
            int actual_tokens = actual_B * g_img_cfg.tokens_per_image;
            embeddings = mx::slice(embeddings, {0, 0}, {actual_tokens, TEXT_DIM});
        }
        int B = actual_B;

        embeddings = mx::reshape(embeddings, {B, g_img_cfg.tokens_per_image, TEXT_DIM});

        auto emb_fp16 = mx::astype(embeddings, mx::float16);
        mx::async_eval({emb_fp16});

        if (prev_embeddings) {
            const uint16_t* data = prev_embeddings->data<uint16_t>();
            size_t bytes = static_cast<size_t>(prev_embeddings->shape(0)) * prev_embeddings->shape(1)
                         * prev_embeddings->shape(2) * sizeof(uint16_t);
            emb_out.write(reinterpret_cast<const char*>(data), bytes);

            for (const auto& path : prev_paths) {
                names_out << path << "\n";
            }
        }

        prev_embeddings = emb_fp16;
        prev_paths = buf.paths;

        images_processed += B;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double rate = images_processed.load() / elapsed;
        printf("\r  Processed %d/%zu images (%.1f img/s)", images_processed.load(), total_images, rate);
        fflush(stdout);

        {
            std::lock_guard<std::mutex> lock(mu);
            ready[buf_idx] = false;
        }
        cv.notify_all();
        buf_idx = 1 - buf_idx;
    }

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
        fprintf(stderr, "\nError: %s\n", error.c_str());
        return 1;
    }

    emb_out.close();
    names_out.close();

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    printf("\n\n=== Complete ===\n");
    printf("Processed: %d images\n", images_processed.load());
    printf("Time:      %.2f seconds\n", total_time);
    printf("Rate:      %.1f images/sec\n", images_processed.load() / total_time);
    printf("Output:\n  %s (%zu MB)\n  %s\n",
           embeddings_path.c_str(), fs::file_size(embeddings_path) / 1024 / 1024,
           filenames_path.c_str());

    return 0;
}

// Generate Mode: Dual-Image Comparison

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

    int chunk_size = batch_size;
    if (chunk_env) {
        chunk_size = std::atoi(chunk_env);
        if (chunk_size <= 0) chunk_size = batch_size;
    }

    puts("=== Dual-Image Comparison Generation (448x448) ===");
    if (chunk_size < batch_size) {
        printf("Chunk size: %d (processing in chunks)\n", chunk_size);
    }

    std::string filenames_path = embeds_dir + "/filenames.txt";
    auto filenames = load_filenames(filenames_path);
    if (filenames.empty()) {
        fprintf(stderr, "Failed to open %s\n", filenames_path.c_str());
        return 1;
    }
    int num_images = static_cast<int>(filenames.size());
    printf("Loaded %d image filenames\n", num_images);

    if (num_images < 2) {
        fputs("Need at least 2 images for comparison\n", stderr);
        return 1;
    }

    std::vector<std::pair<int, int>> pairs;

    if (pairs_file_env) {
        int max_pairs = batch_size_env ? std::atoi(batch_size_env) : 64;
        std::ifstream pairs_file(pairs_file_env);
        if (!pairs_file) {
            fprintf(stderr, "Failed to open pairs file: %s\n", pairs_file_env);
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
        printf("Loaded %zu pairs from %s\n", pairs.size(), pairs_file_env);
    } else if (idx_a_env && idx_b_env) {
        int idx_a = std::atoi(idx_a_env);
        int idx_b = std::atoi(idx_b_env);
        if (idx_a >= 0 && idx_a < num_images && idx_b >= 0 && idx_b < num_images) {
            pairs.push_back({idx_a, idx_b});
            printf("Using explicit pair: A=%d, B=%d\n", idx_a, idx_b);
        } else {
            fprintf(stderr, "Invalid indices: idx_a=%d, idx_b=%d (num_images=%d)\n",
                    idx_a, idx_b, num_images);
            return 1;
        }
    } else {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < batch_size; ++i) {
            int idx_a = std::rand() % num_images;
            int idx_b;
            do { idx_b = std::rand() % num_images; } while (idx_b == idx_a);
            pairs.push_back({idx_a, idx_b});
        }
        printf("Generated %zu random pairs\n", pairs.size());
    }

    int B = static_cast<int>(pairs.size());
    if (B == 0) {
        fputs("No valid pairs to process\n", stderr);
        return 1;
    }

    printf("\nComparing %d pair%s:\n", B, B > 1 ? "s" : "");
    for (int i = 0; i < B; ++i) {
        printf("  [%d] A=%d (%s) vs B=%d (%s)\n", i,
               pairs[i].first, filenames[pairs[i].first].c_str(),
               pairs[i].second, filenames[pairs[i].second].c_str());
    }

    std::string embeds_path = embeds_dir + "/embeddings.bin";
    std::ifstream embeds_file(embeds_path, std::ios::binary);
    if (!embeds_file) {
        fprintf(stderr, "Failed to open %s\n", embeds_path.c_str());
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

    std::string tokens_path = weights_dir + "/dueling_prompt_tokens_448.bin";
    auto prompt_tokens = load_prompt_tokens(tokens_path);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "Failed to open %s\n", tokens_path.c_str());
        return 1;
    }
    int num_tokens = static_cast<int>(prompt_tokens.size());
    printf("Loaded %d prompt tokens\n", num_tokens);

    puts("Loading text model...");
    TextModelWeights text;
    if (!load_text_weights(&text, weights_dir + "/text_model.bin")) {
        fputs("Failed to load text weights\n", stderr);
        return 1;
    }

    auto input_ids = mx::array(prompt_tokens.data(), {1, num_tokens}, mx::int32);
    auto base_embeds = mx::take(text.embed_tokens, input_ids, 0);

    auto prefix = mx::slice(base_embeds, {0, 0, 0}, {1, g_img_cfg.image_a_start, TEXT_DIM});
    auto mid = mx::slice(base_embeds, {0, g_img_cfg.image_a_end, 0}, {1, g_img_cfg.image_b_start, TEXT_DIM});
    auto suffix = mx::slice(base_embeds, {0, g_img_cfg.image_b_end, 0}, {1, num_tokens, TEXT_DIM});

    auto config = parse_generation_config(max_tokens);
    print_generation_config(config);

    int num_chunks = (B + chunk_size - 1) / chunk_size;
    printf("\nGenerating up to %d tokens for %d pair%s", max_tokens, B, B > 1 ? "s" : "");
    if (num_chunks > 1) {
        printf(" in %d chunks of %d", num_chunks, chunk_size);
    }
    puts("...");

    std::vector<std::vector<int32_t>> all_generated;
    all_generated.reserve(B);

    auto start = std::chrono::high_resolution_clock::now();

    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int chunk_start = chunk_idx * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, B);
        int chunk_B = chunk_end - chunk_start;

        if (num_chunks > 1) {
            printf("\n--- Chunk %d/%d (pairs %d-%d) ---\n",
                   chunk_idx + 1, num_chunks, chunk_start, chunk_end - 1);
        }

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

        auto prefix_batched = mx::broadcast_to(prefix, {chunk_B, prefix.shape(1), TEXT_DIM});
        auto mid_batched = mx::broadcast_to(mid, {chunk_B, mid.shape(1), TEXT_DIM});
        auto suffix_batched = mx::broadcast_to(suffix, {chunk_B, suffix.shape(1), TEXT_DIM});

        auto inputs_embeds = mx::concatenate({prefix_batched, batched_img_a, mid_batched, batched_img_b, suffix_batched}, 1);

        std::vector<int32_t> batched_ids(chunk_B * num_tokens);
        mx::eval(input_ids);
        const int32_t* src_ids = input_ids.data<int32_t>();
        for (int b = 0; b < chunk_B; ++b) {
            std::memcpy(batched_ids.data() + b * num_tokens, src_ids, num_tokens * sizeof(int32_t));
        }
        auto chunk_input_ids = mx::array(batched_ids.data(), {chunk_B, num_tokens}, mx::int32);

        std::vector<GridTHW> grids;
        for (int i = 0; i < chunk_B; ++i) {
            grids.push_back({1, g_img_cfg.grid, g_img_cfg.grid});
            grids.push_back({1, g_img_cfg.grid, g_img_cfg.grid});
        }
        auto rope = get_rope_index(chunk_input_ids, grids, {}, empty_array());
        auto position_ids = rope.position_ids;

        auto chunk_generated = generate_batched(inputs_embeds, &text, position_ids, config);

        for (auto& gen : chunk_generated) {
            all_generated.push_back(std::move(gen));
        }

        if (num_chunks > 1) {
            int chunk_tokens = 0;
            for (int i = chunk_start; i < chunk_end; ++i) {
                chunk_tokens += all_generated[i].size();
            }
            printf("Chunk generated %d tokens\n", chunk_tokens);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    int total_tokens = 0;
    for (const auto& gen : all_generated) {
        total_tokens += gen.size();
    }

    print_generation_timing(ms, total_tokens);

    for (int i = 0; i < B; ++i) {
        printf("\n--- Pair %d (A=%d vs B=%d) ---\n", i, pairs[i].first, pairs[i].second);
        printf("Generated %zu tokens\nToken IDs:", all_generated[i].size());
        for (size_t j = 0; j < all_generated[i].size(); ++j) {
            printf(" %d", all_generated[i][j]);
        }
        putchar('\n');

        print_token_analysis(all_generated[i], true);
    }

    return 0;
}

// Describe Mode: Single-Image Description

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

    int chunk_size = batch_size;
    if (chunk_env) {
        chunk_size = std::atoi(chunk_env);
        if (chunk_size <= 0) chunk_size = batch_size;
    }

    puts("=== Image Description Mode (448x448) ===");
    printf("Batch size: %d\n", batch_size);
    if (chunk_size < batch_size) {
        printf("Chunk size: %d (processing in chunks)\n", chunk_size);
    }

    std::string filenames_path = embeds_dir + "/filenames.txt";
    auto filenames = load_filenames(filenames_path);
    if (filenames.empty()) {
        fprintf(stderr, "Failed to open %s\n", filenames_path.c_str());
        return 1;
    }
    int num_images = static_cast<int>(filenames.size());
    printf("Loaded %d image filenames\n", num_images);

    if (num_images == 0) {
        fputs("No images found in embeddings directory\n", stderr);
        return 1;
    }

    std::vector<int> indices;
    if (indices_file_env) {
        std::ifstream indices_file(indices_file_env);
        if (!indices_file) {
            fprintf(stderr, "Failed to open indices file: %s\n", indices_file_env);
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
        printf("Loaded %zu indices from %s\n", indices.size(), indices_file_env);
    } else if (idx_env) {
        int idx = std::atoi(idx_env);
        if (idx >= 0 && idx < num_images) {
            indices.push_back(idx);
            printf("Using single image index: %d\n", idx);
        } else {
            fprintf(stderr, "GLM_IDX=%d is out of range [0, %d]\n", idx, num_images - 1);
            return 1;
        }
    } else if (random_indices) {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < batch_size && i < num_images; ++i) {
            indices.push_back(std::rand() % num_images);
        }
        printf("Generated %zu random indices\n", indices.size());
    } else {
        for (int i = 0; i < batch_size && i < num_images; ++i) {
            indices.push_back(i);
        }
        printf("Using first %zu images\n", indices.size());
    }

    int B = static_cast<int>(indices.size());
    if (B == 0) {
        fputs("No valid indices to process\n", stderr);
        return 1;
    }
    printf("\nProcessing %d images:\n", B);
    for (int i = 0; i < std::min(B, 10); ++i) {
        printf("  [%d] idx=%d (%s)\n", i, indices[i], filenames[indices[i]].c_str());
    }
    if (B > 10) {
        printf("  ... and %d more\n", B - 10);
    }

    std::string embeds_path = embeds_dir + "/embeddings.bin";
    std::ifstream embeds_file(embeds_path, std::ios::binary);
    if (!embeds_file) {
        fprintf(stderr, "Failed to open %s\n", embeds_path.c_str());
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
    printf("Loaded embeddings for %d images\n", B);

    std::string tokens_path = weights_dir + "/single_image_prompt_448.bin";
    auto prompt_tokens = load_prompt_tokens(tokens_path);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "Failed to open %s\n", tokens_path.c_str());
        return 1;
    }
    int num_tokens = static_cast<int>(prompt_tokens.size());
    printf("Loaded %d prompt tokens (single-image)\n", num_tokens);

    puts("Loading text model...");
    TextModelWeights text;
    if (!load_text_weights(&text, weights_dir + "/text_model.bin")) {
        fputs("Failed to load text weights\n", stderr);
        return 1;
    }

    const int image_start = g_img_cfg.image_start;
    const int image_end = g_img_cfg.image_end;

    auto input_ids = mx::array(prompt_tokens.data(), {1, num_tokens}, mx::int32);
    auto base_embeds = mx::take(text.embed_tokens, input_ids, 0);

    auto prefix = mx::slice(base_embeds, {0, 0, 0}, {1, image_start, TEXT_DIM});
    auto suffix = mx::slice(base_embeds, {0, image_end, 0}, {1, num_tokens, TEXT_DIM});

    auto config = parse_generation_config(max_tokens);
    print_generation_config(config);

    int num_chunks = (B + chunk_size - 1) / chunk_size;
    printf("\nGenerating up to %d tokens for %d images", max_tokens, B);
    if (num_chunks > 1) {
        printf(" in %d chunks of %d", num_chunks, chunk_size);
    }
    puts("...");

    std::vector<std::vector<int32_t>> all_generated;
    all_generated.reserve(B);

    auto start = std::chrono::high_resolution_clock::now();

    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int chunk_start = chunk_idx * chunk_size;
        int chunk_end = std::min(chunk_start + chunk_size, B);
        int chunk_B = chunk_end - chunk_start;

        if (num_chunks > 1) {
            printf("\n--- Chunk %d/%d (images %d-%d) ---\n",
                   chunk_idx + 1, num_chunks, chunk_start, chunk_end - 1);
        }

        std::vector<uint16_t> img_data(chunk_B * tokens_per_image * TEXT_DIM);
        const size_t img_bytes = tokens_per_image * TEXT_DIM * sizeof(uint16_t);
        for (int i = 0; i < chunk_B; ++i) {
            std::memcpy(img_data.data() + i * tokens_per_image * TEXT_DIM,
                        embeds[chunk_start + i].data(), img_bytes);
        }
        auto batched_img = mx::array(reinterpret_cast<const mlx::core::float16_t*>(img_data.data()),
                                     {chunk_B, tokens_per_image, TEXT_DIM}, mx::float16);

        auto prefix_batched = mx::broadcast_to(prefix, {chunk_B, prefix.shape(1), TEXT_DIM});
        auto suffix_batched = mx::broadcast_to(suffix, {chunk_B, suffix.shape(1), TEXT_DIM});

        auto inputs_embeds = mx::concatenate({prefix_batched, batched_img, suffix_batched}, 1);

        std::vector<int32_t> batched_ids(chunk_B * num_tokens);
        mx::eval(input_ids);
        const int32_t* src_ids = input_ids.data<int32_t>();
        for (int b = 0; b < chunk_B; ++b) {
            std::memcpy(batched_ids.data() + b * num_tokens, src_ids, num_tokens * sizeof(int32_t));
        }
        auto chunk_input_ids = mx::array(batched_ids.data(), {chunk_B, num_tokens}, mx::int32);

        std::vector<GridTHW> grids;
        for (int i = 0; i < chunk_B; ++i) {
            grids.push_back({1, g_img_cfg.grid, g_img_cfg.grid});
        }
        auto rope = get_rope_index(chunk_input_ids, grids, {}, empty_array());
        auto position_ids = rope.position_ids;

        auto chunk_generated = generate_batched(inputs_embeds, &text, position_ids, config);

        for (auto& gen : chunk_generated) {
            all_generated.push_back(std::move(gen));
        }

        if (num_chunks > 1) {
            int chunk_tokens = 0;
            for (int i = chunk_start; i < chunk_end; ++i) {
                chunk_tokens += all_generated[i].size();
            }
            printf("Chunk generated %d tokens\n", chunk_tokens);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    int total_tokens = 0;
    for (const auto& gen : all_generated) {
        total_tokens += gen.size();
    }

    print_generation_timing(ms, total_tokens);

    for (int i = 0; i < B; ++i) {
        printf("\n--- Image %d (idx=%d: %s) ---\n", i, indices[i], filenames[indices[i]].c_str());
        printf("Generated %zu tokens\nToken IDs:", all_generated[i].size());
        for (size_t j = 0; j < all_generated[i].size(); ++j) {
            printf(" %d", all_generated[i][j]);
        }
        putchar('\n');

        print_token_analysis(all_generated[i], false);
    }

    return 0;
}

// Print Usage Information

void print_usage() {
    puts(R"(GLM-4V MLX Inference Engine

Usage: Set one of the following environment variables:

Modes:
  GLM_PRECOMPUTE=1         Precompute vision embeddings from images
  GLM_GENERATE=1           Dual-image comparison (single or batched)
  GLM_DESCRIBE=1           Single-image description (single or batched)

GLM_GENERATE options:
  GLM_IDX_A=N GLM_IDX_B=M  Single pair (both required for single mode)
  GLM_PAIRS_FILE=path      Load pairs from file (idx_a,idx_b per line)
  GLM_BATCH_SIZE=N         Number of random pairs (default: 4)
  GLM_CHUNK_SIZE=N         Inference batch size (default: BATCH_SIZE)

GLM_DESCRIBE options:
  GLM_IDX=N                Single explicit image index
  GLM_INDICES_FILE=path    Load indices from file (one per line)
  GLM_RANDOM_INDICES=1     Use random indices instead of first N
  GLM_BATCH_SIZE=N         Number of images to process (default: 64)
  GLM_CHUNK_SIZE=N         Inference batch size (default: BATCH_SIZE)

Common options:
  GLM_IMAGE_SIZE=N         Image size in pixels (default: 448, also: 336)
  GLM_WEIGHTS_DIR=path     Model weights directory (default: vision_weights)
  GLM_EMBEDS_DIR=path      Embeddings directory (default: embeddings_out)
  GLM_MAX_TOKENS=N         Max tokens to generate (default: 100)

Sampling options:
  GLM_DO_SAMPLE=1          Enable sampling (default: greedy)
  GLM_TEMPERATURE=0.8      Temperature for sampling
  GLM_TOP_K=50             Top-K sampling
  GLM_TOP_P=0.9            Top-P (nucleus) sampling
  GLM_REP_PENALTY=1.1      Repetition penalty
  GLM_SEED=N               Random seed

Examples:
  GLM_PRECOMPUTE=1 GLM_INPUT_DIR=images GLM_OUTPUT_DIR=embeds ./glm46v_mlx
  GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds ./glm46v_mlx
  GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds GLM_IDX_A=0 GLM_IDX_B=1 ./glm46v_mlx
  GLM_DESCRIBE=1 GLM_EMBEDS_DIR=embeds ./glm46v_mlx
  GLM_DESCRIBE=1 GLM_EMBEDS_DIR=embeds GLM_IDX=5 ./glm46v_mlx)");
}

int main(int argc, char* argv[]) {
    auto gpu = mx::Device(mx::Device::gpu, 0);
    if (mx::is_available(gpu)) {
        mx::set_default_device(gpu);
    } else {
        mx::set_default_device(mx::Device(mx::Device::cpu, 0));
        puts("GPU not available; using CPU.");
    }

    mx::enable_compile();
    mx::set_compile_mode(mx::CompileMode::enabled);

    const char* weights_dir_env = std::getenv("GLM_WEIGHTS_DIR");

    const char* image_size_env = std::getenv("GLM_IMAGE_SIZE");
    int image_size = image_size_env ? std::atoi(image_size_env) : 448;
    g_img_cfg = ImageConfig::from_size(image_size);
    puts("Image configuration:");
    g_img_cfg.print();

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

    const char* generate_env = std::getenv("GLM_GENERATE");
    if (generate_env && std::string(generate_env) == "1") {
        std::string weights_dir = weights_dir_env ? weights_dir_env : "vision_weights";
        return run_generate_mode(weights_dir);
    }

    const char* describe_env = std::getenv("GLM_DESCRIBE");
    if (describe_env && std::string(describe_env) == "1") {
        std::string weights_dir = weights_dir_env ? weights_dir_env : "vision_weights";
        return run_describe_mode(weights_dir);
    }

    print_usage();
    return 0;
}
