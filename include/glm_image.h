#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <string>
#include <turbojpeg.h>

namespace mx = mlx::core;

// JPEG/Image loading
std::vector<uint8_t> read_file_bytes(const std::string& path);
bool decode_jpeg(tjhandle handle, const std::string& path,
                 int* out_w, int* out_h, std::vector<uint8_t>* pixels);

// Bicubic resizing
void resize_and_pad(const std::vector<uint8_t>& src, int sw, int sh,
                    std::vector<uint8_t>* dst, int target_size);

// Directory listing
std::vector<std::string> list_images_recursive(const std::string& dir);

// GLM-4V preprocessing
mx::array normalize_and_patchify(const std::vector<uint8_t>& pixels, int size, int patch_size);
mx::array batch_patches(const std::vector<mx::array>& patches_list);
