# Makefile for GLM-4.6V MLX C++ implementation

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra \
           -Wunused-function \
           -Wunused-variable \
           -Wunused-parameter \
           -Wunreachable-code \
           -ffunction-sections \
           -fdata-sections

# Directory structure
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# MLX paths (homebrew installation)
MLX_INCLUDE = -I/opt/homebrew/include
LOCAL_INCLUDE = -I$(INC_DIR)
MLX_LIB = -L/opt/homebrew/lib -lmlx

# macOS frameworks required by MLX
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate

# TurboJPEG (for image precompute)
TURBOJPEG = -lturbojpeg

# Linker flags
LDFLAGS = $(MLX_LIB) $(FRAMEWORKS) $(TURBOJPEG) -Wl,-rpath,/opt/homebrew/lib -Wl,-dead_strip -Wl,-map,dead_code_map.txt

TARGET = glm46v_mlx
SRCS = glm46v_mlx.cpp glm_image.cpp glm_models.cpp glm_vision.cpp glm_text.cpp
OBJS = $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean run verify verify-step bench bench-int8 bench-dueling verify-text verify-cpu export-weights test test-fp16 test-int8 export-int8 export-prompt precompute dueling analyze

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(MLX_INCLUDE) $(LOCAL_INCLUDE) -c -o $@ $<

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -rf $(TARGET) $(BUILD_DIR) dead_code_map.txt

bench: $(TARGET)
	./$(TARGET)

verify: $(TARGET)
	GLM_VERIFY=1 GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

verify-text: $(TARGET)
	GLM_VERIFY_TEXT=1 GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

verify-step: $(TARGET)
	GLM_VERIFY_STEP=1 GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

verify-cpu: $(TARGET)
	GLM_FORCE_CPU=1 GLM_VERIFY=1 GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

test-fp16: $(TARGET)
	GLM_FP16_TEST=1 GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

export-int8:
	python3 export_vision_weights.py --text-int8

test-int8: $(TARGET)
	GLM_INT8_TEST=1 GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

bench-int8: $(TARGET)
	GLM_INT8_BENCH=1 GLM_BATCH_SIZE=16 GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

bench-dueling: $(TARGET)
	GLM_DUELING_BENCH=1 GLM_BATCH_SIZE=8 GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

export-prompt:
	python3 export_prompt_tokens.py

export-weights:
	python3 export_vision_weights.py

test: export-weights $(TARGET) verify

# Precompute vision embeddings from a folder of images
# Usage: make precompute INPUT=/path/to/images OUTPUT=/path/to/output
precompute: $(TARGET)
	GLM_PRECOMPUTE=1 GLM_INPUT_DIR=$(INPUT) GLM_OUTPUT_DIR=$(OUTPUT) \
	GLM_BATCH_SIZE=$(or $(BATCH),8) GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

# Compare two images from precomputed embeddings
# Usage: make dueling [EMBEDS=embeddings_512] [A=idx] [B=idx]
# Random indices if A/B not specified
dueling: $(TARGET)
	GLM_DUELING=1 GLM_EMBEDS_DIR=$(or $(EMBEDS),embeddings_512) \
	GLM_IDX_A=$(or $(A),-1) GLM_IDX_B=$(or $(B),-1) \
	GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

# Single-pair generation with KV cache
# Usage: make generate [EMBEDS=embeddings_512] [A=idx] [B=idx] [TOKENS=100]
generate: $(TARGET)
	GLM_GENERATE=1 GLM_EMBEDS_DIR=$(or $(EMBEDS),embeddings_512) \
	GLM_IDX_A=$(or $(A),-1) GLM_IDX_B=$(or $(B),-1) \
	GLM_MAX_TOKENS=$(or $(TOKENS),100) \
	GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

# Batched generation - process multiple pairs in parallel
# Usage: make batch-generate [EMBEDS=embeddings_512] [BATCH=4] [TOKENS=100] [PAIRS=pairs.txt]
# PAIRS file format: idx_a,idx_b per line (e.g., "0,1\n2,3\n...")
batch-generate: $(TARGET)
	GLM_BATCH_GENERATE=1 GLM_EMBEDS_DIR=$(or $(EMBEDS),embeddings_512) \
	GLM_BATCH_SIZE=$(or $(BATCH),4) GLM_MAX_TOKENS=$(or $(TOKENS),100) \
	$(if $(PAIRS),GLM_PAIRS_FILE=$(PAIRS)) \
	GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

# Batched single-image description - process multiple images in parallel
# Usage: make batch-single [EMBEDS=embeddings_out] [BATCH=64] [TOKENS=100] [INDICES=indices.txt]
# INDICES file format: one index per line
batch-single: $(TARGET)
	GLM_BATCH_SINGLE_IMAGE=1 GLM_EMBEDS_DIR=$(or $(EMBEDS),embeddings_out) \
	GLM_BATCH_SIZE=$(or $(BATCH),64) GLM_MAX_TOKENS=$(or $(TOKENS),100) \
	$(if $(INDICES),GLM_INDICES_FILE=$(INDICES)) \
	GLM_WEIGHTS_DIR=vision_weights ./$(TARGET)

# Static analysis for unused functions
analyze:
	cppcheck --enable=unusedFunction $(SRC_DIR) $(INC_DIR)
