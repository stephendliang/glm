# Makefile for GLM-4.6V MLX C++ implementation

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall

# MLX paths (homebrew installation)
MLX_INCLUDE = -I/opt/homebrew/include
MLX_LIB = -L/opt/homebrew/lib -lmlx

# macOS frameworks required by MLX
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate

# TurboJPEG (for image precompute)
TURBOJPEG = -lturbojpeg

# Linker flags
LDFLAGS = $(MLX_LIB) $(FRAMEWORKS) $(TURBOJPEG) -Wl,-rpath,/opt/homebrew/lib

TARGET = glm46v_mlx
SRC = glm46v_mlx.cpp

.PHONY: all clean run verify verify-step bench bench-text bench-int8 bench-dueling verify-text verify-cpu export-weights test test-fp16 test-int8 export-int8 export-prompt precompute dueling

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(MLX_INCLUDE) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

bench: $(TARGET)
	./$(TARGET)

bench-text: $(TARGET)
	GLM_RUN_TEXT=1 ./$(TARGET)

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
