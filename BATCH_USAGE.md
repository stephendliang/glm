# Batched Generation Usage

## Quick Start

```bash
# Single pair (baseline)
make generate EMBEDS=embeddings_out A=0 B=1 TOKENS=100

# Batched pairwise (4x parallel)
make batch-generate EMBEDS=embeddings_out BATCH=4 TOKENS=100

# Optimal pairwise throughput (64 pairs)
make batch-generate EMBEDS=embeddings_out BATCH=64 TOKENS=50

# Batched single-image description (128 images, optimal)
make batch-single EMBEDS=embeddings_out BATCH=128 TOKENS=256
```

## Environment Variables

### Pairwise Comparison Mode
| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_BATCH_GENERATE` | - | Set to `1` to enable batch pairwise mode |
| `GLM_GENERATE` | - | Set to `1` for single-pair mode |
| `GLM_EMBEDS_DIR` | `embeddings_512` | Directory with `embeddings.bin` + `filenames.txt` |
| `GLM_BATCH_SIZE` | `4` | Number of pairs to process in parallel |
| `GLM_PAIRS_FILE` | - | Optional file with `idx_a,idx_b` per line |
| `GLM_IDX_A` / `GLM_IDX_B` | random | Specific indices for single-pair mode |
| `GLM_MAX_TOKENS` | `100` | Max tokens to generate per sequence |
| `GLM_WEIGHTS_DIR` | `vision_weights` | Directory with model weights |

### Single-Image Description Mode
| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_BATCH_SINGLE_IMAGE` | - | Set to `1` to enable batch single-image mode |
| `GLM_SINGLE_IMAGE` | - | Set to `1` for single image mode (B=1) |
| `GLM_EMBEDS_DIR` | `embeddings_out` | Directory with `embeddings.bin` + `filenames.txt` |
| `GLM_BATCH_SIZE` | `64` | Number of images to process in parallel |
| `GLM_INDICES_FILE` | - | Optional file with one index per line |
| `GLM_RANDOM_INDICES` | - | Set to `1` to use random indices |
| `GLM_IMAGE_IDX` | `0` | Specific index for single-image mode |
| `GLM_MAX_TOKENS` | `100` | Max tokens to generate per sequence |

### Sampling Config
| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_DO_SAMPLE` | `1` | `0`=greedy, `1`=sampling |
| `GLM_TEMPERATURE` | `0.8` | Temperature for sampling |
| `GLM_TOP_K` | `50` | Top-k filtering |
| `GLM_TOP_P` | `0.6` | Nucleus sampling threshold |
| `GLM_REP_PENALTY` | `1.0` | Repetition penalty (1.0=none) |
| `GLM_SEED` | random | Random seed for reproducibility |

## Examples

### Single Pair Generation
```bash
# Random pair
GLM_GENERATE=1 GLM_EMBEDS_DIR=embeddings_out GLM_WEIGHTS_DIR=vision_weights ./glm46v_mlx

# Specific pair
GLM_GENERATE=1 GLM_IDX_A=10 GLM_IDX_B=20 GLM_EMBEDS_DIR=embeddings_out ./glm46v_mlx

# Greedy decoding
GLM_GENERATE=1 GLM_DO_SAMPLE=0 GLM_EMBEDS_DIR=embeddings_out ./glm46v_mlx

# Reproducible
GLM_GENERATE=1 GLM_SEED=42 GLM_EMBEDS_DIR=embeddings_out ./glm46v_mlx
```

### Batched Generation
```bash
# 8 random pairs
GLM_BATCH_GENERATE=1 GLM_BATCH_SIZE=8 GLM_EMBEDS_DIR=embeddings_out ./glm46v_mlx

# From pairs file
echo "0,1
2,3
4,5
6,7" > pairs.txt
GLM_BATCH_GENERATE=1 GLM_PAIRS_FILE=pairs.txt GLM_EMBEDS_DIR=embeddings_out ./glm46v_mlx

# Max throughput config
GLM_BATCH_GENERATE=1 GLM_BATCH_SIZE=64 GLM_MAX_TOKENS=50 GLM_SEED=42 \
  GLM_EMBEDS_DIR=embeddings_out GLM_WEIGHTS_DIR=vision_weights ./glm46v_mlx
```

### Makefile Shortcuts (Pairwise)
```bash
make generate                              # Random single pair
make generate A=5 B=10 TOKENS=50           # Specific pair
make batch-generate                        # 4 random pairs
make batch-generate BATCH=64 TOKENS=50    # 64 pairs, 50 tokens each
make batch-generate PAIRS=pairs.txt        # From file
```

### Batched Single-Image Description
```bash
# First 64 images
GLM_BATCH_SINGLE_IMAGE=1 GLM_BATCH_SIZE=64 GLM_EMBEDS_DIR=embeddings_out ./glm46v_mlx

# From indices file
echo "0
5
10
15" > indices.txt
GLM_BATCH_SINGLE_IMAGE=1 GLM_INDICES_FILE=indices.txt GLM_EMBEDS_DIR=embeddings_out ./glm46v_mlx

# Random indices
GLM_BATCH_SINGLE_IMAGE=1 GLM_RANDOM_INDICES=1 GLM_BATCH_SIZE=32 GLM_EMBEDS_DIR=embeddings_out ./glm46v_mlx

# Max throughput config (~92 tok/s)
GLM_BATCH_SINGLE_IMAGE=1 GLM_BATCH_SIZE=128 GLM_MAX_TOKENS=256 GLM_SEED=42 \
  GLM_EMBEDS_DIR=embeddings_out GLM_WEIGHTS_DIR=vision_weights ./glm46v_mlx
```

### Makefile Shortcuts (Single-Image)
```bash
make batch-single                          # First 64 images
make batch-single BATCH=32 TOKENS=50       # 32 images, 50 tokens each
make batch-single INDICES=indices.txt      # From file
```

## Performance

### Pairwise Comparison (546 tokens/seq)
| Batch | tok/s | Speedup | Notes |
|-------|-------|---------|-------|
| 1 | 8.0 | 1.0x | Baseline |
| 4 | 21.7 | 2.7x | |
| 8 | 27.2 | 3.4x | Good efficiency |
| 32 | 29.5 | 3.7x | |
| **64** | **33.3** | **4.2x** | **Recommended** |
| 96 | 33.4 | 4.2x | Peak throughput |
| 128 | 30.4 | 3.8x | Max stable |
| 192+ | - | - | OOM |

**Recommendation**: Use `BATCH=64` for optimal throughput (~33 tok/s, ~40 comparisons/min at 50 tokens each).

### Single-Image Description (270 tokens/seq, 256 max output)
| Batch | tok/s | Speedup | Notes |
|-------|-------|---------|-------|
| 32 | 46.1 | 1.0x | Baseline |
| 64 | 79.0 | 1.7x | Good balance |
| 80 | 76.0 | 1.6x | |
| 96 | 69.6 | 1.5x | Performance dip |
| 100 | 76.0 | 1.6x | |
| 112 | 84.1 | 1.8x | |
| **128** | **91.8** | **2.0x** | **Peak throughput** |
| 138 | 86.2 | 1.9x | Max tested |

**Recommendation**: Use `BATCH=128` for max throughput (~92 tok/s). Use `BATCH=64` for balanced memory/speed (~79 tok/s). Avoid `BATCH=96` due to unexplained performance dip.

Single-image has ~50% shorter sequences (270 vs 546 tokens) compared to pairwise, enabling higher throughput.

## Pairs File Format (Pairwise Mode)

Simple CSV, one pair per line:
```
0,1
2,3
10,20
```

Indices reference `filenames.txt` in the embeddings directory.

## Indices File Format (Single-Image Mode)

One index per line:
```
0
5
10
15
```

Indices reference `filenames.txt` in the embeddings directory.

## Precomputing Embeddings

Before generation, precompute vision embeddings:
```bash
make precompute INPUT=/path/to/images OUTPUT=my_embeddings BATCH=8
```

This creates:
- `my_embeddings/embeddings.bin` - float16 embeddings [N, 256, 4096]
- `my_embeddings/filenames.txt` - image paths, one per line
