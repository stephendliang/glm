# Batched Generation Usage

## Quick Start

```bash
# Single pair (baseline)
make generate EMBEDS=embeddings_out A=0 B=1 TOKENS=100

# Batched (4x parallel)
make batch-generate EMBEDS=embeddings_out BATCH=4 TOKENS=100

# Optimal throughput (64 pairs)
make batch-generate EMBEDS=embeddings_out BATCH=64 TOKENS=50
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_BATCH_GENERATE` | - | Set to `1` to enable batch mode |
| `GLM_GENERATE` | - | Set to `1` for single-pair mode |
| `GLM_EMBEDS_DIR` | `embeddings_512` | Directory with `embeddings.bin` + `filenames.txt` |
| `GLM_BATCH_SIZE` | `4` | Number of pairs to process in parallel |
| `GLM_PAIRS_FILE` | - | Optional file with `idx_a,idx_b` per line |
| `GLM_IDX_A` / `GLM_IDX_B` | random | Specific indices for single-pair mode |
| `GLM_MAX_TOKENS` | `100` | Max tokens to generate per sequence |
| `GLM_WEIGHTS_DIR` | `vision_weights` | Directory with model weights |

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

### Makefile Shortcuts
```bash
make generate                              # Random single pair
make generate A=5 B=10 TOKENS=50           # Specific pair
make batch-generate                        # 4 random pairs
make batch-generate BATCH=64 TOKENS=50    # 64 pairs, 50 tokens each
make batch-generate PAIRS=pairs.txt        # From file
```

## Performance

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

## Pairs File Format

Simple CSV, one pair per line:
```
0,1
2,3
10,20
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
