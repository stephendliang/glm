# GLM API Reference

## Current API Inventory (22 env vars)

### Modes (3)
| Var | Purpose | Status |
|-----|---------|--------|
| `GLM_PRECOMPUTE=1` | Precompute vision embeddings from images | Keep |
| `GLM_GENERATE=1` | Dual-image comparison (single or batched) | Keep |
| `GLM_DESCRIBE=1` | Single-image description (single or batched) | Keep |

### PRECOMPUTE Options (3)
| Var | Purpose | Status |
|-----|---------|--------|
| `GLM_INPUT_DIR=path` | Source images directory | Keep |
| `GLM_OUTPUT_DIR=path` | Output embeddings directory | Keep |
| `GLM_BATCH_SIZE=N` | Vision batch size | Keep |

### GENERATE Options (4)
| Var | Purpose | Status |
|-----|---------|--------|
| `GLM_IDX_A=N`, `GLM_IDX_B=M` | Single explicit pair | Keep |
| `GLM_PAIRS_FILE=path` | Load pairs from file (idx_a,idx_b per line) | Review |
| `GLM_BATCH_SIZE=N` | Number of random pairs (default: 4) | Keep |

### DESCRIBE Options (4)
| Var | Purpose | Status |
|-----|---------|--------|
| `GLM_IDX=N` | Single explicit image index | Keep |
| `GLM_INDICES_FILE=path` | Load indices from file (one per line) | Review |
| `GLM_RANDOM_INDICES=1` | Use random indices instead of first N | **Cull candidate** |
| `GLM_BATCH_SIZE=N` | Number of images (default: 64) | Keep |

### Common Options (3)
| Var | Purpose | Status |
|-----|---------|--------|
| `GLM_WEIGHTS_DIR=path` | Model weights directory (default: vision_weights) | Keep |
| `GLM_EMBEDS_DIR=path` | Embeddings directory (default: embeddings_out) | Keep |
| `GLM_MAX_TOKENS=N` | Max tokens to generate (default: 100) | Keep |

### Sampling Options (6)
| Var | Purpose | Status |
|-----|---------|--------|
| `GLM_DO_SAMPLE=1` | Enable sampling (default: greedy) | **Cull candidate** |
| `GLM_TEMPERATURE=0.8` | Temperature for sampling | Keep |
| `GLM_TOP_K=50` | Top-K sampling | **Cull candidate** |
| `GLM_TOP_P=0.9` | Top-P (nucleus) sampling | Keep |
| `GLM_REP_PENALTY=1.1` | Repetition penalty | **Cull candidate** |
| `GLM_SEED=N` | Random seed | **Cull candidate** |

---

## Future Cull Candidates (5 vars)

| Var | Reason to cull | Risk |
|-----|----------------|------|
| `GLM_RANDOM_INDICES` | Rarely needed - use `GLM_INDICES_FILE` with random list, or accept first N | Low |
| `GLM_DO_SAMPLE` | Could default to sampling ON with temp=0.8 (current behavior anyway) | Medium - loses greedy mode |
| `GLM_TOP_K` | Almost never tweaked, top_p is more useful | Low |
| `GLM_REP_PENALTY` | Marginal effect, often left at 1.0 | Low |
| `GLM_SEED` | Only needed for reproducibility testing | Very low |

---

## Usage Examples

```bash
# Precompute embeddings
GLM_PRECOMPUTE=1 GLM_INPUT_DIR=images GLM_OUTPUT_DIR=embeds ./glm46v_mlx

# Describe single image
GLM_DESCRIBE=1 GLM_EMBEDS_DIR=embeds GLM_IDX=5 ./glm46v_mlx

# Describe first 64 images
GLM_DESCRIBE=1 GLM_EMBEDS_DIR=embeds ./glm46v_mlx

# Compare single pair
GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds GLM_IDX_A=0 GLM_IDX_B=1 ./glm46v_mlx

# Compare 8 random pairs
GLM_GENERATE=1 GLM_EMBEDS_DIR=embeds GLM_BATCH_SIZE=8 ./glm46v_mlx
```

---

*Last updated: 2026-01-19*
