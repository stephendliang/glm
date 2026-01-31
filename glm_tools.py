#!/usr/bin/env python3
"""
GLM-4.6V unified tooling for debugging and testing.

Usage:
    python3.13 glm_tools.py <command> [options]

Commands:
    build             Build the C++ binary via Makefile
    run               Run C++ binary with env vars
    export            Export weights or prompts
    compare           Compare Python vs C++ outputs
    generate          Run Python generation
    generate-cpp      Run C++ generation
    test              Run parity tests
    infer             Inference with embedding cache
    inspect           Inspect model forward methods
"""

import argparse
import importlib
import inspect
import numpy as np
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Configuration
MODEL_PATH = "zai-org/GLM-4.6V-Flash"
PROJECT_DIR = Path(__file__).parent.absolute()
WEIGHTS_DIR = PROJECT_DIR / "vision_weights"
os.environ['HF_HOME'] = '/Users/Stephen/.cache/huggingface'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# GLM-4V normalization constants (must match C++)
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Lazy model loading
_model = None
_processor = None
_tokenizer = None


def get_tokenizer():
    """Lazy load tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        print("Loading tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return _tokenizer


def get_model(dtype='bfloat16'):
    """Lazy load model."""
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        print("Loading model...")
        _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        torch_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
        _model = Glm4vForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            device_map='cpu',
            trust_remote_code=True,
        )
        _model.eval()
    return _model, _processor


def preprocess_image(image_path: str, target_size: int = 448) -> np.ndarray:
    """Preprocess image identically to C++."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    # Resize with aspect ratio preservation
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Center pad with black
    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(img_resized, (paste_x, paste_y))

    # Convert to numpy and normalize
    pixels = np.array(padded, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    return pixels


# =============================================================================
# C++ Integration Commands
# =============================================================================

def cmd_build(args):
    """Build the C++ binary via Makefile."""
    print("Building C++ binary...")
    result = subprocess.run(
        ['make', '-j4'],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Build failed!")
        print(result.stderr)
        return 1
    print("Build successful!")
    if args.verbose:
        print(result.stdout)
    return 0


def cmd_run(args):
    """Run C++ binary with environment variables."""
    env = os.environ.copy()
    env['GLM_WEIGHTS_DIR'] = str(args.weights_dir or WEIGHTS_DIR)

    if args.generate:
        env['GLM_GENERATE'] = '1'
    if args.single_image:
        env['GLM_SINGLE_IMAGE'] = '1'
    if args.precompute:
        env['GLM_PRECOMPUTE'] = '1'
    if args.embeds_dir:
        env['GLM_EMBEDS_DIR'] = args.embeds_dir
    if args.images_dir:
        env['GLM_IMAGES_DIR'] = args.images_dir
    if args.image_idx is not None:
        env['GLM_IMAGE_IDX'] = str(args.image_idx)
    if args.max_tokens:
        env['GLM_MAX_TOKENS'] = str(args.max_tokens)
    if args.do_sample is not None:
        env['GLM_DO_SAMPLE'] = '1' if args.do_sample else '0'
    if args.debug_vision:
        env['GLM_DEBUG_VISION'] = '1'

    binary = PROJECT_DIR / 'glm46v_mlx'
    if not binary.exists():
        print("Binary not found. Building...")
        if cmd_build(argparse.Namespace(verbose=False)) != 0:
            return 1

    result = subprocess.run(
        [str(binary)],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=not args.stream,
        text=True
    )
    if not args.stream:
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    return result.returncode


def cmd_generate_cpp(args):
    """Run C++ generation (convenience wrapper)."""
    run_args = argparse.Namespace(
        generate=True,
        single_image=True,
        precompute=False,
        embeds_dir=args.embeds_dir,
        images_dir=None,
        image_idx=args.image_idx,
        max_tokens=args.max_tokens,
        do_sample=args.do_sample,
        weights_dir=args.weights_dir,
        stream=args.stream,
        debug_vision=args.debug_vision
    )
    return cmd_run(run_args)


def cmd_precompute(args):
    """Precompute embeddings for images (C++)."""
    run_args = argparse.Namespace(
        generate=False,
        single_image=False,
        precompute=True,
        embeds_dir=args.output_dir,
        images_dir=args.images_dir,
        image_idx=None,
        max_tokens=None,
        do_sample=None,
        weights_dir=args.weights_dir,
        stream=False,
        debug_vision=args.debug_vision
    )
    return cmd_run(run_args)


# =============================================================================
# Export Commands
# =============================================================================

def cmd_export_weights(args):
    """Export vision and/or text weights to binary."""
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = get_model(dtype='float16')
    visual = model.model.visual

    if args.vision:
        print("\n=== Exporting Vision Weights ===")
        weights = {}

        # Patch embedding conv3d: PyTorch [out, in, kD, kH, kW] -> MLX [out, kD, kH, kW, in]
        patch_weight = visual.patch_embed.proj.weight.data
        weights["patch_embed.weight"] = patch_weight.permute(0, 2, 3, 4, 1).contiguous()
        if visual.patch_embed.proj.bias is not None:
            weights["patch_embed.bias"] = visual.patch_embed.proj.bias.data

        # Position embedding
        weights["pos_embed.weight"] = visual.embeddings.position_embedding.weight.data

        # Layernorms
        weights["post_conv_layernorm.weight"] = visual.post_conv_layernorm.weight.data
        weights["post_layernorm.weight"] = visual.post_layernorm.weight.data

        # Transformer blocks
        for i, block in enumerate(visual.blocks):
            prefix = f"blocks.{i}."
            weights[prefix + "norm1.weight"] = block.norm1.weight.data
            weights[prefix + "norm2.weight"] = block.norm2.weight.data
            weights[prefix + "attn.qkv.weight"] = block.attn.qkv.weight.data.T.contiguous()
            if block.attn.qkv.bias is not None:
                weights[prefix + "attn.qkv.bias"] = block.attn.qkv.bias.data
            weights[prefix + "attn.proj.weight"] = block.attn.proj.weight.data.T.contiguous()
            if block.attn.proj.bias is not None:
                weights[prefix + "attn.proj.bias"] = block.attn.proj.bias.data
            weights[prefix + "mlp.gate_proj.weight"] = block.mlp.gate_proj.weight.data.T.contiguous()
            weights[prefix + "mlp.up_proj.weight"] = block.mlp.up_proj.weight.data.T.contiguous()
            weights[prefix + "mlp.down_proj.weight"] = block.mlp.down_proj.weight.data.T.contiguous()

        # Downsample conv2d: PyTorch [out, in, kH, kW] -> MLX [out, kH, kW, in]
        downsample_weight = visual.downsample.weight.data
        weights["downsample.weight"] = downsample_weight.permute(0, 2, 3, 1).contiguous()
        if visual.downsample.bias is not None:
            weights["downsample.bias"] = visual.downsample.bias.data

        # Merger
        weights["merger.proj.weight"] = visual.merger.proj.weight.data.T.contiguous()
        if visual.merger.proj.bias is not None:
            weights["merger.proj.bias"] = visual.merger.proj.bias.data
        weights["merger.post_ln.weight"] = visual.merger.post_projection_norm.weight.data
        weights["merger.post_ln.bias"] = visual.merger.post_projection_norm.bias.data
        weights["merger.gate_proj.weight"] = visual.merger.gate_proj.weight.data.T.contiguous()
        weights["merger.up_proj.weight"] = visual.merger.up_proj.weight.data.T.contiguous()
        weights["merger.down_proj.weight"] = visual.merger.down_proj.weight.data.T.contiguous()

        # Save to binary
        weights_path = output_dir / "vision_encoder.bin"
        print(f"Saving to {weights_path}...")
        total_bytes = 0

        with open(weights_path, 'wb') as f:
            def write(name, tensor):
                nonlocal total_bytes
                arr = tensor.numpy().astype(np.float16)
                arr.tofile(f)
                total_bytes += arr.nbytes
                if args.verbose:
                    print(f"  {name}: {list(arr.shape)}")

            def write_zeros(name, shape):
                nonlocal total_bytes
                arr = np.zeros(shape, dtype=np.float16)
                arr.tofile(f)
                total_bytes += arr.nbytes
                if args.verbose:
                    print(f"  {name}: {list(shape)} (zeros)")

            write("patch_embed.weight", weights["patch_embed.weight"])
            write("patch_embed.bias", weights.get("patch_embed.bias", torch.zeros(1536)))
            write("pos_embed.weight", weights["pos_embed.weight"])
            write("post_conv_layernorm.weight", weights["post_conv_layernorm.weight"])

            for i in range(len(visual.blocks)):
                p = f"blocks.{i}."
                write(f"{p}norm1.weight", weights[f"{p}norm1.weight"])
                write(f"{p}norm2.weight", weights[f"{p}norm2.weight"])
                write(f"{p}attn.qkv.weight", weights[f"{p}attn.qkv.weight"])
                write(f"{p}attn.qkv.bias", weights.get(f"{p}attn.qkv.bias", torch.zeros(4608)))
                write(f"{p}attn.proj.weight", weights[f"{p}attn.proj.weight"])
                write(f"{p}attn.proj.bias", weights.get(f"{p}attn.proj.bias", torch.zeros(1536)))
                write(f"{p}mlp.gate_proj.weight", weights[f"{p}mlp.gate_proj.weight"])
                write(f"{p}mlp.up_proj.weight", weights[f"{p}mlp.up_proj.weight"])
                write(f"{p}mlp.down_proj.weight", weights[f"{p}mlp.down_proj.weight"])

            write("post_layernorm.weight", weights["post_layernorm.weight"])
            write("downsample.weight", weights["downsample.weight"])
            write("downsample.bias", weights.get("downsample.bias", torch.zeros(4096)))
            write("merger.proj.weight", weights["merger.proj.weight"])
            write("merger.proj.bias", weights.get("merger.proj.bias", torch.zeros(4096)))
            write("merger.post_ln.weight", weights["merger.post_ln.weight"])
            write("merger.post_ln.bias", weights["merger.post_ln.bias"])
            write("merger.gate_proj.weight", weights["merger.gate_proj.weight"])
            write("merger.up_proj.weight", weights["merger.up_proj.weight"])
            write("merger.down_proj.weight", weights["merger.down_proj.weight"])

        print(f"Vision weights: {total_bytes / 1024 / 1024:.1f} MB")

    if args.text:
        print("\n=== Exporting Text Weights ===")
        text = model.model.language_model
        lm_head = model.lm_head
        weights_path = output_dir / "text_model.bin"
        print(f"Saving to {weights_path}...")
        total_bytes = 0

        with open(weights_path, 'wb') as f:
            def write(name, tensor):
                nonlocal total_bytes
                arr = tensor.detach().numpy().astype(np.float16)
                arr.tofile(f)
                total_bytes += arr.nbytes
                if args.verbose:
                    print(f"  {name}: {list(arr.shape)}")

            def write_zeros(name, shape):
                nonlocal total_bytes
                arr = np.zeros(shape, dtype=np.float16)
                arr.tofile(f)
                total_bytes += arr.nbytes

            # Embedding table
            write("embed_tokens", text.embed_tokens.weight.data)

            # Transformer layers
            for i, layer in enumerate(text.layers):
                p = f"layers.{i}."
                write(f"{p}input_layernorm", layer.input_layernorm.weight.data)
                write(f"{p}post_self_attn_layernorm", layer.post_self_attn_layernorm.weight.data)
                write(f"{p}post_attention_layernorm", layer.post_attention_layernorm.weight.data)
                write(f"{p}post_mlp_layernorm", layer.post_mlp_layernorm.weight.data)

                attn = layer.self_attn
                write(f"{p}q_proj.weight", attn.q_proj.weight.data.T.contiguous())
                if attn.q_proj.bias is not None:
                    write(f"{p}q_proj.bias", attn.q_proj.bias.data)
                else:
                    write_zeros(f"{p}q_proj.bias", (attn.q_proj.weight.shape[0],))

                write(f"{p}k_proj.weight", attn.k_proj.weight.data.T.contiguous())
                if attn.k_proj.bias is not None:
                    write(f"{p}k_proj.bias", attn.k_proj.bias.data)
                else:
                    write_zeros(f"{p}k_proj.bias", (attn.k_proj.weight.shape[0],))

                write(f"{p}v_proj.weight", attn.v_proj.weight.data.T.contiguous())
                if attn.v_proj.bias is not None:
                    write(f"{p}v_proj.bias", attn.v_proj.bias.data)
                else:
                    write_zeros(f"{p}v_proj.bias", (attn.v_proj.weight.shape[0],))

                write(f"{p}o_proj.weight", attn.o_proj.weight.data.T.contiguous())

                mlp = layer.mlp
                write(f"{p}gate_up_proj.weight", mlp.gate_up_proj.weight.data.T.contiguous())
                write(f"{p}down_proj.weight", mlp.down_proj.weight.data.T.contiguous())

            # Final norm
            write("norm.weight", text.norm.weight.data)

            # LM head
            write("lm_head.weight", lm_head.weight.data.T.contiguous())

        print(f"Text weights: {total_bytes / 1024 / 1024:.1f} MB")

    print("\nDone!")
    return 0


def cmd_export_prompt(args):
    """Export tokenized prompt to binary."""
    tokenizer = get_tokenizer()

    # Compute image tokens
    grid = args.size // 14
    merged_grid = grid // 2
    num_image_tokens = merged_grid * merged_grid
    print(f"Image size: {args.size}x{args.size} -> {num_image_tokens} tokens")

    # Build prompt
    prompt_parts = [
        "[gMASK]", "<sop>", "<|user|>", "\n",
        "<|begin_of_image|>",
        "<|image|>" * num_image_tokens,
        "<|end_of_image|>",
        args.prompt,
        "<|assistant|>", "\n",
    ]
    prompt = "".join(prompt_parts)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Tokenized: {len(tokens)} tokens")

    # Find image positions
    image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    begin_image_id = tokenizer.convert_tokens_to_ids("<|begin_of_image|>")
    end_image_id = tokenizer.convert_tokens_to_ids("<|end_of_image|>")

    image_start = None
    image_end = None
    for i, tok in enumerate(tokens):
        if tok == image_token_id and image_start is None:
            image_start = i
        if tok == end_image_id:
            image_end = i

    print(f"Image tokens: [{image_start}:{image_end}] ({image_end - image_start} tokens)")

    # Save tokens
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"single_image_prompt_{args.size}.bin"
    np.array(tokens, dtype=np.int32).tofile(output_path)
    print(f"Saved to {output_path}")

    # Save config header
    header_path = output_dir / f"single_image_config_{args.size}.h"
    with open(header_path, 'w') as f:
        f.write("// Auto-generated single-image prompt configuration\n")
        f.write(f"// Generated from: {MODEL_PATH}\n")
        f.write(f"// Image size: {args.size}x{args.size}\n\n")
        f.write("#pragma once\n\n")
        f.write(f"constexpr int SINGLE_PROMPT_LEN_{args.size} = {len(tokens)};\n")
        f.write(f"constexpr int SINGLE_IMAGE_START_{args.size} = {image_start};\n")
        f.write(f"constexpr int SINGLE_IMAGE_END_{args.size} = {image_end};\n")
        f.write(f"constexpr int SINGLE_NUM_IMAGE_TOKENS_{args.size} = {num_image_tokens};\n")
    print(f"Saved header to {header_path}")

    return 0


# =============================================================================
# Compare Commands
# =============================================================================

def cmd_compare_pipeline(args):
    """Compare C++ and Python vision encoder outputs at each stage."""
    import torch
    import torch.nn.functional as F

    print("=" * 60)
    print("Vision Pipeline Comparison: Python vs C++")
    print("=" * 60)
    print(f"Image: {args.image}")
    print()

    model, _ = get_model()
    visual = model.model.visual

    # Preprocess image
    print("Preprocessing image...")
    pixels = preprocess_image(args.image, args.size)
    print(f"Preprocessed pixels: shape={pixels.shape} mean={pixels.mean():.6f}")

    # Prepare input: [H, W, C] -> [1, C, T, H, W] with T=2
    pixels_chw = pixels.transpose(2, 0, 1)  # [C, H, W]
    pixels_cthw = np.stack([pixels_chw, pixels_chw], axis=1)  # [C, T, H, W]
    pixel_values = torch.from_numpy(pixels_cthw).unsqueeze(0).to(torch.bfloat16).contiguous()

    with torch.no_grad():
        # Step 1: patch_embed (conv3d)
        patch_embed = visual.patch_embed
        weight = patch_embed.proj.weight
        bias = patch_embed.proj.bias

        x = pixel_values.view(-1, 3, 2, 14, 14)  # [1024, 3, 2, 14, 14]
        x = F.conv3d(x, weight, bias, stride=(2, 14, 14))
        x = x.view(-1, 1536)  # [1024, 1536]

        print(f"\nAfter patch_embed: mean={x.float().mean():.6f} std={x.float().std():.6f}")
        print(f"  First 10 of patch 0: {x[0, :10].float().numpy()}")
        print(f"  First 10 of patch 512: {x[512, :10].float().numpy()}")

        # Get final embeddings
        grid = args.size // 14
        grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long)
        final_embeds = model.get_image_features(pixel_values, grid_thw)

        print(f"\nFinal embeddings: shape={final_embeds[0].shape}")
        py_embeds = final_embeds[0].float().numpy()
        print(f"  Mean: {py_embeds.mean():.6f}")
        print(f"  Std: {py_embeds.std():.6f}")
        print(f"  First 10 of token 0: {py_embeds[0, :10]}")

    # Load C++ embeddings if provided
    if args.cpp_embeds and os.path.exists(args.cpp_embeds):
        print(f"\nLoading C++ embeddings from {args.cpp_embeds}...")
        cpp_embeds = np.fromfile(args.cpp_embeds, dtype=np.float16)
        num_tokens = py_embeds.shape[0]
        dim = py_embeds.shape[1]
        cpp_embeds = cpp_embeds[args.image_idx * num_tokens * dim:(args.image_idx + 1) * num_tokens * dim]
        cpp_embeds = cpp_embeds.reshape(num_tokens, dim).astype(np.float32)

        print(f"C++ embeddings: mean={cpp_embeds.mean():.6f} std={cpp_embeds.std():.6f}")

        # Compare
        diff = np.abs(py_embeds - cpp_embeds)
        corr = np.corrcoef(py_embeds.flatten(), cpp_embeds.flatten())[0, 1]

        print(f"\nComparison:")
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Correlation: {corr:.6f}")

        if corr > 0.99:
            print("  [PASS] Embeddings match well!")
        elif corr > 0.90:
            print("  [WARN] Embeddings have minor differences")
        else:
            print("  [FAIL] Embeddings differ significantly!")

    return 0


def cmd_compare_embeds(args):
    """Compare C++ and Python embeddings."""
    import torch

    print("=" * 60)
    print("Embedding Comparison")
    print("=" * 60)

    # Load C++ embeddings
    if not os.path.exists(args.cpp_embeds):
        print(f"Error: C++ embeddings not found at {args.cpp_embeds}")
        return 1

    cpp_embeds = np.fromfile(args.cpp_embeds, dtype=np.float16)
    num_images = cpp_embeds.size // (256 * 4096)
    cpp_embeds = cpp_embeds.reshape(num_images, 256, 4096).astype(np.float32)

    print(f"Loaded {num_images} images from {args.cpp_embeds}")
    print(f"C++ stats: mean={cpp_embeds.mean():.6f} std={cpp_embeds.std():.6f}")

    if args.image:
        # Run Python vision encoder
        model, _ = get_model()

        pixels = preprocess_image(args.image, args.size)
        pixels_chw = pixels.transpose(2, 0, 1)
        pixels_cthw = np.stack([pixels_chw, pixels_chw], axis=1)
        pixel_values = torch.from_numpy(pixels_cthw).unsqueeze(0).to(torch.bfloat16)

        grid = args.size // 14
        grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long)

        with torch.no_grad():
            py_embeds = model.get_image_features(pixel_values, grid_thw)[0].float().numpy()

        print(f"Python stats: mean={py_embeds.mean():.6f} std={py_embeds.std():.6f}")

        # Compare with specified index
        cpp_single = cpp_embeds[args.image_idx]
        diff = np.abs(py_embeds - cpp_single)
        corr = np.corrcoef(py_embeds.flatten(), cpp_single.flatten())[0, 1]

        print(f"\nComparison (image index {args.image_idx}):")
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Correlation: {corr:.6f}")

    return 0


def cmd_compare_preproc(args):
    """Compare Python and C++ preprocessing."""
    from PIL import Image

    print("=" * 60)
    print("Preprocessing Comparison")
    print("=" * 60)

    # Python preprocessing
    py_pixels = preprocess_image(args.image, args.size)
    print(f"Python: shape={py_pixels.shape} mean={py_pixels.mean():.6f}")

    # Load C++ preprocessing if available
    if args.cpp_preproc and os.path.exists(args.cpp_preproc):
        cpp_pixels = np.fromfile(args.cpp_preproc, dtype=np.float32).reshape(args.size, args.size, 3)
        print(f"C++: shape={cpp_pixels.shape} mean={cpp_pixels.mean():.6f}")

        diff = np.abs(py_pixels - cpp_pixels)
        print(f"\nDifference:")
        print(f"  Max: {diff.max():.6f}")
        print(f"  Mean: {diff.mean():.6f}")

        if diff.max() < 1e-5:
            print("  [PASS] Preprocessing matches exactly!")
        else:
            print("  [WARN] Preprocessing differs")
    else:
        # Save for C++ comparison
        output_path = args.output or "python_preproc.bin"
        py_pixels.astype(np.float32).tofile(output_path)
        print(f"Saved Python preprocessing to {output_path}")

    return 0


# =============================================================================
# Generation Commands
# =============================================================================

def cmd_generate(args):
    """Run Python generation on images."""
    import torch
    from PIL import Image

    model, processor = get_model()
    tokenizer = get_tokenizer()

    for img_path in args.images:
        print(f"\n=== {img_path} ===")
        image = Image.open(img_path).convert('RGB')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )

        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                inputs[k] = v.to(torch.bfloat16)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        generated_tokens = output[0, input_len:].tolist()
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"Tokens: {generated_tokens}")
        print(f"Output: {decoded}")

    return 0


# =============================================================================
# Test Commands
# =============================================================================

def cmd_test_text(args):
    """Test text-only parity with C++."""
    import torch

    tokenizer = get_tokenizer()
    model, _ = get_model()

    prompts = args.prompts or [
        "What is the capital of France?",
        "How many pints are in a gallon?",
        "Name three colors.",
    ]

    for prompt in prompts:
        print(f"\n=== {prompt} ===")

        # Tokenize
        chat_prompt = f"[gMASK]<sop><|user|>\n{prompt}<|assistant|>\n"
        tokens = tokenizer.encode(chat_prompt, add_special_tokens=False)
        print(f"Input: {len(tokens)} tokens")

        # Python generation
        input_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        py_tokens = output[0, len(tokens):].tolist()
        print(f"Python: {py_tokens}")

        # TODO: Run C++ and compare
        # cpp_tokens = run_cpp_text_generation(tokens, args.max_tokens)
        # print(f"C++: {cpp_tokens}")

    return 0


def cmd_test_full(args):
    """Run full parity test suite."""
    print("=" * 60)
    print("Full Parity Test Suite")
    print("=" * 60)

    # Test 1: Text generation
    print("\n--- Test 1: Text Generation ---")
    cmd_test_text(argparse.Namespace(prompts=None, max_tokens=10))

    # Test 2: Vision pipeline
    if args.images:
        print("\n--- Test 2: Vision Pipeline ---")
        for img in args.images:
            cmd_compare_pipeline(argparse.Namespace(
                image=img, size=448, cpp_embeds=args.cpp_embeds, image_idx=0
            ))

    print("\n" + "=" * 60)
    print("Tests complete!")
    return 0


# =============================================================================
# Utility Commands
# =============================================================================

def cmd_inspect(args):
    """Inspect model forward methods."""
    try:
        import transformers.models as models_pkg
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Find GLM4v module
    import pkgutil
    for mod in pkgutil.iter_modules(models_pkg.__path__):
        if "glm" in mod.name:
            module_name = f"transformers.models.{mod.name}.modeling_{mod.name}"
            try:
                module = importlib.import_module(module_name)
                for cls_name in ["Glm4vForConditionalGeneration", "Glm4vModel", "Glm4vVisionModel"]:
                    if hasattr(module, cls_name):
                        cls = getattr(module, cls_name)
                        if hasattr(cls, "forward"):
                            print("=" * 80)
                            print(f"{cls_name}.forward")
                            print(f"File: {inspect.getsourcefile(cls)}")
                            print("-" * 80)
                            print(inspect.getsource(cls.forward))
            except Exception:
                continue

    return 0


# =============================================================================
# REPL Mode for Interactive Debugging
# =============================================================================

# Pipeline stages for REPL
PIPELINE_STAGES = [
    ('preprocess', 'Normalized pixels', (448, 448, 3)),
    ('patch_embed', 'After conv3d', (1024, 1536)),
    ('post_conv_ln', 'After layernorm', (1024, 1536)),
    ('pos_embed', 'After position embedding', (1024, 1536)),
    ('block_0', 'After transformer block 0', (1024, 1536)),
    ('block_11', 'After transformer block 11', (1024, 1536)),
    ('block_23', 'After transformer block 23', (1024, 1536)),
    ('post_ln', 'After final layernorm', (1024, 1536)),
    ('downsample', 'After spatial merge', (256, 4096)),
    ('merger', 'Final embeddings', (256, 4096)),
]


class REPLState:
    """State container for REPL session."""
    def __init__(self):
        self.image_path = None
        self.pixels = None           # Preprocessed image [H, W, C]
        self.current_stage = None
        self.py_tensors = {}         # {stage_name: np.array}
        self.cpp_tensors = {}        # {stage_name: np.array}
        self.model = None            # Lazy-loaded HuggingFace model
        self.size = 448


def run_python_to_stage(model, pixels, stage_name):
    """Run Python vision encoder up to specified stage and return tensor."""
    import torch
    import torch.nn.functional as F

    visual = model.model.visual
    size = pixels.shape[0]
    grid = size // 14

    # Prepare input: [H, W, C] -> [1, C, T, H, W] with T=2
    pixels_chw = pixels.transpose(2, 0, 1)  # [C, H, W]
    pixels_cthw = np.stack([pixels_chw, pixels_chw], axis=1)  # [C, T, H, W]
    pixel_values = torch.from_numpy(pixels_cthw).unsqueeze(0).to(torch.bfloat16).contiguous()

    with torch.no_grad():
        if stage_name == 'preprocess':
            return pixels

        # Stage 1: patch_embed (conv3d)
        patch_embed = visual.patch_embed
        weight = patch_embed.proj.weight
        bias = patch_embed.proj.bias
        x = pixel_values.view(-1, 3, 2, 14, 14)
        x = F.conv3d(x, weight, bias, stride=(2, 14, 14))
        x = x.view(-1, 1536)

        if stage_name == 'patch_embed':
            return x.float().numpy()

        # Stage 2: post_conv_layernorm
        x_ln = visual.post_conv_layernorm(x)
        if stage_name == 'post_conv_ln':
            return x_ln.float().numpy()

        # For stages beyond this, use get_image_features
        # which runs the full pipeline
        grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long)

        if stage_name == 'merger':
            final_embeds = model.get_image_features(pixel_values, grid_thw)
            return final_embeds[0].float().numpy()

        # For intermediate stages, we'd need to hook into the model
        # For now, return the closest available stage
        print(f"  Note: Stage '{stage_name}' requires model hooks, using closest available")
        if stage_name in ['pos_embed', 'block_0', 'block_11', 'block_23', 'post_ln', 'downsample']:
            final_embeds = model.get_image_features(pixel_values, grid_thw)
            return final_embeds[0].float().numpy()

    return None


def repl_load(state, args):
    """Load and preprocess an image."""
    if not args:
        print("Usage: load <image_path>")
        return

    image_path = args[0]
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    state.image_path = image_path
    state.pixels = preprocess_image(image_path, state.size)
    state.py_tensors = {'preprocess': state.pixels}
    state.cpp_tensors = {}
    state.current_stage = 'preprocess'

    print(f"Loaded: shape={state.pixels.shape} mean={state.pixels.mean():.6f}")


def repl_stages(state):
    """List all pipeline stages."""
    print("\nPipeline Stages:")
    for i, (name, desc, shape) in enumerate(PIPELINE_STAGES):
        marker = " *" if name == state.current_stage else ""
        print(f"  {i}: {name:12} {str(shape):20} - {desc}{marker}")
    print()


def repl_python(state, args):
    """Run Python vision encoder to specified stage."""
    if state.pixels is None:
        print("No image loaded. Use 'load <image>' first.")
        return

    stage_name = args[0] if args else state.current_stage or 'patch_embed'

    # Validate stage name
    valid_stages = [s[0] for s in PIPELINE_STAGES]
    if stage_name not in valid_stages:
        print(f"Unknown stage: {stage_name}")
        print(f"Valid stages: {', '.join(valid_stages)}")
        return

    if state.model is None:
        state.model, _ = get_model()

    print(f"Running Python to {stage_name}...")
    tensor = run_python_to_stage(state.model, state.pixels, stage_name)

    if tensor is not None:
        state.py_tensors[stage_name] = tensor
        state.current_stage = stage_name
        print(f"  Python {stage_name}: shape={tensor.shape}")
        print(f"    mean={tensor.mean():.6f} std={tensor.std():.6f}")
        print(f"    min={tensor.min():.6f} max={tensor.max():.6f}")


def repl_cpp(state, args):
    """Run C++ vision encoder to specified stage."""
    if state.pixels is None:
        print("No image loaded. Use 'load <image>' first.")
        return

    stage_name = args[0] if args else state.current_stage or 'patch_embed'

    # For now, we can only get final embeddings from C++
    # Full stage-stop support requires C++ changes
    if stage_name not in ['preprocess', 'merger']:
        print(f"  Note: C++ stage-stop not yet implemented for '{stage_name}'")
        print(f"  Available: 'preprocess', 'merger'")
        return

    if stage_name == 'preprocess':
        # Just use the same preprocessed pixels
        state.cpp_tensors[stage_name] = state.pixels.copy()
        print(f"  C++ {stage_name}: shape={state.pixels.shape}")
        print(f"    mean={state.pixels.mean():.6f}")
        return

    # For merger, we'd need to run C++ with the image
    # This requires the precompute mode
    print(f"  C++ {stage_name}: Not yet implemented")
    print(f"  Use 'compare-pipeline' command for full comparison")


def repl_step(state, args):
    """Run both Python and C++ to specified stage."""
    stage_name = args[0] if args else None

    if stage_name is None:
        # Get next stage
        valid_stages = [s[0] for s in PIPELINE_STAGES]
        if state.current_stage is None:
            stage_name = 'patch_embed'
        else:
            try:
                idx = valid_stages.index(state.current_stage)
                if idx + 1 < len(valid_stages):
                    stage_name = valid_stages[idx + 1]
                else:
                    print("Already at last stage")
                    return
            except ValueError:
                stage_name = 'patch_embed'

    print(f"\n=== Stepping to {stage_name} ===")
    repl_python(state, [stage_name])
    repl_cpp(state, [stage_name])


def repl_compare(state):
    """Compare Python and C++ at current stage."""
    stage = state.current_stage

    if stage is None:
        print("No stage selected. Use 'step' or 'py <stage>' first.")
        return

    py = state.py_tensors.get(stage)
    cpp = state.cpp_tensors.get(stage)

    if py is None:
        print(f"No Python tensor for stage '{stage}'. Run 'py {stage}' first.")
        return
    if cpp is None:
        print(f"No C++ tensor for stage '{stage}'. Run 'cpp {stage}' first.")
        return

    py_flat = py.flatten()
    cpp_flat = cpp.flatten()

    if py_flat.shape != cpp_flat.shape:
        print(f"Shape mismatch: Python {py.shape} vs C++ {cpp.shape}")
        return

    diff = np.abs(py_flat - cpp_flat)
    corr = np.corrcoef(py_flat, cpp_flat)[0, 1]

    print(f"\nComparison at {stage}:")
    print(f"  Python: mean={py.mean():.6f} std={py.std():.6f}")
    print(f"  C++:    mean={cpp.mean():.6f} std={cpp.std():.6f}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Max diff:    {diff.max():.6f}")
    print(f"  Mean diff:   {diff.mean():.6f}")

    if corr > 0.99:
        print("  Status: PASS")
    elif corr > 0.90:
        print("  Status: WARN - minor differences")
    else:
        print("  Status: FAIL - significant divergence")


def repl_show(state, args):
    """Show tensor values/stats."""
    if not args:
        print("Usage: show py|cpp [n_elements]")
        return

    which = args[0]
    n = int(args[1]) if len(args) > 1 else 10
    stage = state.current_stage

    if which == 'py':
        tensor = state.py_tensors.get(stage)
        label = "Python"
    elif which == 'cpp':
        tensor = state.cpp_tensors.get(stage)
        label = "C++"
    else:
        print("Usage: show py|cpp [n_elements]")
        return

    if tensor is None:
        print(f"No {label} tensor for stage '{stage}'")
        return

    print(f"\n{label} tensor at {stage}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Mean: {tensor.mean():.6f}")
    print(f"  Std: {tensor.std():.6f}")
    print(f"  Min: {tensor.min():.6f}")
    print(f"  Max: {tensor.max():.6f}")
    print(f"\n  First {n} elements: {tensor.flatten()[:n]}")


def repl_diff(state):
    """Show element-wise differences."""
    stage = state.current_stage
    py = state.py_tensors.get(stage)
    cpp = state.cpp_tensors.get(stage)

    if py is None or cpp is None:
        print("Need both Python and C++ tensors. Run 'step' first.")
        return

    diff = np.abs(py.flatten() - cpp.flatten())

    # Find largest differences
    indices = np.argsort(diff)[::-1][:10]

    print(f"\nTop 10 differences at {stage}:")
    for i, idx in enumerate(indices):
        print(f"  [{idx}]: py={py.flatten()[idx]:.6f} cpp={cpp.flatten()[idx]:.6f} diff={diff[idx]:.6f}")


def repl_dump(state, args):
    """Save current tensors to files."""
    name = args[0] if args else "repl_dump"
    stage = state.current_stage or "unknown"

    py = state.py_tensors.get(stage)
    cpp = state.cpp_tensors.get(stage)

    if py is not None:
        path = f"{name}_py_{stage}.npy"
        np.save(path, py)
        print(f"Saved Python tensor to {path}")

    if cpp is not None:
        path = f"{name}_cpp_{stage}.npy"
        np.save(path, cpp)
        print(f"Saved C++ tensor to {path}")


def repl_help():
    """Show REPL help."""
    print("""
GLM Vision Pipeline REPL Commands:

  load <image>      Load and preprocess an image
  stages            List all pipeline stages

  step [stage]      Run both Python and C++ to stage (default: next)
  py <stage>        Run Python only to stage
  cpp <stage>       Run C++ only to stage

  compare           Compare Python and C++ at current stage
  show py|cpp [n]   Show tensor values (first n elements)
  diff              Show element-wise differences

  dump [name]       Save tensors to .npy files
  help              Show this help
  quit              Exit REPL

Stages: preprocess, patch_embed, post_conv_ln, pos_embed,
        block_0, block_11, block_23, post_ln, downsample, merger
""")


def cmd_repl(args):
    """Interactive REPL for debugging vision pipeline."""
    try:
        import readline  # For command history
    except ImportError:
        pass  # readline not available on all platforms

    state = REPLState()
    state.size = args.size

    print("GLM Vision Pipeline REPL")
    print("Type 'help' for commands, 'quit' to exit")
    print()

    # Auto-load image if provided
    if args.image:
        repl_load(state, [args.image])

    while True:
        try:
            line = input("glm> ").strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0]
            cmd_args = parts[1:]

            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'load':
                repl_load(state, cmd_args)
            elif cmd == 'stages':
                repl_stages(state)
            elif cmd == 'step':
                repl_step(state, cmd_args)
            elif cmd == 'py':
                repl_python(state, cmd_args)
            elif cmd == 'cpp':
                repl_cpp(state, cmd_args)
            elif cmd == 'compare':
                repl_compare(state)
            elif cmd == 'show':
                repl_show(state, cmd_args)
            elif cmd == 'diff':
                repl_diff(state)
            elif cmd == 'dump':
                repl_dump(state, cmd_args)
            elif cmd == 'help':
                repl_help()
            else:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")

        except KeyboardInterrupt:
            print()
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")
    return 0


# =============================================================================
# Python Server Mode (for persistent subprocess)
# =============================================================================

def cmd_serve(args):
    """Run as a server that accepts commands via stdin and returns JSON.

    This keeps the model loaded and allows fast queries without reloading.
    Protocol: One JSON command per line, one JSON response per line.
    """
    import json
    import torch
    import itertools

    print("LOADING_MODEL", flush=True)
    model, processor = get_model()
    tokenizer = get_tokenizer()
    print("READY", flush=True)

    while True:
        try:
            line = input()
            if not line.strip():
                continue

            cmd = json.loads(line)
            action = cmd.get('action')
            result = {'status': 'ok'}

            if action == 'ping':
                result['response'] = 'pong'

            elif action == 'quit':
                print(json.dumps({'status': 'ok', 'response': 'goodbye'}), flush=True)
                break

            elif action == 'get_rope_index':
                # Compute MROPE position IDs for given input
                input_ids_list = cmd['input_ids']  # List of token IDs
                image_grid_thw = cmd.get('image_grid_thw')  # [[t, h, w], ...]
                video_grid_thw = cmd.get('video_grid_thw')

                input_ids = torch.tensor([input_ids_list], dtype=torch.long)

                img_grid = None
                if image_grid_thw:
                    img_grid = torch.tensor(image_grid_thw, dtype=torch.long)

                vid_grid = None
                if video_grid_thw:
                    vid_grid = torch.tensor(video_grid_thw, dtype=torch.long)

                position_ids, rope_deltas = model.model.get_rope_index(
                    input_ids, img_grid, vid_grid, attention_mask=None
                )

                # [3, B, T] -> [3, T] for batch 0
                pos_ids_np = position_ids[:, 0, :].numpy().tolist()

                result['position_ids'] = pos_ids_np  # [3][T] - [t_ids, h_ids, w_ids]
                result['rope_deltas'] = rope_deltas[0].item()
                result['shape'] = list(position_ids.shape)

            elif action == 'get_image_features':
                # Run vision encoder on preprocessed pixels
                pixels_flat = cmd['pixels']  # Flat list of floats
                size = cmd.get('size', 448)

                pixels = np.array(pixels_flat, dtype=np.float32).reshape(size, size, 3)
                pixels_chw = pixels.transpose(2, 0, 1)
                pixels_cthw = np.stack([pixels_chw, pixels_chw], axis=1)
                pixel_values = torch.from_numpy(pixels_cthw).unsqueeze(0).to(torch.bfloat16).contiguous()

                grid = size // 14
                grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long)

                with torch.no_grad():
                    embeds = model.get_image_features(pixel_values, grid_thw)[0].float().numpy()

                result['embeds'] = embeds.flatten().tolist()
                result['shape'] = list(embeds.shape)
                result['mean'] = float(embeds.mean())
                result['std'] = float(embeds.std())

            elif action == 'get_embeddings_and_positions':
                # Full multimodal forward - get embeddings and position IDs
                image_path = cmd['image']
                prompt = cmd.get('prompt', 'Describe this image.')
                size = cmd.get('size', 448)

                # Manually preprocess image (same as C++)
                pixels = preprocess_image(image_path, size)
                pixels_chw = pixels.transpose(2, 0, 1)
                pixels_cthw = np.stack([pixels_chw, pixels_chw], axis=1)
                pixel_values = torch.from_numpy(pixels_cthw).unsqueeze(0).to(torch.bfloat16).contiguous()

                grid = size // 14
                image_grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.long)

                # Get image features
                with torch.no_grad():
                    image_embeds = model.get_image_features(pixel_values, image_grid_thw)
                    image_embeds = torch.cat(image_embeds, dim=0)

                # Number of image tokens after spatial merge
                num_image_tokens = (grid // 2) * (grid // 2)  # 16x16 = 256 for 448px

                # Build input_ids to match C++ structure EXACTLY:
                # [gMASK]<sop><|user|>\n<|begin_of_image|><|image|>x256<|end_of_image|>prompt<|assistant|>\n
                image_token_id = model.config.image_token_id

                # Special token IDs (must match C++)
                gmask_id = 151331
                sop_id = 151333
                user_id = 151336
                newline_id = 198
                begin_image_id = 151339
                end_image_id = 151340
                assistant_id = 151337

                # Tokenize prompt text
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

                # Build sequence matching C++ exactly
                input_ids_list = (
                    [gmask_id, sop_id, user_id, newline_id, begin_image_id] +
                    [image_token_id] * num_image_tokens +
                    [end_image_id] +
                    prompt_tokens +
                    [assistant_id, newline_id]
                )
                input_ids = torch.tensor([input_ids_list], dtype=torch.long)

                # Get position IDs
                position_ids, rope_deltas = model.model.get_rope_index(
                    input_ids, image_grid_thw, None, attention_mask=None
                )

                # Get input embeddings before image insertion
                text_embeds = model.model.get_input_embeddings()(input_ids)

                # Find image token positions
                image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()

                result['input_ids'] = input_ids[0].tolist()
                result['position_ids'] = position_ids[:, 0, :].numpy().tolist()  # [3, T]
                result['rope_deltas'] = rope_deltas[0].item()
                result['image_token_positions'] = image_positions
                result['num_image_tokens'] = len(image_positions)
                result['image_grid_thw'] = image_grid_thw[0].tolist()
                result['text_embeds_shape'] = list(text_embeds.shape)
                result['image_embeds_shape'] = list(image_embeds.shape)
                result['image_embeds_mean'] = float(image_embeds.float().mean())
                result['image_embeds_std'] = float(image_embeds.float().std())
                # First 10 image embedding values
                result['image_embeds_first10'] = image_embeds.float().flatten()[:10].tolist()

            elif action == 'tokenize':
                # Tokenize text and return token IDs
                text = cmd['text']
                tokens = tokenizer.encode(text, add_special_tokens=False)
                result['tokens'] = tokens
                result['decoded'] = tokenizer.decode(tokens)

            elif action == 'decode':
                # Decode token IDs to text
                tokens = cmd['tokens']
                text = tokenizer.decode(tokens)
                result['text'] = text

            elif action == 'get_special_tokens':
                result['image_token_id'] = model.config.image_token_id
                result['video_start_token_id'] = model.config.video_start_token_id
                result['video_end_token_id'] = model.config.video_end_token_id
                result['bos_token_id'] = tokenizer.bos_token_id
                result['eos_token_id'] = tokenizer.eos_token_id

            else:
                result = {'status': 'error', 'message': f'Unknown action: {action}'}

            print(json.dumps(result), flush=True)

        except EOFError:
            break
        except json.JSONDecodeError as e:
            print(json.dumps({'status': 'error', 'message': f'JSON error: {e}'}), flush=True)
        except Exception as e:
            import traceback
            print(json.dumps({'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()}), flush=True)

    return 0


class PythonServer:
    """Manages a persistent Python subprocess for fast queries."""

    def __init__(self):
        self.process = None

    def start(self):
        """Start the Python server subprocess."""
        if self.process is not None:
            return

        print("Starting Python server (loading model)...")
        self.process = subprocess.Popen(
            ['python3.13', str(PROJECT_DIR / 'glm_tools.py'), 'serve'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_DIR)
        )

        # Wait for READY signal
        while True:
            line = self.process.stdout.readline().strip()
            print(f"  Server: {line}")
            if line == 'READY':
                break
            if line == '' and self.process.poll() is not None:
                raise RuntimeError("Server process died during startup")

        print("Python server ready!")

    def query(self, cmd: dict) -> dict:
        """Send a command and get response."""
        import json
        if self.process is None:
            self.start()

        self.process.stdin.write(json.dumps(cmd) + '\n')
        self.process.stdin.flush()

        response = self.process.stdout.readline()
        return json.loads(response)

    def stop(self):
        """Stop the server."""
        if self.process is not None:
            try:
                self.query({'action': 'quit'})
            except:
                pass
            self.process.terminate()
            self.process.wait()
            self.process = None


# =============================================================================
# Debug Multimodal REPL
# =============================================================================

def cmd_debug_mm(args):
    """Interactive REPL for debugging multimodal mixing (position IDs, embedding insertion)."""
    try:
        import readline
    except ImportError:
        pass

    server = PythonServer()
    state = {
        'image_path': args.image,
        'size': args.size,
        'py_data': None,
        'cpp_data': None,
    }

    print("=" * 60)
    print("Multimodal Debug REPL")
    print("=" * 60)
    print("Commands: load, py, cpp, compare, show, tokens, help, quit")
    print()

    # Auto-load image if provided
    if args.image:
        debug_mm_load(server, state, args.image)

    while True:
        try:
            line = input("mm> ").strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0]
            cmd_args = parts[1:]

            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'load':
                if cmd_args:
                    debug_mm_load(server, state, cmd_args[0])
                else:
                    print("Usage: load <image_path>")
            elif cmd == 'py':
                debug_mm_python(server, state)
            elif cmd == 'cpp':
                debug_mm_cpp(state)
            elif cmd == 'compare':
                debug_mm_compare(state)
            elif cmd == 'show':
                what = cmd_args[0] if cmd_args else 'both'
                debug_mm_show(state, what)
            elif cmd == 'tokens':
                debug_mm_tokens(state)
            elif cmd == 'posids':
                debug_mm_posids(state, cmd_args)
            elif cmd == 'special':
                debug_mm_special(server)
            elif cmd == 'help':
                debug_mm_help()
            else:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")

        except KeyboardInterrupt:
            print()
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

    server.stop()
    print("Goodbye!")
    return 0


def debug_mm_load(server, state, image_path):
    """Load an image for debugging."""
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    state['image_path'] = image_path
    state['py_data'] = None
    state['cpp_data'] = None

    print(f"Image set: {image_path}")
    print("Use 'py' to run Python, 'cpp' to run C++, 'compare' to compare")


def debug_mm_python(server, state):
    """Run Python multimodal forward and get position IDs, embeddings."""
    if not state['image_path']:
        print("No image loaded. Use 'load <image>' first.")
        return

    print("Running Python multimodal forward...")
    result = server.query({
        'action': 'get_embeddings_and_positions',
        'image': state['image_path'],
        'prompt': 'What colors form this image?',
        'size': state['size']
    })

    if result.get('status') != 'ok':
        print(f"Error: {result.get('message')}")
        return

    state['py_data'] = result

    print(f"\nPython Results:")
    print(f"  Input IDs length: {len(result['input_ids'])}")
    print(f"  Image token positions: {result['image_token_positions'][:5]}... ({result['num_image_tokens']} total)")
    print(f"  Image grid THW: {result['image_grid_thw']}")
    print(f"  Rope deltas: {result['rope_deltas']}")
    print(f"  Image embeds shape: {result['image_embeds_shape']}")
    print(f"  Image embeds mean: {result['image_embeds_mean']:.6f}")
    print(f"  Image embeds std: {result['image_embeds_std']:.6f}")


def debug_mm_cpp(state):
    """Load C++ multimodal debug output."""
    print("Running C++ multimodal forward with debug...")

    # Run C++ with debug-vision which dumps position IDs
    env = os.environ.copy()
    env['GLM_WEIGHTS_DIR'] = str(WEIGHTS_DIR)
    env['GLM_DEBUG_VISION'] = '1'
    env['GLM_GENERATE'] = '1'
    env['GLM_SINGLE_IMAGE'] = '1'
    env['GLM_IMAGES_DIR'] = str(Path(state['image_path']).parent)
    env['GLM_IMAGE_IDX'] = '0'  # Assuming first image
    env['GLM_MAX_TOKENS'] = '1'  # Just need the position IDs

    result = subprocess.run(
        ['./glm46v_mlx'],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True
    )

    # Parse the debug output for position IDs
    output = result.stdout
    state['cpp_output'] = output

    # Try to extract position IDs from debug output
    cpp_data = {'raw_output': output}

    # Look for position_ids line
    for line in output.split('\n'):
        if 'position_ids:' in line:
            cpp_data['position_ids_line'] = line
        if 'inputs_embeds:' in line:
            cpp_data['inputs_embeds_line'] = line
        if 'image_embeds' in line.lower():
            cpp_data['image_embeds_line'] = line

    state['cpp_data'] = cpp_data

    print(f"\nC++ Debug Output (last 30 lines):")
    lines = output.strip().split('\n')
    for line in lines[-30:]:
        print(f"  {line}")


def debug_mm_compare(state):
    """Compare Python and C++ multimodal mixing."""
    py = state.get('py_data')
    cpp = state.get('cpp_data')

    if py is None:
        print("Run 'py' first to get Python results")
        return
    if cpp is None:
        print("Run 'cpp' first to get C++ results")
        return

    print("\n=== Comparison ===")
    print(f"\nPython:")
    print(f"  Sequence length: {len(py['input_ids'])}")
    print(f"  Image tokens: {py['num_image_tokens']}")
    print(f"  Grid THW: {py['image_grid_thw']}")

    # Show position IDs for image region
    pos_ids = py['position_ids']
    img_start = py['image_token_positions'][0]
    img_end = py['image_token_positions'][-1] + 1

    print(f"\n  Position IDs at image region [{img_start}:{img_end}]:")
    print(f"    t: {pos_ids[0][img_start:min(img_start+10, img_end)]}...")
    print(f"    h: {pos_ids[1][img_start:min(img_start+10, img_end)]}...")
    print(f"    w: {pos_ids[2][img_start:min(img_start+10, img_end)]}...")

    print(f"\nC++:")
    if 'position_ids_line' in cpp:
        print(f"  {cpp['position_ids_line']}")

    print("\nNote: For full comparison, add GLM_DUMP_POSIDS=1 support to C++")


def debug_mm_show(state, what):
    """Show detailed data."""
    if what in ('py', 'python', 'both'):
        py = state.get('py_data')
        if py:
            print("\n=== Python Data ===")
            print(f"Input IDs ({len(py['input_ids'])} tokens):")
            print(f"  First 20: {py['input_ids'][:20]}")
            print(f"  Last 10: {py['input_ids'][-10:]}")
            print(f"\nPosition IDs shape: {len(py['position_ids'])} x {len(py['position_ids'][0])}")
            print(f"  t first 20: {py['position_ids'][0][:20]}")
            print(f"  h first 20: {py['position_ids'][1][:20]}")
            print(f"  w first 20: {py['position_ids'][2][:20]}")
            print(f"\nImage embeds first 10: {py['image_embeds_first10']}")
        else:
            print("No Python data. Run 'py' first.")

    if what in ('cpp', 'both'):
        cpp = state.get('cpp_data')
        if cpp:
            print("\n=== C++ Data ===")
            print(cpp.get('raw_output', 'No output')[:2000])
        else:
            print("No C++ data. Run 'cpp' first.")


def debug_mm_tokens(state):
    """Show token analysis."""
    py = state.get('py_data')
    if not py:
        print("Run 'py' first")
        return

    input_ids = py['input_ids']
    img_positions = py['image_token_positions']

    print(f"\nToken Analysis:")
    print(f"  Total tokens: {len(input_ids)}")
    print(f"  Image tokens: {len(img_positions)}")
    print(f"  Image region: [{img_positions[0]}, {img_positions[-1]}]")
    print(f"  Text before image: {img_positions[0]} tokens")
    print(f"  Text after image: {len(input_ids) - img_positions[-1] - 1} tokens")

    # Expected image tokens: t * h * w / merge^2
    grid = py['image_grid_thw']
    expected = grid[0] * (grid[1] // 2) * (grid[2] // 2)
    print(f"\n  Expected image tokens: {grid[0]} * ({grid[1]}//2) * ({grid[2]}//2) = {expected}")
    print(f"  Actual image tokens: {len(img_positions)}")
    if expected != len(img_positions):
        print(f"  *** MISMATCH! ***")


def debug_mm_posids(state, args):
    """Show position IDs at specific indices."""
    py = state.get('py_data')
    if not py:
        print("Run 'py' first")
        return

    if not args:
        # Show position IDs around image boundary
        img_start = py['image_token_positions'][0]
        indices = list(range(max(0, img_start-3), min(img_start+13, len(py['position_ids'][0]))))
    else:
        indices = [int(x) for x in args]

    print(f"\nPosition IDs at indices {indices}:")
    print(f"{'idx':>5} {'t':>5} {'h':>5} {'w':>5}  {'note':>10}")
    print("-" * 35)

    img_positions = set(py['image_token_positions'])
    for idx in indices:
        t = py['position_ids'][0][idx]
        h = py['position_ids'][1][idx]
        w = py['position_ids'][2][idx]
        note = "IMAGE" if idx in img_positions else "text"
        print(f"{idx:>5} {t:>5} {h:>5} {w:>5}  {note:>10}")


def debug_mm_special(server):
    """Show special token IDs."""
    result = server.query({'action': 'get_special_tokens'})
    print("\nSpecial Tokens:")
    for k, v in result.items():
        if k != 'status':
            print(f"  {k}: {v}")


def debug_mm_help():
    """Show help."""
    print("""
Multimodal Debug REPL Commands:

  load <image>    Load an image for debugging
  py              Run Python multimodal forward (get position IDs, embeddings)
  cpp             Run C++ multimodal forward with debug output
  compare         Compare Python vs C++ results
  show [py|cpp]   Show detailed data
  tokens          Analyze token structure
  posids [idx...] Show position IDs at indices (default: around image start)
  special         Show special token IDs
  help            Show this help
  quit            Exit REPL

Workflow:
  1. load test_images_colors/blue.jpg
  2. py                    # Run Python, see position IDs
  3. cpp                   # Run C++, see debug output
  4. compare               # Compare the two
  5. posids 5 6 7 8 9 10   # Inspect specific position IDs
""")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='glm_tools',
        description='GLM-4.6V unified tooling'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Build command
    p_build = subparsers.add_parser('build', help='Build C++ binary')
    p_build.add_argument('-v', '--verbose', action='store_true')

    # Run command
    p_run = subparsers.add_parser('run', help='Run C++ binary')
    p_run.add_argument('--generate', action='store_true', help='Enable generation mode')
    p_run.add_argument('--single-image', action='store_true', help='Single image mode')
    p_run.add_argument('--precompute', action='store_true', help='Precompute embeddings')
    p_run.add_argument('--embeds-dir', type=str, help='Embeddings directory')
    p_run.add_argument('--images-dir', type=str, help='Images directory')
    p_run.add_argument('--image-idx', type=int, help='Image index')
    p_run.add_argument('--max-tokens', type=int, default=20, help='Max tokens')
    p_run.add_argument('--do-sample', type=lambda x: x.lower() == 'true', help='Sampling mode')
    p_run.add_argument('--weights-dir', type=str, help='Weights directory')
    p_run.add_argument('--debug-vision', action='store_true', help='Debug vision output')
    p_run.add_argument('--stream', action='store_true', help='Stream output')

    # Generate-cpp command
    p_gen_cpp = subparsers.add_parser('generate-cpp', help='Run C++ generation')
    p_gen_cpp.add_argument('--embeds-dir', type=str, required=True, help='Embeddings directory')
    p_gen_cpp.add_argument('--image-idx', type=int, default=0, help='Image index')
    p_gen_cpp.add_argument('--max-tokens', type=int, default=20, help='Max tokens')
    p_gen_cpp.add_argument('--do-sample', type=lambda x: x.lower() == 'true', help='Sampling mode')
    p_gen_cpp.add_argument('--weights-dir', type=str, help='Weights directory')
    p_gen_cpp.add_argument('--debug-vision', action='store_true', help='Debug vision output')
    p_gen_cpp.add_argument('--stream', action='store_true', help='Stream output')

    # Precompute command
    p_precompute = subparsers.add_parser('precompute', help='Precompute embeddings')
    p_precompute.add_argument('--images-dir', type=str, required=True, help='Images directory')
    p_precompute.add_argument('--output-dir', type=str, required=True, help='Output directory')
    p_precompute.add_argument('--weights-dir', type=str, help='Weights directory')
    p_precompute.add_argument('--debug-vision', action='store_true', help='Debug vision output')

    # Export weights command
    p_export_weights = subparsers.add_parser('export-weights', help='Export model weights')
    p_export_weights.add_argument('--vision', action='store_true', help='Export vision weights')
    p_export_weights.add_argument('--text', action='store_true', help='Export text weights')
    p_export_weights.add_argument('--output-dir', type=str, default=str(WEIGHTS_DIR))
    p_export_weights.add_argument('-v', '--verbose', action='store_true')

    # Export prompt command
    p_export_prompt = subparsers.add_parser('export-prompt', help='Export tokenized prompt')
    p_export_prompt.add_argument('--size', type=int, default=448, help='Image size')
    p_export_prompt.add_argument('--prompt', type=str, required=True, help='Prompt text')
    p_export_prompt.add_argument('--output-dir', type=str, default=str(WEIGHTS_DIR))

    # Compare pipeline command
    p_cmp_pipeline = subparsers.add_parser('compare-pipeline', help='Compare vision pipeline')
    p_cmp_pipeline.add_argument('--image', type=str, required=True, help='Image path')
    p_cmp_pipeline.add_argument('--size', type=int, default=448, help='Image size')
    p_cmp_pipeline.add_argument('--cpp-embeds', type=str, help='C++ embeddings file')
    p_cmp_pipeline.add_argument('--image-idx', type=int, default=0, help='Image index in embeddings')

    # Compare embeds command
    p_cmp_embeds = subparsers.add_parser('compare-embeds', help='Compare embeddings')
    p_cmp_embeds.add_argument('--cpp-embeds', type=str, required=True, help='C++ embeddings file')
    p_cmp_embeds.add_argument('--image', type=str, help='Image for Python comparison')
    p_cmp_embeds.add_argument('--image-idx', type=int, default=0, help='Image index')
    p_cmp_embeds.add_argument('--size', type=int, default=448, help='Image size')

    # Compare preproc command
    p_cmp_preproc = subparsers.add_parser('compare-preproc', help='Compare preprocessing')
    p_cmp_preproc.add_argument('--image', type=str, required=True, help='Image path')
    p_cmp_preproc.add_argument('--size', type=int, default=448, help='Image size')
    p_cmp_preproc.add_argument('--cpp-preproc', type=str, help='C++ preprocessed file')
    p_cmp_preproc.add_argument('--output', type=str, help='Output path for Python result')

    # Generate command
    p_generate = subparsers.add_parser('generate', help='Run Python generation')
    p_generate.add_argument('--images', nargs='+', required=True, help='Image paths')
    p_generate.add_argument('--prompt', type=str, default="Describe this image.", help='Prompt')
    p_generate.add_argument('--max-tokens', type=int, default=50, help='Max tokens')
    p_generate.add_argument('--do-sample', action='store_true', help='Use sampling')

    # Test text command
    p_test_text = subparsers.add_parser('test-text', help='Test text parity')
    p_test_text.add_argument('--prompts', nargs='+', help='Prompts to test')
    p_test_text.add_argument('--max-tokens', type=int, default=20, help='Max tokens')

    # Test full command
    p_test_full = subparsers.add_parser('test-full', help='Run full test suite')
    p_test_full.add_argument('--images', nargs='+', help='Images to test')
    p_test_full.add_argument('--cpp-embeds', type=str, help='C++ embeddings file')

    # Inspect command
    subparsers.add_parser('inspect', help='Inspect model forward methods')

    # REPL command
    p_repl = subparsers.add_parser('repl', help='Interactive debugging REPL')
    p_repl.add_argument('--image', type=str, help='Image to load on startup')
    p_repl.add_argument('--size', type=int, default=448, help='Image size')

    # Server mode (for persistent subprocess)
    subparsers.add_parser('serve', help='Run as server (internal use)')

    # Debug multimodal REPL
    p_debug_mm = subparsers.add_parser('debug-mm', help='Debug multimodal mixing')
    p_debug_mm.add_argument('--image', type=str, help='Image to load on startup')
    p_debug_mm.add_argument('--size', type=int, default=448, help='Image size')

    args = parser.parse_args()

    # Route to command handler
    handlers = {
        'build': cmd_build,
        'run': cmd_run,
        'generate-cpp': cmd_generate_cpp,
        'precompute': cmd_precompute,
        'export-weights': cmd_export_weights,
        'export-prompt': cmd_export_prompt,
        'compare-pipeline': cmd_compare_pipeline,
        'compare-embeds': cmd_compare_embeds,
        'compare-preproc': cmd_compare_preproc,
        'generate': cmd_generate,
        'test-text': cmd_test_text,
        'test-full': cmd_test_full,
        'inspect': cmd_inspect,
        'repl': cmd_repl,
        'serve': cmd_serve,
        'debug-mm': cmd_debug_mm,
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
