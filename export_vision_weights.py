#!/usr/bin/env python3
"""
Export GLM-4.6V weights (vision and/or text) to raw float16 binary for C++ loading.

Usage:
    python export_vision_weights.py [--model MODEL_PATH] [--output OUTPUT_DIR] [--vision] [--text]

Then verify in C++:
    GLM_VERIFY=1 GLM_WEIGHTS_DIR=vision_weights ./glm46v_mlx
"""

import torch
import numpy as np
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import json
import os
import argparse

MODEL_PATH = "zai-org/GLM-4.6V-Flash"


def export_vision_weights(output_dir: str = "vision_weights"):
    """Export vision encoder weights and test data."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Load in float16 to match C++ precision for fair comparison
    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    visual = model.model.visual
    weights = {}

    # Patch embedding conv3d
    # PyTorch: [out, in, kD, kH, kW] -> MLX: [out, kD, kH, kW, in]
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

    return model, processor, output_dir, weights


def export_weights_raw(weights: dict, output_dir: str, visual):
    """Export weights as raw float16 binary in fixed order matching C++ loader."""
    weights_path = os.path.join(output_dir, "vision_encoder.bin")
    print(f"\nSaving weights to {weights_path}...")

    total_bytes = 0

    with open(weights_path, 'wb') as f:
        def write(name, tensor):
            nonlocal total_bytes
            arr = tensor.numpy().astype(np.float16)
            arr.tofile(f)
            total_bytes += arr.nbytes
            print(f"  {name}: {list(arr.shape)}")

        def write_zeros(name, shape):
            nonlocal total_bytes
            arr = np.zeros(shape, dtype=np.float16)
            arr.tofile(f)
            total_bytes += arr.nbytes
            print(f"  {name}: {list(shape)} (zeros)")

        write("patch_embed.weight", weights["patch_embed.weight"])
        write("patch_embed.bias", weights["patch_embed.bias"]) if "patch_embed.bias" in weights else write_zeros("patch_embed.bias", (1536,))
        write("pos_embed.weight", weights["pos_embed.weight"])
        write("post_conv_layernorm.weight", weights["post_conv_layernorm.weight"])

        for i in range(len(visual.blocks)):
            p = f"blocks.{i}."
            write(f"{p}norm1.weight", weights[f"{p}norm1.weight"])
            write(f"{p}norm2.weight", weights[f"{p}norm2.weight"])
            write(f"{p}attn.qkv.weight", weights[f"{p}attn.qkv.weight"])
            write(f"{p}attn.qkv.bias", weights[f"{p}attn.qkv.bias"]) if f"{p}attn.qkv.bias" in weights else write_zeros(f"{p}attn.qkv.bias", (4608,))
            write(f"{p}attn.proj.weight", weights[f"{p}attn.proj.weight"])
            write(f"{p}attn.proj.bias", weights[f"{p}attn.proj.bias"]) if f"{p}attn.proj.bias" in weights else write_zeros(f"{p}attn.proj.bias", (1536,))
            write(f"{p}mlp.gate_proj.weight", weights[f"{p}mlp.gate_proj.weight"])
            write(f"{p}mlp.up_proj.weight", weights[f"{p}mlp.up_proj.weight"])
            write(f"{p}mlp.down_proj.weight", weights[f"{p}mlp.down_proj.weight"])

        write("post_layernorm.weight", weights["post_layernorm.weight"])
        write("downsample.weight", weights["downsample.weight"])
        write("downsample.bias", weights["downsample.bias"]) if "downsample.bias" in weights else write_zeros("downsample.bias", (4096,))
        write("merger.proj.weight", weights["merger.proj.weight"])
        write("merger.proj.bias", weights["merger.proj.bias"]) if "merger.proj.bias" in weights else write_zeros("merger.proj.bias", (4096,))
        write("merger.post_ln.weight", weights["merger.post_ln.weight"])
        write("merger.post_ln.bias", weights["merger.post_ln.bias"])
        write("merger.gate_proj.weight", weights["merger.gate_proj.weight"])
        write("merger.up_proj.weight", weights["merger.up_proj.weight"])
        write("merger.down_proj.weight", weights["merger.down_proj.weight"])

    print(f"\nTotal: {total_bytes / 1024 / 1024:.1f} MB")


def create_test_data(model, output_dir: str):
    """Create test patches and expected output for verification."""
    print("\nCreating vision test data...")

    visual = model.model.visual
    visual.eval()

    with torch.no_grad():
        H, W = 336, 336
        np.random.seed(42)
        img = np.random.rand(H, W, 3).astype(np.float32)

        # Create patches: [1, 3, 2, 336, 336] -> [576, 1176]
        x = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
        x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [1, 3, 2, H, W]

        patch_size, temporal_patch = 14, 2
        grid_t, grid_h, grid_w = 1, H // patch_size, W // patch_size

        x = x.view(1, 3, grid_t, temporal_patch, grid_h, patch_size, grid_w, patch_size)
        x = x.permute(0, 2, 4, 6, 3, 5, 1, 7).contiguous()
        patches = x.view(-1, temporal_patch * patch_size * patch_size * 3)  # [576, 1176]

        # Save patches
        patches.numpy().astype(np.float32).tofile(os.path.join(output_dir, "test_patches.bin"))
        print(f"  Patches: {list(patches.shape)}")

        # Run forward and save output
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
        output = visual(patches, grid_thw)
        output.numpy().astype(np.float32).tofile(os.path.join(output_dir, "expected_output.bin"))
        print(f"  Output: {list(output.shape)}")


def create_text_test_data(model, output_dir: str):
    """Create test input_ids and expected output for text model verification."""
    print("\nCreating text test data...")

    model.eval()

    # Fix: Set rope_scaling on attention layers (not set by default in HF)
    # mrope_section must sum to 64 (half of head_dim=128), gets doubled internally to 128
    rope_scaling = {"mrope_section": [16, 24, 24]}
    for layer in model.model.language_model.layers:
        layer.self_attn.rope_scaling = rope_scaling

    with torch.no_grad():
        # Create deterministic test input_ids
        np.random.seed(123)
        seq_len = 32  # Short sequence for testing
        batch_size = 1

        # Use token IDs in valid range (avoid special tokens)
        input_ids = np.random.randint(1000, 10000, size=(batch_size, seq_len)).astype(np.int32)
        input_ids_tensor = torch.from_numpy(input_ids).long()

        # Save input_ids
        input_ids.tofile(os.path.join(output_dir, "test_input_ids.bin"))
        print(f"  Input IDs: {list(input_ids.shape)}")
        print(f"  First 5 input IDs: {input_ids[0, :5]}")

        # Step 1: Save embeddings (after embed_tokens lookup)
        text_model = model.model.language_model
        embeddings = text_model.embed_tokens(input_ids_tensor)
        embeddings.numpy().astype(np.float32).tofile(os.path.join(output_dir, "expected_embeddings.bin"))
        print(f"  Embeddings: {list(embeddings.shape)}")
        print(f"  Embeddings first 5 values: {embeddings[0, 0, :5].numpy()}")

        # Step 2: Save after first layer (manually run first layer)
        # Create position embeddings
        cache_position = torch.arange(seq_len)
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
        position_embeddings = text_model.rotary_emb(embeddings, position_ids)

        # Debug layer 0 step by step
        layer0 = text_model.layers[0]

        # 2a: Input layernorm
        x_norm = layer0.input_layernorm(embeddings)
        x_norm.numpy().astype(np.float32).tofile(os.path.join(output_dir, "debug_norm1.bin"))
        print(f"  2a. After input_layernorm: {list(x_norm.shape)}, first 5: {x_norm[0, 0, :5].numpy()}")

        # 2b: Q, K, V projections
        attn = layer0.self_attn
        q = attn.q_proj(x_norm)
        k = attn.k_proj(x_norm)
        v = attn.v_proj(x_norm)
        print(f"  2b. Q: {list(q.shape)}, first 5: {q[0, 0, :5].detach().numpy()}")
        print(f"      K: {list(k.shape)}, first 5: {k[0, 0, :5].detach().numpy()}")
        print(f"      V: {list(v.shape)}, first 5: {v[0, 0, :5].detach().numpy()}")

        # Debug: Q, K, V at token 13
        print(f"\n  Debug Q/K/V at token 13:")
        print(f"    x_norm @ token 13, first 5 dims: {x_norm[0, 13, :5].detach().numpy()}")
        print(f"    Q @ token 13, first 5 dims: {q[0, 13, :5].detach().numpy()}")
        print(f"    K @ token 13, first 5 dims: {k[0, 13, :5].detach().numpy()}")
        print(f"    V @ token 13, first 5 dims: {v[0, 13, :5].detach().numpy()}")
        # Debug: position embeddings at token 13
        # position_embeddings is (cos, sin) each of shape [3, B, T, rotary_dim]
        cos_emb, sin_emb = position_embeddings
        print(f"    pos_emb cos shape: {list(cos_emb.shape)}")
        # Section 0 at batch 0, token 13
        print(f"    pos_emb cos @ sec0, token 13, first 5 dims: {cos_emb[0, 0, 13, :5].detach().numpy()}")
        print(f"    pos_emb sin @ sec0, token 13, first 5 dims: {sin_emb[0, 0, 13, :5].detach().numpy()}")

        # Debug: Manually run rotary embedding to get Q and K after rotation
        from transformers.models.glm4v.modeling_glm4v import apply_multimodal_rotary_pos_emb
        # Reshape Q and K to [B, num_heads, T, head_dim]
        num_heads = 32
        num_kv_heads = 2
        head_dim = 128
        mrope_section = [16, 24, 24]  # from config
        q_rs = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)  # [1, 32, 32, 128]
        k_rs = k.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # [1, 2, 32, 128]
        q_rot, k_rot = apply_multimodal_rotary_pos_emb(q_rs, k_rs, cos_emb, sin_emb, mrope_section)
        print(f"    Q after rotation shape: {list(q_rot.shape)}")
        print(f"    Q_rot @ token 13, head 0, first 10 dims: {q_rot[0, 0, 13, :10].detach().numpy()}")
        print(f"    K_rot @ token 13, head 0, first 10 dims: {k_rot[0, 0, 13, :10].detach().numpy()}")

        # 2c: Full attention output (manually run attention)
        attn_result = attn(x_norm, position_embeddings=position_embeddings, attention_mask=None)
        # attn returns a tuple (output, None, None) or similar
        attn_output = attn_result[0] if isinstance(attn_result, tuple) else attn_result
        print(f"  2c. Attention output (raw): {list(attn_output.shape)}, first 5: {attn_output[0, 0, :5].detach().numpy()}")

        # 2c': Apply post_self_attn_layernorm to attention output (before residual)
        attn_normed = layer0.post_self_attn_layernorm(attn_output)
        print(f"  2c'. After post_self_attn_layernorm: {list(attn_normed.shape)}, first 5: {attn_normed[0, 0, :5].detach().numpy()}")

        # 2d: After attention residual
        after_attn = embeddings + attn_normed
        print(f"  2d. After attention residual: {list(after_attn.shape)}, first 5: {after_attn[0, 0, :5].detach().numpy()}")

        # 2e: Post attention layernorm
        mlp_norm = layer0.post_attention_layernorm(after_attn)
        print(f"  2e. After post_attn_layernorm: {list(mlp_norm.shape)}, first 5: {mlp_norm[0, 0, :5].detach().numpy()}")

        # 2f: MLP gate_up projection
        mlp = layer0.mlp
        gate_up = mlp.gate_up_proj(mlp_norm)
        print(f"  2f. gate_up projection: {list(gate_up.shape)}, first 5: {gate_up[0, 0, :5].detach().numpy()}")

        # 2g: Full MLP output (raw)
        mlp_output = mlp(mlp_norm)
        print(f"  2g. MLP output (raw): {list(mlp_output.shape)}, first 5: {mlp_output[0, 0, :5].detach().numpy()}")

        # 2g': Apply post_mlp_layernorm to MLP output (before residual)
        mlp_normed = layer0.post_mlp_layernorm(mlp_output)
        print(f"  2g'. After post_mlp_layernorm: {list(mlp_normed.shape)}, first 5: {mlp_normed[0, 0, :5].detach().numpy()}")

        # 2h: Full layer output (step-by-step with 4 norms)
        layer0_output_manual = after_attn + mlp_normed
        print(f"  2h. Layer 0 output (manual 4-norm): {list(layer0_output_manual.shape)}, first 5: {layer0_output_manual[0, 0, :5].detach().numpy()}")

        # Debug: print values at token 13 (where max diff occurs)
        print(f"\n  Debug: Values at token 13 (max diff position):")
        print(f"      attn_output token 13, dims 0-4: {attn_output[0, 13, :5].detach().numpy()}")
        print(f"      attn_output token 13, dims 3556-3560: {attn_output[0, 13, 3556:3561].detach().numpy()}")
        print(f"      attn_normed token 13, dims 3556-3560: {attn_normed[0, 13, 3556:3561].detach().numpy()}")
        print(f"      after_attn token 13, dims 3556-3560: {after_attn[0, 13, 3556:3561].detach().numpy()}")
        print(f"      mlp_norm token 13, dims 3556-3560: {mlp_norm[0, 13, 3556:3561].detach().numpy()}")
        print(f"      mlp_output token 13, dims 3556-3560: {mlp_output[0, 13, 3556:3561].detach().numpy()}")
        print(f"      mlp_normed token 13, dims 3556-3560: {mlp_normed[0, 13, 3556:3561].detach().numpy()}")
        print(f"      layer0_output token 13, dims 3556-3560: {layer0_output_manual[0, 13, 3556:3561].detach().numpy()}")

        # Verify against calling layer0 directly
        layer0_output_direct = layer0(embeddings, position_embeddings=position_embeddings, attention_mask=None)
        print(f"  2h. Layer 0 output (direct): {list(layer0_output_direct.shape)}, first 5: {layer0_output_direct[0, 0, :5].detach().numpy()}")

        # Check if they match
        diff = (layer0_output_manual - layer0_output_direct).abs().max().item()
        print(f"  2h. Manual (4-norm) vs Direct max diff: {diff}")

        # Save the direct output
        layer0_output_direct.detach().numpy().astype(np.float32).tofile(os.path.join(output_dir, "expected_layer0.bin"))

        # Use the full model forward (text-only, no images)
        outputs = model.model(input_ids=input_ids_tensor)
        hidden_states = outputs.last_hidden_state  # [B, T, dim]

        # Save hidden states
        hidden_states.numpy().astype(np.float32).tofile(os.path.join(output_dir, "expected_text_hidden.bin"))
        print(f"  Hidden states: {list(hidden_states.shape)}")
        print(f"  Hidden first 5 values: {hidden_states[0, 0, :5].numpy()}")

        # Also compute and save logits (for full pipeline test)
        logits = model.lm_head(hidden_states)
        logits.numpy().astype(np.float32).tofile(os.path.join(output_dir, "expected_text_logits.bin"))
        print(f"  Logits: {list(logits.shape)}")


def export_text_weights_raw(model, output_dir: str):
    """Export text model weights as raw float16 binary.

    Order must match C++ load_text_weights() exactly:
    1. embed_tokens: [vocab_size, dim]
    2. For each of 40 layers:
       - input_layernorm.weight
       - post_attention_layernorm.weight
       - self_attn.q_proj.weight, q_proj.bias
       - self_attn.k_proj.weight, k_proj.bias
       - self_attn.v_proj.weight, v_proj.bias
       - self_attn.o_proj.weight
       - mlp.gate_up_proj.weight (fused gate+up)
       - mlp.down_proj.weight
    3. norm.weight
    4. lm_head.weight
    """
    weights_path = os.path.join(output_dir, "text_model.bin")
    print(f"\nSaving text weights to {weights_path}...")

    text = model.model.language_model
    lm_head = model.lm_head
    total_bytes = 0

    with open(weights_path, 'wb') as f:
        def write(name, tensor):
            nonlocal total_bytes
            arr = tensor.detach().numpy().astype(np.float16)
            arr.tofile(f)
            total_bytes += arr.nbytes
            print(f"  {name}: {list(arr.shape)}")

        def write_zeros(name, shape):
            nonlocal total_bytes
            arr = np.zeros(shape, dtype=np.float16)
            arr.tofile(f)
            total_bytes += arr.nbytes
            print(f"  {name}: {list(shape)} (zeros)")

        # 1. Embedding table [vocab_size, dim]
        write("embed_tokens", text.embed_tokens.weight.data)

        # 2. Transformer layers
        for i, layer in enumerate(text.layers):
            p = f"layers.{i}."

            # All 4 layer norms (GLM-4V has 4 per layer, not 2)
            write(f"{p}input_layernorm", layer.input_layernorm.weight.data)
            write(f"{p}post_self_attn_layernorm", layer.post_self_attn_layernorm.weight.data)
            write(f"{p}post_attention_layernorm", layer.post_attention_layernorm.weight.data)
            write(f"{p}post_mlp_layernorm", layer.post_mlp_layernorm.weight.data)

            # Self attention - transpose weights for MLX [in, out]
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

            # MLP - already fused gate_up_proj in GLM-4V
            mlp = layer.mlp
            write(f"{p}gate_up_proj.weight", mlp.gate_up_proj.weight.data.T.contiguous())
            write(f"{p}down_proj.weight", mlp.down_proj.weight.data.T.contiguous())

        # 3. Final norm
        write("norm.weight", text.norm.weight.data)

        # 4. LM head (output projection)
        write("lm_head.weight", lm_head.weight.data.T.contiguous())

    print(f"\nTotal text weights: {total_bytes / 1024 / 1024:.1f} MB")


def export_int8_text_weights(model, output_dir: str):
    """Export INT8 quantized text model weights.

    Uses symmetric per-channel quantization for linear layers.
    Embeddings and layer norms stay in FP16.
    """
    weights_path = os.path.join(output_dir, "text_model_int8.bin")
    print(f"\nSaving INT8 quantized text weights to {weights_path}...")

    text = model.model.language_model
    lm_head = model.lm_head
    total_bytes = 0

    def quantize_per_channel(tensor):
        """Symmetric per-channel INT8 quantization.

        For weight [in, out], quantize per output channel.
        Returns (int8_weights, fp16_scales)
        """
        # Per-channel (output dim) max absolute value
        abs_max = tensor.abs().max(dim=0, keepdim=True).values.clamp(min=1e-8)
        scale = abs_max / 127.0  # Scale to int8 range [-127, 127]
        quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
        return quantized, scale.squeeze(0).half()

    with open(weights_path, 'wb') as f:
        def write_fp16(name, tensor):
            nonlocal total_bytes
            arr = tensor.detach().numpy().astype(np.float16)
            arr.tofile(f)
            total_bytes += arr.nbytes
            print(f"  {name}: {list(arr.shape)} (fp16)")

        def write_int8(name, tensor):
            nonlocal total_bytes
            arr = tensor.detach().numpy().astype(np.int8)
            arr.tofile(f)
            total_bytes += arr.nbytes
            print(f"  {name}: {list(arr.shape)} (int8)")

        def write_quant_linear(name, weight, has_bias, bias=None):
            """Write quantized linear layer: INT8 weights + FP16 scales + optional FP16 bias"""
            nonlocal total_bytes
            # Transpose to [in, out] for MLX
            w_t = weight.T.contiguous()
            qw, scales = quantize_per_channel(w_t)

            # Write INT8 weights
            arr = qw.numpy().astype(np.int8)
            arr.tofile(f)
            total_bytes += arr.nbytes

            # Write FP16 scales
            sarr = scales.numpy().astype(np.float16)
            sarr.tofile(f)
            total_bytes += sarr.nbytes

            print(f"  {name}: weight {list(arr.shape)} (int8), scales {list(sarr.shape)} (fp16)")

            # Write FP16 bias if present
            if has_bias and bias is not None:
                barr = bias.detach().numpy().astype(np.float16)
                barr.tofile(f)
                total_bytes += barr.nbytes
                print(f"  {name}: bias {list(barr.shape)} (fp16)")
            elif has_bias:
                # Write zeros if no bias
                barr = np.zeros(weight.shape[0], dtype=np.float16)
                barr.tofile(f)
                total_bytes += barr.nbytes
                print(f"  {name}: bias {list(barr.shape)} (zeros, fp16)")

        # 1. Embedding table [vocab_size, dim] - keep in FP16
        write_fp16("embed_tokens", text.embed_tokens.weight.data)

        # 2. Transformer layers
        for i, layer in enumerate(text.layers):
            p = f"layers.{i}."

            # Layer norms stay in FP16
            write_fp16(f"{p}input_layernorm", layer.input_layernorm.weight.data)
            write_fp16(f"{p}post_self_attn_layernorm", layer.post_self_attn_layernorm.weight.data)
            write_fp16(f"{p}post_attention_layernorm", layer.post_attention_layernorm.weight.data)
            write_fp16(f"{p}post_mlp_layernorm", layer.post_mlp_layernorm.weight.data)

            # Attention - INT8 quantized
            attn = layer.self_attn
            write_quant_linear(f"{p}q_proj", attn.q_proj.weight.data, True, attn.q_proj.bias)
            write_quant_linear(f"{p}k_proj", attn.k_proj.weight.data, True, attn.k_proj.bias)
            write_quant_linear(f"{p}v_proj", attn.v_proj.weight.data, True, attn.v_proj.bias)
            write_quant_linear(f"{p}o_proj", attn.o_proj.weight.data, False)

            # MLP - INT8 quantized
            mlp = layer.mlp
            write_quant_linear(f"{p}gate_up_proj", mlp.gate_up_proj.weight.data, False)
            write_quant_linear(f"{p}down_proj", mlp.down_proj.weight.data, False)

        # 3. Final norm - FP16
        write_fp16("norm.weight", text.norm.weight.data)

        # 4. LM head - INT8 quantized
        write_quant_linear("lm_head", lm_head.weight.data, False)

    print(f"\nTotal INT8 quantized text weights: {total_bytes / 1024 / 1024:.1f} MB")
    fp16_size = 17929.6  # MB from FP16 export
    compression = fp16_size / (total_bytes / 1024 / 1024)
    print(f"Compression ratio vs FP16: {compression:.2f}x")


def main():
    global MODEL_PATH

    parser = argparse.ArgumentParser(description="Export GLM-4.6V weights")
    parser.add_argument("--model", default=MODEL_PATH, help="Model path")
    parser.add_argument("--output", default="/Users/Stephen/Documents/mlx_models/glm/vision_weights", help="Output directory")
    parser.add_argument("--vision", action="store_true", help="Export vision encoder weights")
    parser.add_argument("--text", action="store_true", help="Export text model weights (FP16)")
    parser.add_argument("--text-int8", action="store_true", help="Export text model weights (INT8 quantized)")
    args = parser.parse_args()

    # Default to vision if nothing specified
    if not args.vision and not args.text and not args.text_int8:
        args.vision = True

    MODEL_PATH = args.model

    try:
        model, processor, output_dir, weights = export_vision_weights(args.output)

        if args.vision:
            export_weights_raw(weights, output_dir, model.model.visual)
            create_test_data(model, output_dir)
            print(f"\nVision done. Verify with: GLM_VERIFY=1 GLM_WEIGHTS_DIR={output_dir} ./glm46v_mlx")

        if args.text:
            export_text_weights_raw(model, output_dir)
            create_text_test_data(model, output_dir)
            print(f"\nText weights exported to: {output_dir}/text_model.bin")

        if args.text_int8:
            export_int8_text_weights(model, output_dir)
            print(f"\nINT8 text weights exported to: {output_dir}/text_model_int8.bin")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
