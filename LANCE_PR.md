# [New Model] Lance (ByteDance) — #3697

Adds Lance (`bytedance-research/Lance`), a 3B unified AR+diffusion
multimodal model on a Qwen2.5-VL backbone.

## Why this is BAGEL, not WAN

The issue thread suggested remapping against `WanTransformer3DModel`. The
released `Lance_3B` checkpoint is instead **BAGEL Mixture-of-Transformers**:
every layer carries `mlp_moe_gen` / `q_proj_moe_gen` /
`input_layernorm_moe_gen` twins plus `q_norm`/`k_norm`, with top-level
`vae2llm` / `llm2vae` / `time_embedder` / `latent_pos_embed` connectors —
identical naming to vLLM-Omni's BAGEL. The only WAN element is the VAE.
Lance therefore reuses BAGEL's transformer core verbatim and specializes
only the LLM (Qwen2.5-VL-MoT), ViT (Qwen2.5-VL vision), VAE (Wan2.2) and
checkpoint layout.

## Scope and status

All six single-stage Lance modalities, verified end-to-end on 1× B300:

| Modality | Status |
|----------|--------|
| `t2i` (text→image) | working — 1024×1024, 0 missing keys |
| `t2v` (text→video) | working — photorealistic frames with motion; 768×768 trained bucket |
| `image_edit` (img2img) | working — Lance-native VAE prefill scatters Wan2.2 latents into the LLM query sequence |
| `video_edit` (video2video) | working — multi-frame VAE prefill + 3-D mRoPE; needs the `Lance_3B_Video` checkpoint |
| `x2t_image` (image understanding) | working — Qwen2.5-VL ViT prefill + 3-D mRoPE |
| `x2t_video` (video understanding) | working — Qwen2-VL video processor + 3-D `video_grid_thw` |

`rope_scaling = {"type": "mrope", "mrope_section": [16, 24, 24]}` is wired
through `BagelRotaryEmbedding`: t2i uses scalar position ids
(BAGEL-equivalent); x2t / video / edit thread per-axis 3-D position ids.
Geometry constants (`latent_patch_size_spatial = 1`, `max_latent_size = 64`,
untied `lm_head`, Wan2.2 VAE `z=48 /16 spatial /4 temporal`) were locked in
by inspecting the released checkpoint directly.

The two-stage AR+DiT topology is the only deferred path (needs
`LanceConfig` / `LanceProcessor` in the `vllm` package — a separate
human-authored PR per vLLM's contribution policy).

## Test plan

Offline (`examples/offline_inference/lance/end2end.py`), e.g. t2i:

```bash
python examples/offline_inference/lance/end2end.py \
    --model bytedance-research/Lance \
    --prompts "a corgi astronaut on the moon, cinematic" \
    --steps 30 --cfg-text-scale 4.0 --timestep-shift 3.5 --output ./out
```

`--modality {text2video,img2img,video2video,img2text,video2text}` selects
the other paths; video paths use `--model bytedance-research/Lance/Lance_3B_Video`.

Online (OpenAI-compatible, `examples/online_serving/lance/`):

```bash
bash examples/online_serving/lance/run_server.sh        # vllm serve … --omni
python examples/online_serving/lance/openai_chat_client.py \
    --prompt "A cute corgi astronaut on the moon, cinematic" \
    --modality text2img --output corgi.png
```

E2E coverage: `tests/e2e/online_serving/test_lance.py`. Tested on 1×
NVIDIA B300, `torch 2.11.0+cu130`, stock attention backend.

## Follow-ups

1. **Two-stage AR + DiT** — register `LanceConfig` / `LanceProcessor` in
   the `vllm` package; separate human-authored PR per vLLM policy.
2. **`video_edit` quality polish** — output is more abstract than `t2v`
   at the same resolution (position-id offset between the VAE-prefill and
   gen-latent blocks). Functionally correct end-to-end.

Closes #3697.
