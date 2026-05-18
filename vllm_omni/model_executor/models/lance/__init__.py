# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance (ByteDance) — vLLM-Omni model_executor package.

The runnable, registered path is the **single-stage** topology: a
self-contained diffusion stage (``LancePipeline``) that owns the Qwen2-MoT
LLM, the Qwen2.5-VL ViT, the Wan2.2 VAE and the tokenizer — directly analogous
to ``bagel_single_stage``.

The two-stage (separate AR "thinker" + DiT) topology additionally needs a
``LanceConfig`` / ``LanceProcessor`` registered in the ``vllm`` package
(mirroring ``vllm.transformers_utils.configs.bagel`` /
``vllm.model_executor.models.bagel``); that is an explicit follow-up and is
intentionally not wired here to avoid shipping a half-working AR stage.
"""

from vllm_omni.model_executor.models.lance.lance import (
    OmniLanceForConditionalGeneration,
)

__all__ = ["OmniLanceForConditionalGeneration"]
