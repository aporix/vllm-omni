# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for Lance.

Lance's classifier-free-guidance prompt expansion and KV-cache collection are
model-agnostic and identical to BAGEL's (3-branch CFG: gen / cfg_text /
cfg_img).  Re-export BAGEL's implementations so the two-stage pipeline config
can reference ``...stage_input_processors.lance.{expand_cfg_prompts,
collect_cfg_kv_caches}`` without duplicating logic.
"""

from vllm_omni.model_executor.stage_input_processors.bagel import (  # noqa: F401
    CFG_IMG_SUFFIX,
    CFG_TEXT_SUFFIX,
    GEN_THINK_SYSTEM_PROMPT,
    VLM_THINK_SYSTEM_PROMPT,
    ExpandedPrompt,
    collect_cfg_kv_caches,
    expand_cfg_prompts,
    expand_cfg_prompts_think,
)

__all__ = [
    "CFG_IMG_SUFFIX",
    "CFG_TEXT_SUFFIX",
    "GEN_THINK_SYSTEM_PROMPT",
    "VLM_THINK_SYSTEM_PROMPT",
    "ExpandedPrompt",
    "collect_cfg_kv_caches",
    "expand_cfg_prompts",
    "expand_cfg_prompts_think",
]
