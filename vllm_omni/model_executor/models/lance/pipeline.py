# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance pipeline topologies.

Single-stage (runnable / registered):
  Stage 0: DiT — self-contained diffusion stage that handles all modalities
           (text2img, image_edit, x2t_image) internally via its own
           Qwen2-MoT LLM, Qwen2.5-VL ViT, Wan2.2 VAE, and tokenizer.
           Mirrors ``bagel_single_stage``.

Two-stage (follow-up — see vllm_omni/model_executor/models/lance/lance.py):
  Stage 0: Thinker (AR) -> Stage 1: DiT.  Needs a LanceConfig/LanceProcessor
  in the ``vllm`` package; defined here for completeness but gated behind the
  follow-up so it is not selected by default.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.lance"

# --- Single-stage (default, fully wired) -------------------------------- #
LANCE_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="lance",
    model_arch="LancePipeline",
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
        ),
    ),
)

# --- Two-stage (follow-up; not registered as default) ------------------- #
LANCE_PIPELINE = PipelineConfig(
    model_type="lance_two_stage",
    model_arch="OmniLanceForConditionalGeneration",
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="OmniLanceForConditionalGeneration",
            engine_output_type="text",
            prompt_expand_func=f"{_PROC}.expand_cfg_prompts",
            omni_kv_config={
                "need_send_cache": True,
                "kv_transfer_criteria": {"type": "prefill_finished"},
            },
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            cfg_kv_collect_func=f"{_PROC}.collect_cfg_kv_caches",
            omni_kv_config={"need_recv_cache": True},
        ),
    ),
)
