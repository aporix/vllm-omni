# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance AR-stage model wrapper (two-stage topology).

Lance is BAGEL-lineage, so the AR/thinker stage is the same Qwen2-MoT model
BAGEL uses.  This subclass exists so the two-stage pipeline has a distinct
arch name to register against; behaviour is inherited unchanged from
:class:`OmniBagelForConditionalGeneration`.

IMPORTANT: the two-stage Lance topology is a follow-up.  It additionally
requires a ``LanceConfig`` + ``LanceProcessor`` registered in the ``vllm``
package (the base ``BagelForConditionalGeneration`` resolves its HF config /
processor by ``model_type == "bagel"``; Lance ships ``model_type ==
"qwen2_5_vl"`` in ``llm_config.json`` and no BAGEL-style top-level config).
Until that lands, use the **single-stage** topology (``model_type: lance`` ->
``LANCE_SINGLE_STAGE_PIPELINE`` -> ``LancePipeline``), which is fully wired and
needs no ``vllm``-submodule changes.
"""

from __future__ import annotations

from vllm_omni.model_executor.models.bagel.bagel import (
    OmniBagelForConditionalGeneration,
)


class OmniLanceForConditionalGeneration(OmniBagelForConditionalGeneration):
    """Lance AR-stage model.  Identical to BAGEL's (same Qwen2-MoT weights);
    distinct class only so the two-stage pipeline can register a ``lance``
    arch separately.  See module docstring for the two-stage follow-up."""

    # Same packed→sublayer LoRA / MoT mapping as BAGEL (weight names match).
    packed_modules_mapping = OmniBagelForConditionalGeneration.packed_modules_mapping
