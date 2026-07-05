"""JetSpec speculative decoding worker.

The worker intentionally has its own public algorithm name and SpecInput tags.
It reuses the DFlash/DDTree execution primitives because JetSpec's draft head
has the same runtime contract: target hidden features condition a draft-only
transformer, target embedding seeds the draft block, and target lm_head scores
the draft hidden states before tree verification.
"""

from __future__ import annotations

import torch

from sglang.srt.speculative.dflash_ddtree_worker import DFlashDDTreeWorker
from sglang.srt.speculative.jetspec_info import JetSpecDraftInput, JetSpecVerifyInput
from sglang.srt.speculative.jetspec_utils import build_jetspec_tree_from_topk
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class JetSpecWorker(DFlashDDTreeWorker):
    draft_input_cls = JetSpecDraftInput
    verify_input_cls = JetSpecVerifyInput
    forward_spec_algorithm = SpeculativeAlgorithm.JETSPEC
    spec_name = "JETSPEC"

    def _build_tree_from_topk(
        self,
        top_log_probs: torch.Tensor,
        top_token_ids: torch.Tensor,
        tree_budget: int,
    ):
        return build_jetspec_tree_from_topk(top_log_probs, top_token_ids, tree_budget)
