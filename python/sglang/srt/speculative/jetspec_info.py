"""JetSpec speculative decoding data structures.

JetSpec uses the DFlash draft-head contract (target hidden states + target
embedding/lm_head) and the DDTree target-verify layout, but keeps distinct
SpecInput tags so scheduler/attention code can identify it independently.
"""

from __future__ import annotations

from sglang.srt.speculative.dflash_ddtree_info import DFlashDDTreeVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_info import SpecInputType


class JetSpecDraftInput(DFlashDraftInputV2):
    def __post_init__(self):
        super().__post_init__()
        self.spec_input_type = SpecInputType.JETSPEC_DRAFT


class JetSpecVerifyInput(DFlashDDTreeVerifyInput):
    def __post_init__(self):
        super().__post_init__()
        self.spec_input_type = SpecInputType.JETSPEC_VERIFY
