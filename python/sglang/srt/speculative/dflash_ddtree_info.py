from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_ddtree_utils import follow_verified_tree
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import move_accept_tokens_to_target_kvcache

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


@dataclass
class DFlashDDTreeVerifyInput(SpecInput):
    """Inputs for a tree-shaped target-model verify forward in DFlash DDTree.

    Mirrors `DFlashVerifyInput` but carries tree topology: a tree visibility
    `custom_mask` (a query attends to the prefix + its ancestor path + itself)
    and EAGLE-style `retrieve_next_token` / `retrieve_next_sibling` for hybrid
    GDN/Mamba targets. KV slots for the verify pack are supplied by the worker
    via `batch.out_cache_loc` (same width as the draft block since
    `block_size == 1 + tree_budget`).
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    child_maps_per_req: list[list[dict[int, int]]]
    visibility_per_req: list[torch.Tensor]
    topk: int = 1
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    num_tokens_per_batch: int = -1
    # EAGLE-style first-child / next-sibling tensors for tree-aware verify in
    # hybrid (GDN/Mamba + softmax) targets. Shape: (bs, draft_token_num), -1
    # padded. None for non-hybrid targets where the GDN tree-aware path is not
    # needed (full-attention targets).
    retrieve_next_token: torch.Tensor | None = None
    retrieve_next_sibling: torch.Tensor | None = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DDTREE_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: "TpModelWorker",
    ) -> tuple[ForwardBatch, bool]:
        """Build the tree visibility mask and package the verify ForwardBatch.

        Mirrors `DFlashVerifyInput.prepare_for_verify`: the worker has already
        placed the verify-pack KV slots in `batch.out_cache_loc`, so this only
        constructs the `custom_mask` (prefix-allow ⊕ tree visibility) and
        packages the forward batch with cuda-graph / eager metadata.
        """
        batch.input_ids = self.draft_token
        batch.spec_info = self
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = self.capture_hidden_mode

        # Tree visibility mask: each query row = [prefix_allow | visibility].
        # `prefix_allow` is True over the whole committed prefix; `visibility`
        # restricts the verify segment to the query's ancestor path + itself.
        # Use `batch.seq_lens` (GPU, never raised to the verify host bound) so
        # the prefix width is the committed length before this verify.
        if not batch.forward_mode.is_idle():
            mask_chunks: List[torch.Tensor] = []
            q_len = int(self.draft_token_num)
            prefix_lens = batch.seq_lens.tolist()
            for req_idx, prefix_len_i in enumerate(prefix_lens):
                prefix_len_i = int(prefix_len_i)
                prefix_allow = torch.ones(
                    (q_len, prefix_len_i), dtype=torch.bool, device=batch.device
                )
                visibility = self.visibility_per_req[req_idx].to(
                    device=batch.device, dtype=torch.bool, non_blocking=True
                )
                if visibility.shape != (q_len, q_len):
                    raise RuntimeError(
                        "DFLASH_DDTREE visibility shape mismatch: "
                        f"expected {(q_len, q_len)}, got {tuple(visibility.shape)}."
                    )
                mask_chunks.append(
                    torch.cat([prefix_allow, visibility], dim=1).flatten()
                )
            self.custom_mask = (
                torch.cat(mask_chunks, dim=0)
                if mask_chunks
                else torch.empty((0,), dtype=torch.bool, device=batch.device)
            )

        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.load_batch(
                verify_forward_batch
            )
        elif not batch.forward_mode.is_idle():
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )

        return verify_forward_batch, can_run_cuda_graph

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
        """Tree accept + KV compaction. Returns the pieces the worker needs.

        Does NOT append to `req.output_ids`, advance `batch.seq_lens`, or touch
        finishing / `kv_committed_len` / `spec_verify_ct`: the scheduler's
        `process_batch_result_decode` does all of that from the returned
        `next_token_ids` / `commit_lens` (same contract as linear DFlash).

        Returns
        -------
        next_token_ids: (bs * draft_token_num,) accepted tokens per req laid out
            at ``[i*stride : i*stride + commit_lens[i]]`` (padded with 0), matching
            `_resolve_spec_v2_tokens`.
        commit_lens: (bs,) number of committed tokens per req (root + accepted drafts).
        next_target_hidden: (sum(commit_lens), hidden) committed path hidden states,
            in per-req path order, for draft-KV materialization.
        committed_indices_per_req: tree node indices of the accepted path per req
            (root-leading); used for the path-end Mamba write-back.
        """
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device
        sampling_info = batch.sampling_info
        if sampling_info is not None and not sampling_info.is_all_greedy:
            raise RuntimeError("DFLASH_DDTREE MVP only supports greedy sampling.")

        stride = int(self.draft_token_num)
        candidates = self.draft_token.view(bs, stride)
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, stride
        )
        candidates_cpu = candidates.cpu()
        target_predict_cpu = target_predict.cpu()

        next_token_ids_2d = torch.zeros(
            (bs, stride), dtype=torch.int64, device=device
        )
        commit_lens_cpu: list[int] = []
        committed_indices_per_req: list[list[int]] = []

        for i in range(bs):
            accepted_indices, next_token = follow_verified_tree(
                self.child_maps_per_req[i], target_predict_cpu[i]
            )
            # Accepted drafts = path nodes after the root; the bonus is the
            # target's prediction past the last accepted node. commit_len =
            # root + accepted drafts == accepted drafts + bonus.
            path_draft_tokens = candidates_cpu[i][
                [idx for idx in accepted_indices[1:]]
            ].tolist()
            emitted = path_draft_tokens + [int(next_token)]
            commit_len = len(accepted_indices)
            if commit_len > stride:
                raise RuntimeError(
                    "DFLASH_DDTREE accepted path longer than verify width: "
                    f"commit_len={commit_len}, stride={stride}."
                )
            next_token_ids_2d[i, :commit_len] = torch.tensor(
                emitted, dtype=torch.int64, device=device
            )
            commit_lens_cpu.append(commit_len)
            committed_indices_per_req.append(accepted_indices)

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)

        # --- Compact the accepted tree path into the CONTIGUOUS front of each
        # per-req verify block, matching linear DFlash's KV layout.
        #
        # The tree verify wrote target KV for every node into `batch.out_cache_loc`
        # -- the reserved, page-aligned block S[0:stride] handed over by
        # DFlashDraftInputV2.prepare_for_decode. The accepted path is a SCATTERED
        # subset S[committed_indices] (path nodes can sit anywhere in the block).
        # Simply pointing req_to_token at those scattered physical slots (the old
        # approach) breaks page-level KV accounting when page_size > 1: at request
        # finish the committed prefix is radix-cached (page-aligned) while the
        # reserved tail [committed_len:kv_allocated_len] is freed, and a boundary
        # page ends up claimed by BOTH -> `pool memory leak detected` on the full
        # pool. Linear DFlash never hits this because its accepted tokens ARE the
        # block's leading contiguous slots.
        #
        # Fix: physically MOVE the accepted path's KV to S[0:commit_len] (the same
        # EAGLE tree-verify path via `move_accept_tokens_to_target_kvcache`). Since
        # req_to_token[prefix:prefix+commit_len] already maps to S[0:commit_len]
        # (that region was reserved contiguous and `out_cache_loc` was gathered
        # from it), the mapping is correct after the move with NO scatter rewrite.
        # Non-accepted slots stay in the reserved region and are reclaimed at
        # request finish, exactly like linear DFlash's [commit_len:block] tail.
        out_cache_loc_2d = batch.out_cache_loc.view(bs, stride)
        accept_index = torch.full((bs, stride), -1, dtype=torch.int64, device=device)
        for i, committed_indices in enumerate(committed_indices_per_req):
            idx_tensor = torch.tensor(
                committed_indices, dtype=torch.int64, device=device
            )
            accept_index[i, : idx_tensor.numel()] = i * stride + idx_tensor

        move_accept_tokens_to_target_kvcache(
            batch,
            accept_index,
            commit_lens.to(torch.int64) - 1,
            batch.token_to_kv_pool_allocator,
        )

        # After the move the committed slots are the block's contiguous prefix;
        # expose them as `out_cache_loc` for downstream bookkeeping.
        selected_locs = [out_cache_loc_2d[i, : commit_lens_cpu[i]] for i in range(bs)]
        batch.out_cache_loc = (
            torch.cat(selected_locs, dim=0)
            if selected_locs
            else batch.out_cache_loc.reshape(-1)[:0]
        )

        # --- Concatenate committed path hidden states (path order) for the
        # draft-KV materialization step in the worker. ---
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DFLASH_DDTREE verify requires target hidden states.")
        hidden = hidden.view(bs, stride, -1)
        segments: list[torch.Tensor] = []
        for i, committed_indices in enumerate(committed_indices_per_req):
            if committed_indices:
                idx_tensor = torch.tensor(
                    committed_indices, dtype=torch.long, device=hidden.device
                )
                segments.append(hidden[i].index_select(0, idx_tensor))
        next_target_hidden = (
            torch.cat(segments, dim=0)
            if segments
            else hidden.reshape(-1, hidden.shape[-1])[:0]
        )

        logits_output.hidden_states = None

        return (
            next_token_ids_2d.reshape(-1),
            commit_lens,
            next_target_hidden,
            committed_indices_per_req,
        )
