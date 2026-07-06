from __future__ import annotations

import logging
import os

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_ddtree_info import DFlashDDTreeVerifyInput
from sglang.srt.speculative.dflash_ddtree_utils import (
    DFlashDDTreeBuildWorkspace,
    build_chain_prefill_ddtree_tree_from_topk,
    build_ddtree_tree_from_topk,
    build_linear_ddtree_tree_from_topk,
    pad_ddtree_build_outputs,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.triton_ops.dflash import _prepare_dflash_draft_block_unchecked

logger = logging.getLogger(__name__)


class DFlashDDTreeWorker(DFlashWorkerV2):
    """DFlash + DDTree (tree-shaped) speculative decoding worker.

    DDTree extends the V2 DFlash worker. Only the decode round is overridden:
    drafting switches from per-position top-1 to top-k + a best-first tree,
    target verify uses a tree visibility mask (+ retrieve tensors for hybrid
    targets), and accept/commit follows a root-to-leaf path instead of a linear
    prefix. Prefill and idle are identical to DFlash and delegated to the V2
    parent.

    MVP scope: greedy, batch_size == 1, triton custom-mask verify, cuda graph
    disabled. Hybrid GDN/Mamba retrieve and cuda-graph capture/replay land in a
    follow-up (the retrieve tensors are already plumbed through).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree_budget = int(self.server_args.speculative_dflash_tree_budget)
        self._ddtree_topk_values_buf: torch.Tensor | None = None
        self._ddtree_topk_indices_buf: torch.Tensor | None = None
        self._ddtree_tree_tokens_buf: torch.Tensor | None = None
        self._ddtree_tree_positions_buf: torch.Tensor | None = None
        self._ddtree_retrieve_next_token_buf: torch.Tensor | None = None
        self._ddtree_retrieve_next_sibling_buf: torch.Tensor | None = None
        self._ddtree_next_token_ids_bufs: list[torch.Tensor] = []
        self._ddtree_commit_lens_bufs: list[torch.Tensor] = []
        self._ddtree_result_buf_index = 0
        self._ddtree_accept_index_buf: torch.Tensor | None = None
        self._ddtree_build_workspace = DFlashDDTreeBuildWorkspace()
        # CUDA graph capture locks the verify-sequence shape, so the worker must
        # always emit exactly `1 + tree_budget` verify tokens (padding the tree
        # to a fixed width when the heap drains early). `_handle_dflash_ddtree`
        # already enforces speculative_num_draft_tokens == 1 + tree_budget at
        # startup; keep a defensive worker-side check too.
        draft_cap = int(self.server_args.speculative_num_draft_tokens)
        required = 1 + int(self.tree_budget)
        if draft_cap != required:
            raise RuntimeError(
                "DFLASH_DDTREE requires --speculative-num-draft-tokens == "
                "1 + tree_budget for cuda graph capture. Got "
                f"speculative_num_draft_tokens={draft_cap}, "
                f"tree_budget={self.tree_budget} (need == {required})."
            )

    def _ensure_ddtree_topk_buffers(
        self,
        *,
        num_tokens: int,
        k: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (int(num_tokens), int(k))
        if (
            self._ddtree_topk_values_buf is None
            or self._ddtree_topk_indices_buf is None
            or tuple(self._ddtree_topk_values_buf.shape) != shape
            or tuple(self._ddtree_topk_indices_buf.shape) != shape
            or self._ddtree_topk_values_buf.dtype != dtype
            or self._ddtree_topk_values_buf.device != device
            or self._ddtree_topk_indices_buf.device != device
        ):
            self._ddtree_topk_values_buf = torch.empty(
                shape, dtype=dtype, device=device
            )
            self._ddtree_topk_indices_buf = torch.empty(
                shape, dtype=torch.long, device=device
            )
        return self._ddtree_topk_values_buf, self._ddtree_topk_indices_buf

    def _ensure_ddtree_decode_buffers(
        self,
        *,
        bs: int,
        q_len: int,
        device: torch.device,
    ) -> None:
        tree_shape = (int(bs), int(q_len))
        if (
            self._ddtree_tree_tokens_buf is None
            or tuple(self._ddtree_tree_tokens_buf.shape) != tree_shape
            or self._ddtree_tree_tokens_buf.device != device
        ):
            self._ddtree_tree_tokens_buf = torch.empty(
                tree_shape, dtype=torch.long, device=device
            )
            self._ddtree_tree_positions_buf = torch.empty(
                tree_shape, dtype=torch.int64, device=device
            )
            self._ddtree_retrieve_next_token_buf = torch.empty(
                tree_shape, dtype=torch.int32, device=device
            )
            self._ddtree_retrieve_next_sibling_buf = torch.empty(
                tree_shape, dtype=torch.int32, device=device
            )
            self._ddtree_next_token_ids_bufs = [
                torch.empty(tree_shape, dtype=torch.int64, device=device)
                for _ in range(2)
            ]
            self._ddtree_accept_index_buf = torch.empty(
                tree_shape, dtype=torch.int64, device=device
            )
            self._ddtree_commit_lens_bufs = [
                torch.empty((int(bs),), dtype=torch.int32, device=device)
                for _ in range(2)
            ]

    def _next_ddtree_result_buffers(self, bs: int, q_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ddtree_result_buf_index ^= 1
        return (
            self._ddtree_next_token_ids_bufs[self._ddtree_result_buf_index][:bs, :q_len],
            self._ddtree_commit_lens_bufs[self._ddtree_result_buf_index][:bs],
        )

    def _topk_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Global top-k over a vocab-parallel lm_head, returning normalized
        log-probs (the tree builder needs scores, not just argmax). This is the
        top-k generalisation of DFlashWorkerV2._greedy_sample_from_vocab_parallel_head.
        """
        if hidden_states.numel() == 0:
            empty_logits = torch.empty(
                (0, 0), dtype=torch.float32, device=hidden_states.device
            )
            empty_ids = torch.empty((0, 0), dtype=torch.long, device=hidden_states.device)
            return empty_logits, empty_ids

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH_DDTREE requires the target model to expose a vocab-parallel lm_head."
            )

        shard = lm_head.shard_indices
        weight = lm_head.weight
        weight_dtype = weight.dtype
        hs = (
            hidden_states
            if hidden_states.dtype == weight_dtype
            else hidden_states.to(weight_dtype)
        )

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        logits_parts = []
        id_parts = []
        if num_org > 0:
            logits_parts.append(torch.matmul(hs, weight[:num_org].T))
            id_parts.append(
                torch.arange(
                    org_vocab_start,
                    org_vocab_start + num_org,
                    dtype=torch.long,
                    device=hs.device,
                )
            )
        if num_added > 0:
            added_slice_start = num_org_padded
            added_slice_end = num_org_padded + num_added
            logits_parts.append(
                torch.matmul(hs, weight[added_slice_start:added_slice_end].T)
            )
            id_parts.append(
                torch.arange(
                    added_vocab_start,
                    added_vocab_start + num_added,
                    dtype=torch.long,
                    device=hs.device,
                )
            )
        if not logits_parts:
            raise RuntimeError(
                "DFLASH_DDTREE cannot compute logits from an empty vocab shard."
            )

        if tp_size == 1 and num_added == 0 and len(logits_parts) == 1:
            local_logits = logits_parts[0].float()
            local_log_z = torch.logsumexp(local_logits, dim=-1)
            local_k = min(int(k), int(local_logits.shape[-1]))
            local_top_logits, local_top_ids = self._ensure_ddtree_topk_buffers(
                num_tokens=int(local_logits.shape[0]),
                k=local_k,
                dtype=local_logits.dtype,
                device=local_logits.device,
            )
            torch.topk(
                local_logits,
                k=local_k,
                dim=-1,
                out=(local_top_logits, local_top_ids),
            )
            if org_vocab_start:
                local_top_ids.add_(org_vocab_start)
            return local_top_logits - local_log_z[:, None], local_top_ids

        local_logits = torch.cat(logits_parts, dim=-1).float()
        local_ids = torch.cat(id_parts, dim=0)
        local_log_z = torch.logsumexp(local_logits, dim=-1)
        local_k = min(int(k), int(local_logits.shape[-1]))
        local_top_logits, local_top_pos = torch.topk(local_logits, k=local_k, dim=-1)
        local_top_ids = (
            local_ids.index_select(0, local_top_pos.reshape(-1)).view_as(local_top_pos)
        )

        if tp_size == 1:
            return local_top_logits - local_log_z[:, None], local_top_ids

        num_tokens = int(hidden_states.shape[0])
        gathered_logits = torch.empty(
            (tp_size * num_tokens * local_k,),
            dtype=local_top_logits.dtype,
            device=hidden_states.device,
        )
        gathered_ids = torch.empty(
            (tp_size * num_tokens * local_k,),
            dtype=local_top_ids.dtype,
            device=hidden_states.device,
        )
        gathered_log_z = torch.empty(
            (tp_size * num_tokens,),
            dtype=local_log_z.dtype,
            device=hidden_states.device,
        )
        tp_group.all_gather_into_tensor(
            gathered_logits, local_top_logits.contiguous().view(-1)
        )
        tp_group.all_gather_into_tensor(
            gathered_ids, local_top_ids.contiguous().view(-1)
        )
        tp_group.all_gather_into_tensor(gathered_log_z, local_log_z.contiguous())

        gathered_logits = gathered_logits.view(tp_size, num_tokens, local_k).transpose(
            0, 1
        )
        gathered_ids = gathered_ids.view(tp_size, num_tokens, local_k).transpose(0, 1)
        flat_logits = gathered_logits.reshape(num_tokens, tp_size * local_k)
        flat_ids = gathered_ids.reshape(num_tokens, tp_size * local_k)
        global_k = min(int(k), int(flat_logits.shape[-1]))
        top_logits, top_pos = torch.topk(flat_logits, k=global_k, dim=-1)
        top_ids = torch.gather(flat_ids, 1, top_pos)

        log_z = torch.logsumexp(
            gathered_log_z.view(tp_size, num_tokens).transpose(0, 1), dim=-1
        )
        return top_logits - log_z[:, None], top_ids

    def _build_tree_from_topk(
        self,
        top_log_probs: torch.Tensor,
        top_token_ids: torch.Tensor,
        tree_budget: int,
    ):
        return build_ddtree_tree_from_topk(
            top_log_probs,
            top_token_ids,
            tree_budget,
            workspace=self._ddtree_build_workspace,
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DFLASH_DDTREE speculative decoding does not support return_logprob yet."
            )
        self._validate_phase1_sampling_support(batch)

        # Prefill and idle are draft-model identical to linear DFlash.
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return super().forward_batch_generation(batch, on_publish)
        if batch.forward_mode.is_idle():
            return super().forward_batch_generation(batch, on_publish)

        # MVP guards (tree accept is per-request and greedy-only for now).
        if batch.batch_size() != 1:
            raise RuntimeError(
                "DFLASH_DDTREE MVP only supports concurrency=1 / batch_size=1."
            )
        if batch.has_grammar:
            raise RuntimeError("DFLASH_DDTREE MVP does not support grammar constraints.")

        return self._ddtree_decode_round(batch, on_publish)

    def _ddtree_decode_round(
        self, batch: ScheduleBatch, on_publish
    ) -> GenerationBatchResult:
        # `seq_lens` is carried over from the previous overlap iteration and may
        # have been produced on another stream.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = len(batch.seq_lens)
        device = self.device
        block_size = int(self.block_size)
        tree_budget = int(self.tree_budget)
        profile = self._profile_start()

        draft_input = batch.spec_info
        if draft_input is None:
            draft_input = DFlashDraftInputV2.create_idle_input(device=device)
            batch.spec_info = draft_input

        # ====================================================================
        # Stage 1) Draft a fixed block, then take per-position top-k and build
        # the DDTree. The block prep + draft forward below mirror
        # DFlashWorkerV2.forward_batch_generation (dflash_worker_v2.py) verbatim:
        # the draft model and block width are identical to linear DFlash
        # (block_size == 1 + tree_budget). Only the post-forward sampling
        # (top-1 -> top-k + tree) differs.
        # TODO: extract a shared _draft_dflash_block() helper on DFlashWorkerV2.
        # ====================================================================
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DFLASH_DDTREE requires the target model to expose `lm_head` with `weight`."
            )

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None

        prefix_lens = batch.seq_lens
        block_ids = self._draft_block_ids_buf[:bs]
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH_DDTREE Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.bonus_tokens)
                torch.add(
                    prefix_lens.unsqueeze(1),
                    self._block_pos_offsets,
                    out=positions_2d,
                )
                end_offset = prefix_lens + block_size
                verify_out_cache_loc = assign_extend_cache_locs_func(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    start_offset=prefix_lens,
                    end_offset=end_offset,
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=device,
                )
                verify_out_cache_loc_2d.copy_(
                    verify_out_cache_loc.view(bs, block_size)
                )
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens)
            torch.add(
                prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )
            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            draft_seq_lens = prefix_lens
            if batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(batch.seq_lens_cpu)
                seq_lens_cpu.add_(block_size)
                draft_seq_lens_sum = int(seq_lens_cpu.sum())
            elif draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH_DDTREE,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._profile_mark(profile, "draft_prepare")

        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(forward_batch)
        draft_hidden = draft_out.logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH_DDTREE draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, block_size, -1)
        self._profile_mark(profile, "draft_forward")

        # --- DDTree divergence: top-k per horizon position + best-first tree. ---
        horizon_hidden = draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1])
        top_log_probs, top_token_ids = self._topk_from_vocab_parallel_head(
            hidden_states=horizon_hidden,
            lm_head=lm_head,
            k=tree_budget,
        )
        top_log_probs = top_log_probs.view(bs, block_size - 1, -1)
        top_token_ids = top_token_ids.view(bs, block_size - 1, -1)
        self._profile_mark(profile, "draft_topk")

        # DEBUG isolation switches (default off), used to localize the branching
        # bug by building degenerate trees while keeping the full verify shape:
        #   DDTREE_FORCE_LINEAR=1 -> pure depth chain, no sibling branches.
        #   DDTREE_FORCE_STAR=1   -> depth-1 star, pure branching, no multi-step
        #                            (slice top-k to a single depth row so
        #                            depth_limit==1 and every node is a root child).
        _ddtree_force_linear = os.environ.get("DDTREE_FORCE_LINEAR") == "1"
        _ddtree_force_star = os.environ.get("DDTREE_FORCE_STAR") == "1"
        _ddtree_chain_prefix = int(os.environ.get("DDTREE_CHAIN_PREFIX", "0"))
        if _ddtree_force_linear:
            (
                node_token_ids,
                node_depths,
                _,
                child_maps,
                visibility,
                retrieve_next_token_flat,
                retrieve_next_sibling_flat,
            ) = build_linear_ddtree_tree_from_topk(top_token_ids[0], tree_budget)
        elif _ddtree_force_star:
            (
                node_token_ids,
                node_depths,
                _,
                child_maps,
                visibility,
                retrieve_next_token_flat,
                retrieve_next_sibling_flat,
            ) = build_ddtree_tree_from_topk(
                top_log_probs[0][:1], top_token_ids[0][:1], tree_budget
            )
        elif _ddtree_chain_prefix > 0:
            (
                node_token_ids,
                node_depths,
                _,
                child_maps,
                visibility,
                retrieve_next_token_flat,
                retrieve_next_sibling_flat,
            ) = build_chain_prefill_ddtree_tree_from_topk(
                top_log_probs[0],
                top_token_ids[0],
                tree_budget,
                _ddtree_chain_prefix,
            )
        else:
            (
                node_token_ids,
                node_depths,
                _,
                child_maps,
                visibility,
                retrieve_next_token_flat,
                retrieve_next_sibling_flat,
            ) = build_ddtree_tree_from_topk(
                top_log_probs[0], top_token_ids[0], tree_budget
            )
        self._profile_mark(profile, "tree_build_cpu")

        # Pad the verify sequence to a fixed width of `1 + tree_budget` so the
        # cuda-graph backends see a fixed shape even when the heap drains early.
        # Pad-slot semantics: tree_tokens[pad] = bonus (hidden by visibility),
        # tree_positions[pad] = prefix len, retrieve_*[pad] = -1,
        # visibility[pad, *] = visibility[*, pad] = False.
        real_q_len = 1 + int(node_token_ids.numel())
        padded_q_len = 1 + tree_budget
        if real_q_len > padded_q_len:
            raise RuntimeError(
                "DFLASH_DDTREE build_ddtree_tree_from_topk produced more nodes "
                f"({real_q_len - 1}) than tree_budget ({tree_budget})."
            )

        padded_visibility, padded_next_token_flat, padded_next_sibling_flat = (
            pad_ddtree_build_outputs(
                visibility,
                retrieve_next_token_flat,
                retrieve_next_sibling_flat,
                padded_q_len,
            )
        )

        self._ensure_ddtree_decode_buffers(bs=bs, q_len=padded_q_len, device=device)
        assert self._ddtree_tree_tokens_buf is not None
        assert self._ddtree_tree_positions_buf is not None
        assert self._ddtree_retrieve_next_token_buf is not None
        assert self._ddtree_retrieve_next_sibling_buf is not None
        assert self._ddtree_next_token_ids_bufs
        assert self._ddtree_commit_lens_bufs
        assert self._ddtree_accept_index_buf is not None
        next_token_ids_buf, commit_lens_buf = self._next_ddtree_result_buffers(
            bs, padded_q_len
        )

        tree_tokens = self._ddtree_tree_tokens_buf[:bs, :padded_q_len]
        tree_tokens.copy_(block_ids[:, 0:1])  # broadcast bonus across all slots
        if real_q_len > 1:
            tree_tokens[0, 1:real_q_len].copy_(
                node_token_ids.to(device=device, non_blocking=True)
            )
        tree_positions = self._ddtree_tree_positions_buf[:bs, :padded_q_len]
        tree_positions.copy_(prefix_lens.to(torch.int64).unsqueeze(1))
        if real_q_len > 1:
            tree_positions[0, 1:real_q_len].copy_(
                prefix_lens[0].to(torch.int64)
                + node_depths.to(device=device, dtype=torch.int64, non_blocking=True)
            )

        # int32 to match the hybrid (GDN/Mamba) backend's captured retrieve
        # buffers and the EAGLE tree-aware kernel convention.
        retrieve_next_token = self._ddtree_retrieve_next_token_buf[:bs, :padded_q_len]
        retrieve_next_sibling = self._ddtree_retrieve_next_sibling_buf[
            :bs, :padded_q_len
        ]
        retrieve_next_token[0].copy_(
            padded_next_token_flat.to(device=device, dtype=torch.int32, non_blocking=True)
        )
        retrieve_next_sibling[0].copy_(
            padded_next_sibling_flat.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
        )

        verify_input = DFlashDDTreeVerifyInput(
            draft_token=tree_tokens.reshape(-1),
            positions=tree_positions.reshape(-1),
            draft_token_num=padded_q_len,
            child_maps_per_req=[child_maps],
            visibility_per_req=[padded_visibility],
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            next_token_ids_buf=next_token_ids_buf,
            commit_lens_buf=commit_lens_buf,
            accept_index_buf=self._ddtree_accept_index_buf[:bs, :padded_q_len],
        )
        self._profile_mark(profile, "tree_tensors")

        # ====================================================================
        # Stage 2) Target tree verify.
        # ====================================================================
        batch.out_cache_loc = verify_out_cache_loc
        sampling_info = batch.sampling_info

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        # Verify host bound = committed prefix + one verify block (matches draft).
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            verify_host_seq_lens = seq_lens_cpu_backup + block_size
            batch.seq_lens_cpu = verify_host_seq_lens
            batch.seq_lens_sum = int(verify_host_seq_lens.sum())
        elif draft_input.reserved_seq_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        self._profile_mark(profile, "verify_prepare")

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph
        self._profile_mark(profile, "target_verify")

        (
            next_token_ids,
            commit_lens,
            next_target_hidden,
            accept_index,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )
        self._profile_mark(profile, "tree_accept")

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_ddtree_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                accept_index=accept_index,
                commit_lens=commit_lens,
                draft_token_num=padded_q_len,
            )
        self._profile_mark(profile, "mamba_update")

        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        # ====================================================================
        # Stage 3) Materialize the committed path's target hidden into the draft
        # KV cache. The accepted path is a linear sequence of `commit_len`
        # tokens, laid into the `[bs, block_size]` prefix-valid layout and
        # written through the same DFlash prefix-direct path as linear DFlash.
        # ====================================================================
        if next_target_hidden.numel() == 0:
            raise RuntimeError(
                "DFLASH_DDTREE verify produced no committed hidden states."
            )
        hidden_dim = next_target_hidden.shape[-1]
        if int(next_target_hidden.shape[0]) != bs * block_size:
            committed_hidden_2d = torch.zeros(
                (bs, block_size, hidden_dim),
                dtype=next_target_hidden.dtype,
                device=device,
            )
            offset = 0
            for i in range(bs):
                cl = int(commit_lens[i].item())
                committed_hidden_2d[i, :cl] = next_target_hidden[offset : offset + cl]
                offset += cl
            target_hidden_for_draft = committed_hidden_2d.reshape(-1, hidden_dim)
        else:
            target_hidden_for_draft = next_target_hidden

        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=target_hidden_for_draft,
            cache_loc=verify_out_cache_loc,
            cache_loc_2d=verify_out_cache_loc_2d,
            positions=positions,
            commit_lens=commit_lens,
        )
        self._profile_mark(profile, "draft_kv_append")

        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        # The bonus for the next round is the last emitted token per req (the
        # target's prediction past the final accepted node), at
        # [i, commit_lens[i] - 1] -- not the padded stride tail.
        bonus = next_token_ids.view(bs, padded_q_len).gather(
            1,
            (commit_lens.to(torch.int64) - 1).clamp(min=0).unsqueeze(1),
        ).reshape(-1)
        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        self._profile_finish(profile, label="DDTREE", commit_lens=commit_lens)

        if not getattr(self, "_logged_first_verify", False) and self.tp_rank == 0:
            logger.info(
                "DFLASH_DDTREE verify completed. commit_lens=%s",
                commit_lens.tolist(),
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            accept_lens=commit_lens,
            can_run_cuda_graph=bool(can_run_cuda_graph),
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=padded_q_len,
            new_seq_lens=new_seq_lens,
        )

    def _update_target_mamba_state_after_ddtree_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        accept_index: torch.Tensor,
        commit_lens: torch.Tensor,
        draft_token_num: int,
    ) -> None:
        """Write back the path-end Mamba/SSM state (tree position of the last
        accepted node, not commit_len-1) after a tree verify. Mirrors
        DFlashWorkerV2._update_target_mamba_state_after_verify but derives the
        step index from the non-contiguous accepted path.
        """
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return

        bs = int(commit_lens.shape[0])
        req_idx = torch.arange(bs, dtype=torch.int64, device=commit_lens.device)
        accept_indices_offset = torch.arange(
            0,
            bs * int(draft_token_num),
            step=int(draft_token_num),
            dtype=torch.int64,
            device=commit_lens.device,
        )
        last_correct_step_indices = (
            accept_index[req_idx, (commit_lens.to(torch.int64) - 1)]
            - accept_indices_offset
        )
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            seq_lens_post_verify = seq_lens_pre_verify + commit_lens.to(
                seq_lens_pre_verify.dtype
            )
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(
                tracking_point - seq_lens_pre_verify - 1, min=0
            ).to(torch.int64)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            candidate_track_steps = (
                accept_index[req_idx, to_track_ith] - accept_indices_offset
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                candidate_track_steps,
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )
