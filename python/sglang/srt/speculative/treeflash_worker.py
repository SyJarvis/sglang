from __future__ import annotations

import logging

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_ddtree_info import DFlashDDTreeVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.treeflash_utils import (
    build_treeflash_layout,
    build_treeflash_verify_tree,
)
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.triton_ops.dflash import _prepare_dflash_draft_block_unchecked

logger = logging.getLogger(__name__)


class TreeFlashWorker(DFlashWorkerV2):
    """TreeFlash speculative decoding worker.

    MVP scope matches the first DDTree integration style: greedy, batch_size=1,
    fixed padded verify width, and q-head pruning over a static root-inclusive
    tree layout.
    """

    draft_input_cls = DFlashDraftInputV2
    verify_input_cls = DFlashDDTreeVerifyInput
    forward_spec_algorithm = SpeculativeAlgorithm.TREEFLASH
    spec_name = "TREEFLASH"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_tree_size = 1 + int(
            self.server_args.speculative_dflash_tree_budget
        )
        self.tree_layout = build_treeflash_layout(
            block_size=int(self.block_size),
            parent_indices_json=self.server_args.speculative_treeflash_parent_indices,
        )
        if self.tp_rank == 0:
            logger.info(
                "Initialized TREEFLASH worker. block_size=%s, candidate_tree_size=%s, parent_indices=%s",
                self.block_size,
                self.candidate_tree_size,
                self.tree_layout.parent_idx.tolist(),
            )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError("TREEFLASH speculative decoding does not support return_logprob yet.")
        self._validate_phase1_sampling_support(batch)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return super().forward_batch_generation(batch, on_publish)
        if batch.forward_mode.is_idle():
            return super().forward_batch_generation(batch, on_publish)

        if batch.batch_size() != 1:
            raise RuntimeError("TREEFLASH MVP only supports batch_size=1.")
        if batch.has_grammar:
            raise RuntimeError("TREEFLASH MVP does not support grammar constraints.")

        return self._treeflash_decode_round(batch, on_publish)

    def _treeflash_decode_round(
        self,
        batch: ScheduleBatch,
        on_publish,
    ) -> GenerationBatchResult:
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = len(batch.seq_lens)
        device = self.device
        block_size = int(self.block_size)
        profile = self._profile_start()

        draft_input = batch.spec_info
        if draft_input is None:
            draft_input = self.draft_input_cls.create_idle_input(device=device)
            batch.spec_info = draft_input

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "TREEFLASH requires the target model to expose `lm_head` with `weight`."
            )

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

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
                    "TREEFLASH Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )

        if not self._use_triton_prepare_block:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens)
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

        layout_depth = self.tree_layout.depth.to(device=device, dtype=torch.int64)
        positions_2d.copy_(prefix_lens.to(torch.int64).unsqueeze(1) + layout_depth)

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
            spec_algorithm=SpeculativeAlgorithm.TREEFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._profile_mark(profile, "draft_prepare")

        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(forward_batch)
        draft_hidden = draft_out.logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("TREEFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, block_size, -1)
        self._profile_mark(profile, "draft_forward")

        q_logits = getattr(self.draft_model, "last_q_logits", None)
        if q_logits is None:
            raise RuntimeError(
                "TREEFLASH draft model must expose `last_q_logits` from its q-head."
            )
        q_scores = torch.sigmoid(q_logits.view(bs, block_size))[0].to(torch.float32)
        q_scores[0] = 1.0

        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, block_size - 1)
        tree_token_ids = self._draft_block_tokens_buf[:bs]
        tree_token_ids[:, 0].copy_(block_ids[:, 0])
        tree_token_ids[:, 1:].copy_(draft_next)
        self._profile_mark(profile, "draft_sample")

        (
            node_token_ids,
            node_depths,
            child_maps,
            visibility,
            retrieve_next_token_flat,
            retrieve_next_sibling_flat,
        ) = build_treeflash_verify_tree(
            layout=self.tree_layout,
            tree_token_ids=tree_token_ids[0],
            q_scores=q_scores,
            candidate_tree_size=int(self.candidate_tree_size),
            padded_q_len=block_size,
        )
        self._profile_mark(profile, "tree_build")

        real_q_len = 1 + int(node_token_ids.numel())
        tree_tokens = torch.empty((bs, block_size), dtype=torch.long, device=device)
        tree_tokens.copy_(block_ids[:, 0:1])
        if real_q_len > 1:
            tree_tokens[0, 1:real_q_len].copy_(
                node_token_ids.to(device=device, non_blocking=True)
            )

        tree_positions = torch.empty((bs, block_size), dtype=torch.int64, device=device)
        tree_positions.copy_(prefix_lens.to(torch.int64).unsqueeze(1))
        if real_q_len > 1:
            tree_positions[0, 1:real_q_len].copy_(
                prefix_lens[0].to(torch.int64)
                + node_depths.to(device=device, dtype=torch.int64, non_blocking=True)
            )

        retrieve_next_token = (
            retrieve_next_token_flat.to(device=device, dtype=torch.int32, non_blocking=True)
            .view(1, block_size)
            .contiguous()
        )
        retrieve_next_sibling = (
            retrieve_next_sibling_flat.to(device=device, dtype=torch.int32, non_blocking=True)
            .view(1, block_size)
            .contiguous()
        )

        verify_input = self.verify_input_cls(
            draft_token=tree_tokens.reshape(-1),
            positions=tree_positions.reshape(-1),
            draft_token_num=block_size,
            child_maps_per_req=[child_maps],
            visibility_per_req=[visibility],
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
        )

        batch.out_cache_loc = verify_out_cache_loc
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = batch.seq_lens.clone() if need_mamba_verify_commit else None

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
            committed_indices_per_req,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )
        self._profile_mark(profile, "tree_accept")

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_treeflash_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                committed_indices_per_req=committed_indices_per_req,
            )

        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        hidden_dim = next_target_hidden.shape[-1]
        committed_hidden_2d = torch.zeros(
            (bs, block_size, hidden_dim), dtype=next_target_hidden.dtype, device=device
        )
        committed_positions_2d = tree_positions.clone()
        offset = 0
        for i in range(bs):
            cl = int(commit_lens[i].item())
            committed_hidden_2d[i, :cl] = next_target_hidden[offset : offset + cl]
            idx = torch.tensor(
                committed_indices_per_req[i],
                dtype=torch.long,
                device=tree_positions.device,
            )
            committed_positions_2d[i, :cl] = tree_positions[i].index_select(0, idx)
            offset += cl

        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=committed_hidden_2d.reshape(-1, hidden_dim),
            cache_loc=verify_out_cache_loc,
            cache_loc_2d=verify_out_cache_loc_2d,
            positions=committed_positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )

        logits_output.hidden_states = None
        bonus = next_token_ids.view(bs, block_size).gather(
            1,
            (commit_lens.to(torch.int64) - 1).clamp(min=0).unsqueeze(1),
        ).reshape(-1)
        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        self._profile_finish(profile, label="TREEFLASH", commit_lens=commit_lens)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            accept_lens=commit_lens,
            can_run_cuda_graph=bool(can_run_cuda_graph),
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=block_size,
            new_seq_lens=new_seq_lens,
        )

    def _update_target_mamba_state_after_treeflash_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        committed_indices_per_req: list[list[int]],
    ) -> None:
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return
        device = seq_lens_pre_verify.device
        last_correct_step_indices = torch.tensor(
            [indices[-1] for indices in committed_indices_per_req],
            dtype=torch.int64,
            device=device,
        )
        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=None,
            model=self.target_worker.model_runner.model,
        )
