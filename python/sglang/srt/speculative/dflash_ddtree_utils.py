from __future__ import annotations

import heapq
import time
from typing import List, Tuple

import numpy as np
import torch


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[int],
    List[dict[int, int]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build a DDTree from per-position draft logits.

    Returns non-root node tokens/depths plus root-inclusive parents, child maps,
    a root-inclusive visibility matrix, and EAGLE-style first-child / next-sibling
    tensors (`retrieve_next_token`, `retrieve_next_sibling`) that drive the
    GDN tree-aware verify kernels. The tree construction is CPU-heavy by design
    for the MVP; SGLang integration keeps request concurrency at 1.
    """

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        retrieve_next_token = torch.full((1,), -1, dtype=torch.long)
        retrieve_next_sibling = torch.full((1,), -1, dtype=torch.long)
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            retrieve_next_token,
            retrieve_next_sibling,
        )

    topk = min(int(budget), int(draft_logits.shape[-1]))
    depth_limit = int(draft_logits.shape[0])

    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs = top_logits - log_z
    return build_ddtree_tree_from_topk(top_log_probs, top_token_ids, budget)


def build_ddtree_tree_from_topk(
    top_log_probs: torch.Tensor,
    top_token_ids: torch.Tensor,
    budget: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[int],
    List[dict[int, int]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if budget <= 0 or top_log_probs.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        retrieve_next_token = torch.full((1,), -1, dtype=torch.long)
        retrieve_next_sibling = torch.full((1,), -1, dtype=torch.long)
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            retrieve_next_token,
            retrieve_next_sibling,
        )

    topk = min(int(budget), int(top_log_probs.shape[-1]))
    depth_limit = int(top_log_probs.shape[0])

    top_log_probs_np = top_log_probs[:, :topk].to(
        device="cpu", dtype=torch.float32
    ).numpy()
    top_token_ids_np = top_token_ids[:, :topk].to(
        device="cpu", dtype=torch.long
    ).numpy()

    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids_np = np.empty(int(budget), dtype=np.int64)
    node_depths_np = np.empty(int(budget), dtype=np.int64)
    parents_np = np.empty(int(budget) + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < int(budget):
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(top_log_probs_np[depth - 1, rank])
                + float(top_log_probs_np[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (
                    -sibling_logw,
                    sibling_ranks,
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(
                heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw)
            )

    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    retrieve_next_token_np = np.full(current_length, -1, dtype=np.int64)
    retrieve_next_sibling_np = np.full(current_length, -1, dtype=np.int64)
    prev_sibling_of_parent = np.full(current_length, -1, dtype=np.int64)
    for child in range(1, current_length):
        parent = int(parents_np[child])
        if retrieve_next_token_np[parent] == -1:
            retrieve_next_token_np[parent] = child
        else:
            retrieve_next_sibling_np[prev_sibling_of_parent[parent]] = child
        prev_sibling_of_parent[parent] = child

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()
    retrieve_next_token = torch.from_numpy(retrieve_next_token_np)
    retrieve_next_sibling = torch.from_numpy(retrieve_next_sibling_np)

    return (
        node_token_ids,
        node_depths,
        parents,
        child_maps,
        visibility,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def build_linear_ddtree_tree_from_topk(
    top_token_ids: torch.Tensor,
    budget: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[int],
    List[dict[int, int]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build a top-1 chain using the DDTree verify data format.

    Kept for tests and as a structural reference for the chain-shaped tree case
    (every node has exactly one child). The worker no longer falls back to this
    on hybrid Mamba/attention targets - non-linear tree verify is now wired
    through the GDN tree-aware verify kernels via retrieve_next_token /
    retrieve_next_sibling.
    """

    if budget <= 0 or top_token_ids.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        retrieve_next_token = torch.full((1,), -1, dtype=torch.long)
        retrieve_next_sibling = torch.full((1,), -1, dtype=torch.long)
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            retrieve_next_token,
            retrieve_next_sibling,
        )

    node_count = min(int(budget), int(top_token_ids.shape[0]))
    node_token_ids = top_token_ids[:node_count, 0].to(device="cpu", dtype=torch.long)
    node_depths = torch.arange(1, node_count + 1, dtype=torch.long)
    parents = [-1] + list(range(node_count))
    child_maps: list[dict[int, int]] = [dict() for _ in range(node_count + 1)]
    for i in range(node_count):
        child_maps[i][int(node_token_ids[i].item())] = i + 1
    visibility = torch.tril(torch.ones((node_count + 1, node_count + 1), dtype=torch.bool))
    current_length = node_count + 1
    retrieve_next_token = torch.full((current_length,), -1, dtype=torch.long)
    retrieve_next_sibling = torch.full((current_length,), -1, dtype=torch.long)
    for i in range(node_count):
        retrieve_next_token[i] = i + 1
    return (
        node_token_ids,
        node_depths,
        parents,
        child_maps,
        visibility,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def pad_ddtree_build_outputs(
    visibility: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_q_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad DDTree CPU tensors to a fixed `target_q_len` so cuda graphs can capture.

    Used by the DDTree worker before moving these tensors to GPU. The real tree
    size can be smaller than `1 + tree_budget` whenever the priority-queue search
    drains early; with a dynamic shape the cuda-graph backends would have to
    re-capture per step. We instead pad every output up to `target_q_len`:

    - `visibility[pad, *] = visibility[*, pad] = False` -> queries see no pad
      keys and pad queries see no real keys.
    - `retrieve_next_token[pad] = -1`, `retrieve_next_sibling[pad] = -1` -> the
      GDN tree-aware kernel reads "no child / no sibling" and skips pad rows.

    Pad slots are also absent from `child_maps`, so `follow_verified_tree` will
    never traverse to them at accept time.
    """

    real_q_len = int(retrieve_next_token.shape[0])
    if real_q_len > target_q_len:
        raise ValueError(
            "DDTree pad target shorter than real tree: "
            f"real_q_len={real_q_len}, target_q_len={target_q_len}."
        )
    if retrieve_next_sibling.shape[0] != real_q_len:
        raise ValueError(
            "DDTree retrieve_next_token / retrieve_next_sibling length mismatch: "
            f"{retrieve_next_token.shape} vs {retrieve_next_sibling.shape}."
        )
    if visibility.shape != (real_q_len, real_q_len):
        raise ValueError(
            "DDTree visibility shape inconsistent with retrieve_next_token: "
            f"visibility={tuple(visibility.shape)}, expected {(real_q_len, real_q_len)}."
        )

    padded_visibility = torch.zeros(
        (target_q_len, target_q_len), dtype=torch.bool
    )
    padded_visibility[:real_q_len, :real_q_len].copy_(visibility)

    padded_next_token = torch.full(
        (target_q_len,), -1, dtype=retrieve_next_token.dtype
    )
    padded_next_token[:real_q_len].copy_(retrieve_next_token)

    padded_next_sibling = torch.full(
        (target_q_len,), -1, dtype=retrieve_next_sibling.dtype
    )
    padded_next_sibling[:real_q_len].copy_(retrieve_next_sibling)

    return padded_visibility, padded_next_token, padded_next_sibling


def follow_verified_tree(
    child_maps: list[dict[int, int]], posterior: torch.Tensor
) -> tuple[list[int], int]:
    posterior_tokens = posterior.tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def empty_ddtree_stage_times() -> dict[str, float]:
    # Placeholder for later profiling without changing the public MVP data flow.
    return {
        "draft": 0.0,
        "tree_build": 0.0,
        "verify": 0.0,
        "commit": 0.0,
        "created_at": time.perf_counter(),
    }
