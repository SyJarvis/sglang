from __future__ import annotations

import heapq
from typing import List, Tuple

import numpy as np
import torch


def build_jetspec_tree_from_topk(
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
    """Build a JetSpec accum-logp tree from per-depth top-k rows.

    This mirrors jetspec.tree.baselines.accum_logp._build_from_topk, adapted to
    SGLang's DDTree verify format. `budget` is the max number of non-root nodes;
    JetSpec's standalone builder counts the root in its budget.
    """
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
    max_nodes = int(budget) + 1

    top_log_probs_np = top_log_probs[:, :topk].to(
        device="cpu", dtype=torch.float32
    ).numpy()
    top_token_ids_np = top_token_ids[:, :topk].to(
        device="cpu", dtype=torch.long
    ).numpy()

    tokens_list: list[int] = [0]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    cum_lp_list: list[float] = [0.0]
    child_maps: list[dict[int, int]] = [dict()]

    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and len(tokens_list) < max_nodes:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        depth = depths_list[node_idx]
        if depth >= depth_limit:
            continue
        children_to_add = min(topk, max_nodes - len(tokens_list))
        for rank in range(children_to_add):
            child_token = int(top_token_ids_np[depth, rank])
            child_cum_lp = -neg_cum_lp + float(top_log_probs_np[depth, rank])
            child_idx = len(tokens_list)
            tokens_list.append(child_token)
            parents_list.append(node_idx)
            depths_list.append(depth + 1)
            cum_lp_list.append(child_cum_lp)
            child_maps.append(dict())
            child_maps[node_idx][child_token] = child_idx
            counter += 1
            heapq.heappush(heap, (-child_cum_lp, counter, child_idx))

    num_nodes = len(tokens_list)
    visibility_np = np.zeros((num_nodes, num_nodes), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, num_nodes):
        parent_index = int(parents_list[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    retrieve_next_token_np = np.full(num_nodes, -1, dtype=np.int64)
    retrieve_next_sibling_np = np.full(num_nodes, -1, dtype=np.int64)
    prev_sibling_of_parent = np.full(num_nodes, -1, dtype=np.int64)
    for child in range(1, num_nodes):
        parent = int(parents_list[child])
        if retrieve_next_token_np[parent] == -1:
            retrieve_next_token_np[parent] = child
        else:
            retrieve_next_sibling_np[prev_sibling_of_parent[parent]] = child
        prev_sibling_of_parent[parent] = child

    return (
        torch.tensor(tokens_list[1:], dtype=torch.long),
        torch.tensor(depths_list[1:], dtype=torch.long),
        parents_list,
        child_maps,
        torch.from_numpy(visibility_np),
        torch.from_numpy(retrieve_next_token_np),
        torch.from_numpy(retrieve_next_sibling_np),
    )
