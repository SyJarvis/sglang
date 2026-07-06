from __future__ import annotations

import json
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TreeFlashLayout:
    parent_idx: torch.Tensor
    depth: torch.Tensor
    visibility: torch.Tensor

    @property
    def block_size(self) -> int:
        return int(self.parent_idx.numel())


def build_treeflash_layout(
    *,
    block_size: int,
    parent_indices_json: str | None,
) -> TreeFlashLayout:
    if parent_indices_json:
        raw = json.loads(parent_indices_json)
        if not isinstance(raw, list):
            raise ValueError("TREEFLASH parent indices must be a JSON list.")
        parent_list = [int(x) for x in raw]
        if len(parent_list) != int(block_size):
            raise ValueError(
                "TREEFLASH parent index length must match speculative_num_draft_tokens. "
                f"Got {len(parent_list)} vs block_size={block_size}."
            )
    else:
        parent_list = [-1] + list(range(int(block_size) - 1))

    if not parent_list or parent_list[0] != -1:
        raise ValueError("TREEFLASH parent indices must start with root parent -1.")

    for idx, parent in enumerate(parent_list[1:], start=1):
        if parent < 0 or parent >= idx:
            raise ValueError(
                "TREEFLASH parent indices must be ancestor-ordered. "
                f"Invalid parent[{idx}]={parent}."
            )

    parent_idx = torch.tensor(parent_list, dtype=torch.long)
    depth = torch.zeros((int(block_size),), dtype=torch.long)
    for idx in range(1, int(block_size)):
        depth[idx] = depth[int(parent_idx[idx].item())] + 1

    visibility = torch.zeros((int(block_size), int(block_size)), dtype=torch.bool)
    for idx in range(int(block_size)):
        cur = idx
        while cur >= 0:
            visibility[idx, cur] = True
            cur = int(parent_idx[cur].item())

    return TreeFlashLayout(parent_idx=parent_idx, depth=depth, visibility=visibility)


def _ancestor_chain(parent_idx: torch.Tensor, node_idx: int) -> list[int]:
    chain = []
    cur = int(node_idx)
    while cur >= 0:
        chain.append(cur)
        cur = int(parent_idx[cur].item())
    return list(reversed(chain))


def pad_treeflash_build_outputs(
    visibility: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_q_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    real_q_len = int(retrieve_next_token.shape[0])
    if real_q_len > int(target_q_len):
        raise ValueError(
            "TREEFLASH pad target shorter than real tree: "
            f"real_q_len={real_q_len}, target_q_len={target_q_len}."
        )
    if retrieve_next_sibling.shape[0] != real_q_len:
        raise ValueError(
            "TREEFLASH retrieve_next_token / retrieve_next_sibling length mismatch: "
            f"{retrieve_next_token.shape} vs {retrieve_next_sibling.shape}."
        )
    if visibility.shape != (real_q_len, real_q_len):
        raise ValueError(
            "TREEFLASH visibility shape inconsistent with retrieve tensors: "
            f"visibility={tuple(visibility.shape)}, expected {(real_q_len, real_q_len)}."
        )

    padded_visibility = torch.zeros(
        (int(target_q_len), int(target_q_len)), dtype=torch.bool
    )
    padded_visibility[:real_q_len, :real_q_len].copy_(visibility)

    padded_next_token = torch.full(
        (int(target_q_len),), -1, dtype=retrieve_next_token.dtype
    )
    padded_next_token[:real_q_len].copy_(retrieve_next_token)

    padded_next_sibling = torch.full(
        (int(target_q_len),), -1, dtype=retrieve_next_sibling.dtype
    )
    padded_next_sibling[:real_q_len].copy_(retrieve_next_sibling)
    return padded_visibility, padded_next_token, padded_next_sibling


def select_treeflash_keep_indices(
    *,
    layout: TreeFlashLayout,
    q_scores: torch.Tensor,
    candidate_tree_size: int,
) -> list[int]:
    """Select an ancestor-closed rooted subset using cumulative q-head scores."""
    budget = max(1, min(int(candidate_tree_size), layout.block_size))
    if budget == 1:
        return [0]

    scores = q_scores.detach().to(device="cpu", dtype=torch.float32).flatten()
    if int(scores.numel()) < layout.block_size:
        raise ValueError(
            "TREEFLASH q-head score length is smaller than layout block size: "
            f"{scores.numel()} < {layout.block_size}."
        )
    scores[0] = 1.0

    path_scores = torch.ones((layout.block_size,), dtype=torch.float32)
    for idx in range(1, layout.block_size):
        parent = int(layout.parent_idx[idx].item())
        path_scores[idx] = path_scores[parent] * scores[idx].clamp(min=0.0, max=1.0)

    ranked = sorted(
        range(1, layout.block_size),
        key=lambda idx: (
            -float(path_scores[idx].item()),
            -int(layout.depth[idx].item()),
            idx,
        ),
    )

    kept: set[int] = {0}
    for idx in ranked:
        chain = _ancestor_chain(layout.parent_idx, idx)
        new_nodes = [node for node in chain if node not in kept]
        if len(kept) + len(new_nodes) > budget:
            continue
        kept.update(new_nodes)
        if len(kept) >= budget:
            break

    return sorted(kept)


def build_treeflash_verify_tree(
    *,
    layout: TreeFlashLayout,
    tree_token_ids: torch.Tensor,
    q_scores: torch.Tensor,
    candidate_tree_size: int,
    padded_q_len: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[dict[int, int]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    keep = select_treeflash_keep_indices(
        layout=layout,
        q_scores=q_scores,
        candidate_tree_size=min(int(candidate_tree_size), int(padded_q_len)),
    )
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep)}
    new_parent = [-1]
    for old_idx in keep[1:]:
        parent = int(layout.parent_idx[old_idx].item())
        new_parent.append(old_to_new[parent])

    q_len = len(keep)
    visibility = torch.zeros((q_len, q_len), dtype=torch.bool)
    for idx in range(q_len):
        cur = idx
        while cur >= 0:
            visibility[idx, cur] = True
            cur = new_parent[cur]

    child_maps: list[dict[int, int]] = [dict() for _ in range(q_len)]
    token_ids_cpu = tree_token_ids.detach().to(device="cpu", dtype=torch.long)
    for new_idx, old_idx in enumerate(keep[1:], start=1):
        parent = new_parent[new_idx]
        token_id = int(token_ids_cpu[old_idx].item())
        child_maps[parent][token_id] = new_idx

    retrieve_next_token = torch.full((q_len,), -1, dtype=torch.long)
    retrieve_next_sibling = torch.full((q_len,), -1, dtype=torch.long)
    prev_sibling: list[int] = [-1 for _ in range(q_len)]
    for child in range(1, q_len):
        parent = new_parent[child]
        if int(retrieve_next_token[parent].item()) == -1:
            retrieve_next_token[parent] = child
        else:
            retrieve_next_sibling[prev_sibling[parent]] = child
        prev_sibling[parent] = child

    padded_visibility, padded_next_token, padded_next_sibling = pad_treeflash_build_outputs(
        visibility,
        retrieve_next_token,
        retrieve_next_sibling,
        int(padded_q_len),
    )

    kept_tensor = torch.tensor(keep, dtype=torch.long)
    node_token_ids = token_ids_cpu.index_select(0, kept_tensor)[1:]
    node_depths = layout.depth.index_select(0, kept_tensor)[1:].to(dtype=torch.long)
    return (
        node_token_ids,
        node_depths,
        child_maps,
        padded_visibility,
        padded_next_token,
        padded_next_sibling,
    )
