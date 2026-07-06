import torch
import triton
import triton.language as tl


@triton.jit
def _ddtree_accept_contig_kernel(
    candidates_ptr,
    target_top1_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    next_token_ids_out_ptr,
    commit_lens_out_ptr,
    accept_index_out_ptr,
    candidates_row_stride,
    target_row_stride,
    retrieve_next_token_row_stride,
    retrieve_next_sibling_row_stride,
    next_token_ids_row_stride,
    commit_lens_stride,
    accept_index_row_stride,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < stride

    candidates_row = candidates_ptr + row * candidates_row_stride
    target_row = target_top1_ptr + row * target_row_stride
    next_token_row = retrieve_next_token_ptr + row * retrieve_next_token_row_stride
    next_sibling_row = retrieve_next_sibling_ptr + row * retrieve_next_sibling_row_stride
    out_row = next_token_ids_out_ptr + row * next_token_ids_row_stride
    accept_row = accept_index_out_ptr + row * accept_index_row_stride

    tl.store(out_row + cols, tl.zeros((BLOCK_SIZE,), tl.int64), mask=row_mask)
    tl.store(accept_row + cols, tl.full((BLOCK_SIZE,), -1, tl.int64), mask=row_mask)

    row_offset = row * stride
    current = tl.full((), 0, tl.int32)
    commit_len = tl.full((), 1, tl.int32)
    tl.store(accept_row, row_offset)

    for _ in range(BLOCK_SIZE - 1):
        in_range = commit_len < stride
        next_token = tl.load(target_row + current, mask=in_range, other=0)
        child = tl.load(next_token_row + current, mask=in_range, other=-1).to(tl.int32)
        found = tl.full((), -1, tl.int32)

        for _ in range(BLOCK_SIZE):
            child_valid = in_range & (child >= 0) & (found < 0)
            child_token = tl.load(candidates_row + child, mask=child_valid, other=-1)
            is_match = child_valid & (child_token == next_token)
            found = tl.where(is_match, child, found)
            child = tl.load(next_sibling_row + child, mask=child_valid, other=-1).to(
                tl.int32
            )

        has_match = found >= 0
        tl.store(out_row + commit_len - 1, next_token, mask=in_range & has_match)
        tl.store(
            accept_row + commit_len,
            row_offset + found.to(tl.int64),
            mask=in_range & has_match,
        )
        current = tl.where(has_match, found, current)
        commit_len += (in_range & has_match).to(tl.int32)

    bonus = tl.load(target_row + current)
    tl.store(out_row + commit_len - 1, bonus)
    tl.store(commit_lens_out_ptr + row * commit_lens_stride, commit_len)


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _compute_ddtree_accept_triton_unchecked(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    next_token_ids_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    accept_index_out: torch.Tensor,
) -> None:
    batch_size, stride = candidates.shape
    if batch_size == 0:
        return

    if not _is_row_major_contiguous_2d(candidates):
        raise ValueError("DDTree Triton accept requires contiguous candidates.")
    if not _is_row_major_contiguous_2d(target_top1):
        raise ValueError("DDTree Triton accept requires contiguous target_top1.")
    if not _is_row_major_contiguous_2d(retrieve_next_token):
        raise ValueError("DDTree Triton accept requires contiguous retrieve_next_token.")
    if not _is_row_major_contiguous_2d(retrieve_next_sibling):
        raise ValueError(
            "DDTree Triton accept requires contiguous retrieve_next_sibling."
        )
    if not _is_row_major_contiguous_2d(next_token_ids_out):
        raise ValueError("DDTree Triton accept requires contiguous next_token_ids_out.")
    if not _is_row_major_contiguous_2d(accept_index_out):
        raise ValueError("DDTree Triton accept requires contiguous accept_index_out.")
    if not commit_lens_out.is_contiguous():
        raise ValueError("DDTree Triton accept requires contiguous commit_lens_out.")

    block = triton.next_power_of_2(stride)
    num_warps = _pick_num_warps(block)
    _ddtree_accept_contig_kernel[(batch_size,)](
        candidates,
        target_top1,
        retrieve_next_token,
        retrieve_next_sibling,
        next_token_ids_out,
        commit_lens_out,
        accept_index_out,
        candidates.stride(0),
        target_top1.stride(0),
        retrieve_next_token.stride(0),
        retrieve_next_sibling.stride(0),
        next_token_ids_out.stride(0),
        commit_lens_out.stride(0),
        accept_index_out.stride(0),
        stride,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
