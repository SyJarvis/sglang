# Adapted from the TreeFlash draft reference implementation. TreeFlash
# intentionally has its own SGLang model class so checkpoints can register as
# `TreeFlashDraftModel` without changing DFlash behavior.

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.utils.hf_transformers_utils import get_rope_config


def _treeflash_layer_attention_params(config, layer_id: int) -> Tuple[int, AttentionType]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return -1, AttentionType.ENCODER_ONLY
    layer_type = layer_types[layer_id]
    if layer_type == "full_attention":
        return -1, AttentionType.ENCODER_ONLY
    if layer_type == "sliding_attention":
        return int(getattr(config, "sliding_window")) - 1, AttentionType.DECODER
    raise ValueError(f"Unsupported TreeFlash draft layer type: {layer_type!r}.")


class TreeFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_parallel().tp_size)
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=attention_bias,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            prefix="o_proj",
        )
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
        rope_is_neox_style = bool(
            getattr(config, "rope_is_neox_style", getattr(config, "is_neox_style", True))
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        self.sliding_window_size, self.attn_type = _treeflash_layer_attention_params(
            config, layer_id
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def kv_proj_only(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class TreeFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class TreeFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = TreeFlashAttention(config=config, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = TreeFlashMLP(config=config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class TreeFlashDraftModel(nn.Module):
    """SGLang TreeFlash draft model.

    This is deliberately separate from `DFlashDraftModel`: TreeFlash uses the
    same target-hidden conditioning and target LM-head projection contract, but
    its inference path consumes a q-head to prune a fixed candidate tree before
    target verification.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        del quant_config, prefix
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [TreeFlashDecoderLayer(config=config, layer_id=i) for i in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        self.num_context_features = int(len(target_layer_ids))
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)
        self.q_head = nn.Linear(hidden_size, 1, bias=False)
        self.last_q_logits: Optional[torch.Tensor] = None

        self.use_tree_pos_emb = bool(getattr(config, "use_tree_pos_emb", False))
        self.tree_pos_embd = (
            nn.Embedding(int(getattr(config, "max_tree_size", self.block_size)), hidden_size)
            if self.use_tree_pos_emb
            else None
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is None or "sliding_attention" not in layer_types:
            return None
        return int(getattr(self.config, "sliding_window")) - 1

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "TREEFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}], got {tuple(target_hidden.shape)}."
            )
        return self.hidden_norm(self.fc(target_hidden))

    def _add_tree_position_embeddings(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tree_pos_embd is None or hidden_states.numel() == 0:
            return hidden_states
        block_size = int(self.block_size)
        if block_size <= 0 or hidden_states.shape[0] % block_size != 0:
            return hidden_states
        bs = hidden_states.shape[0] // block_size
        tree_pos = torch.arange(block_size, device=hidden_states.device).repeat(bs)
        return hidden_states + self.tree_pos_embd(tree_pos)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del input_ids, get_embedding, pp_proxy_tensors
        if input_embeds is None:
            raise ValueError(
                "TreeFlashDraftModel requires `input_embeds` from the target embedding."
            )

        hidden_states = self._add_tree_position_embeddings(input_embeds)
        residual: Optional[torch.Tensor] = None
        backbone_hidden: Optional[torch.Tensor] = None

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
            if layer_idx == len(self.layers) - 2:
                backbone_hidden = hidden_states

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if backbone_hidden is None:
            backbone_hidden = hidden_states
        self.last_q_logits = self.q_head(self.norm(backbone_hidden)).squeeze(-1)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = TreeFlashDraftModel
