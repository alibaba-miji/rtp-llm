import sys

sys.path.append(
    "/home/baowending.bwd/RTP-LLM/github-opensource/bazel-github-opensource/external/flash-linear-attention"
)
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import (
    GptInitModelParameters,
    HybridAttentionType,
)
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import FusedSiluActDenseMLP
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_new import chunk_gated_delta_rule_new
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


# TODO slow impl, to fix
class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


# add_unit_offset
class Qwen3NextRMSNorm(RMSNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        weight = weight + 1
        super().__init__(weight, eps)


class Qwen3NextGatedDeltaNetBase(torch.nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.config = config
        self.weights = weights
        # params
        self.head_k_dim: int = config.linear_attention_config.linear_key_head_dim
        self.head_v_dim: int = config.linear_attention_config.linear_value_head_dim
        self.local_num_k_heads: int = (
            config.linear_attention_config.linear_num_key_heads // config.tp_size
        )
        self.local_num_v_heads: int = (
            config.linear_attention_config.linear_num_value_heads // config.tp_size
        )
        self.num_key_value_heads: int = self.local_num_v_heads // self.local_num_k_heads
        self.linear_conv_kernel_dim: int = (
            self.config.linear_attention_config.linear_conv_kernel_dim
        )
        self.ssm_state_size: int = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        self.qkv_size: int = (
            self.head_k_dim * self.local_num_k_heads * 2
            + self.head_v_dim * self.local_num_v_heads
        )
        # weights
        self.conv_weights = weights[W.linear_attn_conv1d_w].squeeze(1)
        self.dt_bias = weights[W.linear_attn_dt_b]
        self.alog = weights[W.linear_attn_alog]

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        conv_states = torch.as_strided(
            kv_cache_tensor,
            (kv_cache_tensor.shape[0], self.head_v_dim - 1, self.qkv_size),
            (kv_cache_tensor.stride()[0], self.qkv_size, 1),
            storage_offset=self.ssm_state_size,
        )
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        ssm_states = torch.as_strided(
            kv_cache_tensor,
            (
                kv_cache_tensor.shape[0],
                self.local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
            ),
            (
                kv_cache_tensor.stride()[0],
                self.head_k_dim * self.head_v_dim,
                self.head_k_dim,
                1,
            ),
            storage_offset=self.ssm_state_size,
        )
        return ssm_states


class Qwen3NextGatedDeltaNetPrefill(Qwen3NextGatedDeltaNetBase):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__(config, weights)

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        out = causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1),
            weight=self.conv_weights,
            bias=None,
            conv_states=conv_states.transpose(1, 2),
            query_start_loc=attn_inputs.cu_seqlens,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths,
        ).transpose(0, 1)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        g = fused_gdn_gating(self.alog, a, self.dt_bias)
        ssm_states = self._get_ssm_states(kv_cache_tensor)
        context_batch_size = attn_inputs.cu_seqlens.shape[0] - 1
        initial_states = torch.empty(
            context_batch_size,
            self.local_num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=mixed_qkv.device,
            dtype=mixed_qkv.dtype,
        )
        load_initial_state_from_block_map(
            attn_inputs.prefix_lengths,
            attn_inputs.kv_cache_block_id_device,
            ssm_states,
            initial_states,
            seq_size_per_block,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        query = query.reshape(
            1, query.shape[0], self.local_num_k_heads, self.head_k_dim
        ).repeat_interleave(self.num_key_value_heads, dim=2)
        key = key.reshape(
            1, key.shape[0], self.local_num_k_heads, self.head_k_dim
        ).repeat_interleave(self.num_key_value_heads, dim=2)
        value = value.reshape(
            1, value.shape[0], self.local_num_v_heads, self.head_v_dim
        )
        g = g.unsqueeze(0)
        b = b.sigmoid().unsqueeze(0)
        cu_seqlens = attn_inputs.cu_seqlens[:2]
        attn_out, h, final_state = chunk_gated_delta_rule_new(
            query,
            key,
            value,
            g,
            b,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        store_ssm_state_to_block_map(
            h,
            final_state.to(h.dtype),
            attn_inputs.prefix_lengths,
            attn_inputs.cu_seqlens,
            attn_inputs.kv_cache_block_id_device,
            ssm_states,
            seq_size_per_block,
            chunk_size=64,
        )
        return attn_out.squeeze_(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for prefill"
        kv_cache_tensor: torch.Tensor = kv_cache.k_cache_base.reshape(
            kv_cache.k_cache_base.shape[0], -1
        )
        mixed_qkv = self._conv1d(
            mixed_qkv, kv_cache_tensor, kv_cache.seq_size_per_block, attn_inputs
        )
        attn_out = self._fla(
            mixed_qkv, b, a, kv_cache_tensor, kv_cache.seq_size_per_block, attn_inputs
        )
        return attn_out


class Qwen3NextGatedDeltaNetDecode(Qwen3NextGatedDeltaNetBase):
    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        decode_cu_seqlens = torch.arange(
            mixed_qkv.shape[0] + 1, device=mixed_qkv.device, dtype=torch.int32
        )
        out = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            self.conv_weights,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths,
            num_accepted_tokens=None,
            query_start_loc=decode_cu_seqlens,
        )
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        seq_len = mixed_qkv.shape[0]
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        beta = b.sigmoid()
        g = fused_gdn_gating(self.alog, a, self.dt_bias)

        g = g.view(1, self.local_num_v_heads, self.local_num_v_heads)
        beta = beta.view(1, self.local_num_v_heads, self.local_num_v_heads)
        query = query.view(1, seq_len, self.local_num_k_heads, self.head_k_dim)
        key = key.view(1, seq_len, self.local_num_k_heads, self.head_k_dim)
        value = value.view(1, seq_len, self.local_num_v_heads, self.head_v_dim)
        ssm_states = self._get_ssm_states(kv_cache_tensor)

        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=None,
            initial_state=ssm_states,
            inplace_final_state=True,
            cu_seqlens=attn_inputs.decode_cu_seqlens,
            block_map=attn_inputs.kv_cache_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for decode"
        kv_cache_tensor: torch.Tensor = kv_cache.k_cache_base.reshape(
            kv_cache.k_cache_base.shape[0], -1
        )
        mixed_qkv = self._conv1d(
            mixed_qkv, kv_cache_tensor, kv_cache.seq_size_per_block, attn_inputs
        )
        attn_out = self._fla(
            mixed_qkv, b, a, kv_cache_tensor, kv_cache.seq_size_per_block, attn_inputs
        )
        return attn_out


class Qwen3NextAttention(CausalAttention):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__(config, weights)
        # maybe fuse gate in qkv_proj later
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.attn_gate_w, W.attn_gate_s, None, config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
        attention_inputs: Optional[PyAttentionInputs],
    ) -> torch.Tensor:
        gate = self.gate(hidden_states)
        attn_out = super().forward(hidden_states, fmha_impl, kv_cache)
        attn_out = attn_out * gate
        return attn_out


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.config = config
        self.weights = weights
        # in_proj_qkvz is bf16 / fp8
        self.in_proj_qkvz = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_qkvz_w, W.linear_attn_qkvz_s, None, config
        )
        # in_proj_ba is bf16
        self.in_proj_ba = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_ba_w, None, None, config
        )
        self.head_k_dim = config.linear_attention_config.linear_key_head_dim
        self.head_v_dim = config.linear_attention_config.linear_value_head_dim
        self.local_num_k_heads = (
            config.linear_attention_config.linear_num_key_heads // config.tp_size
        )
        self.local_num_v_heads = (
            config.linear_attention_config.linear_num_value_heads // config.tp_size
        )
        self.num_key_value_heads = self.local_num_v_heads // self.local_num_k_heads

        self.prefill_gdn = Qwen3NextGatedDeltaNetPrefill(config, weights)
        self.decode_gdn = Qwen3NextGatedDeltaNetDecode(config, weights)
        self.norm = Qwen3NextRMSNormGated(
            weights[W.linear_attn_norm_w], eps=config.layernorm_eps
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights, W.linear_attn_out_w, W.linear_attn_out_s, None, config
        )

    # mixed_qkvz, mixed_ba -> q, k, v, z, b, a
    def fix_query_key_value_ordering(
        self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.local_num_k_heads,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim) * self.num_key_value_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.local_num_k_heads,
            2 * self.num_key_value_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_key_value_heads * self.head_v_dim),
            (self.num_key_value_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_key_value_heads,
            self.num_key_value_heads,
        ]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # reshape to [token, v_head_num, v_head_dim]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.local_num_v_heads)
        a = a.reshape(a.size(0), self.local_num_v_heads)

        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
        attention_inputs: Optional[PyAttentionInputs],
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for prefill"
        assert attention_inputs is not None, "attention_inputs is required for prefill"
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        # [token, head, dim] -> [token, head * dim]
        query, key, value = (x.reshape(x.shape[0], -1) for x in (query, key, value))
        # TODO bad performance here
        mixed_qkv = torch.cat((query, key, value), dim=-1)
        if attention_inputs.is_prefill:
            attn_output = self.prefill_gdn(mixed_qkv, b, a, attention_inputs, kv_cache)
        else:
            attn_output = self.decode_gdn(mixed_qkv, b, a, attention_inputs, kv_cache)
        attn_output = self.norm(attn_output, z).reshape(attn_output.shape[0], -1)
        return self.out_proj(attn_output)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = Qwen3NextGatedDeltaNet(config, weights)
        else:
            self.self_attn = Qwen3NextAttention(config, weights)
        self.mlp = GenericMoeLayer(config, weights)

        self.input_layernorm = Qwen3NextRMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            attention_inputs=attention_inputs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3NextModel(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [
                Qwen3NextDecoderLayer(config, weights.weights[idx], idx)
                for idx in range(self.layer_num)
            ]
        )
        self.norm = Qwen3NextRMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        attention_inputs.prefix_lengths = attention_inputs.prefix_lengths.cuda()
        attention_inputs.sequence_lengths = attention_inputs.sequence_lengths.cuda()
        fmha_impl = self.get_fmha_impl(attention_inputs)
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=attention_inputs,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states)
