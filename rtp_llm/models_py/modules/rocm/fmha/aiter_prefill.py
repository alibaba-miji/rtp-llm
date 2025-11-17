import logging

import aiter
import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.fmha import PREFILL_MHA_IMPS, FMHAPrefillImplBase
from rtp_llm.models_py.modules.rocm.fmha.params import FMHAParams
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import FusedRopeKVCachePrefillOp, PyAttentionInputs


class AiterPrefillAttnOp:
    def __init__(self, config: GptInitModelParameters):
        self.head_num = config.head_num // config.tp_size
        self.head_dim = config.size_per_head
        self.head_num_kv = config.head_num_kv // config.tp_size
        self.kv_cache_data_type = config.kv_cache_data_type

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create prefill parameters using pure Python implementation
        self.fmha_params = FMHAParams(
            input_lengths=attn_inputs.input_lengths, is_prefill=True
        )
        return self.fmha_params

    def advanced_qkv_split(self, qkv, head_num, head_num_kv, size_per_head):
        token_num = qkv.shape[0]
        qkv_reshaped = qkv.reshape(token_num, head_num + 2 * head_num_kv, size_per_head)
        q = qkv_reshaped[:, :head_num, :]
        k = qkv_reshaped[:, head_num : head_num + head_num_kv, :]
        v = qkv_reshaped[:, head_num + head_num_kv : head_num + 2 * head_num_kv, :]
        return q, k, v

    def forward(self, qkv, kv_cache, fmha_params):
        cu_seqlens_q = fmha_params.cu_seqlens_q.to(qkv.device)
        cu_seqlens_k = fmha_params.cu_seqlens_k.to(qkv.device)
        max_seqlen_q = fmha_params.max_seqlen_q
        max_seqlen_k = fmha_params.max_seqlen_k

        q_tensor, k_tensor, v_tensor = self.advanced_qkv_split(
            qkv, self.head_num, self.head_num_kv, self.head_dim
        )
        res = aiter.flash_attn_varlen_func(
            q_tensor,  # Query张量: (total_q, nheads, headdim_q) - 批次中所有query token的总数
            k_tensor,  # Key张量: (total_k, nheads_k, headdim_q) - 批次中所有key token的总数
            v_tensor,  # Value张量: (total_k, nheads_k, headdim_v) - 批次中所有value token的总数
            cu_seqlens_q,  # Query累积序列长度: (batch_size + 1,) dtype=int32 - 用于索引q张量
            cu_seqlens_k,  # Key累积序列长度: (batch_size + 1,) dtype=int32 - 用于索引k/v张量
            max_seqlen_q,  # 批次中最大query序列长度
            max_seqlen_k,  # 批次中最大key序列长度
            dropout_p=0.0,  # Dropout概率 - 评估时应设为0.0
            causal=True,  # 因果注意力掩码 - 用于自回归建模，每个位置只能关注自己和之前的位置
        )
        token_num = res.shape[0]
        final_result = res.reshape(token_num, self.head_num * self.head_dim)
        return final_result


class AiterPrefillImpl(FMHAPrefillImplBase):
    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterPrefillAttnOp(config),
            FusedRopeKVCachePrefillOp(config.gpt_init_params),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.AITER_PREFILL


PREFILL_MHA_IMPS.append(AiterPrefillImpl)
