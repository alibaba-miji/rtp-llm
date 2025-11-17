import logging
from typing import Optional

import aiter
import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.fmha import DECODE_MHA_IMPS, FMHADecodeImplBase
from rtp_llm.models_py.modules.rocm.fmha.params import FMHAParams
from rtp_llm.ops.compute_ops import FusedRopeKVCacheDecodeOp, KVCache, PyAttentionInputs


class AiterDecodeAttnOp:
    def __init__(self, config: GptInitModelParameters):
        self.head_num = config.head_num // config.tp_size
        self.head_dim = config.size_per_head
        self.head_num_kv = config.head_num_kv // config.tp_size
        self.kv_cache_data_type = config.kv_cache_data_type
        self.use_asm_pa = config.hw_kernel_config.use_asm_pa
        self.enable_cuda_graph = (
            config.gpt_init_params.hw_kernel_config.enable_cuda_graph
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        # Create decode parameters using pure Python implementation
        self.fmha_params = FMHAParams(
            input_lengths=attn_inputs.input_lengths,
            is_prefill=False,
            sequence_lengths=attn_inputs.sequence_lengths,
            kv_cache_block_id_device=attn_inputs.kv_cache_block_id_device,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        return self.fmha_params

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[KVCache], fmha_params
    ) -> torch.Tensor:
        seq_lens = fmha_params.seq_lens
        key_cache = kv_cache.k_cache_base
        value_cache = kv_cache.v_cache_base

        block_tables_id_device = fmha_params.kv_cache_block_id_device
        max_num_blocks = block_tables_id_device.shape[1]
        # for now not support fp8
        if self.use_asm_pa:
            output = aiter.pa_fwd_asm(
                query,  # [num_seqs, num_heads, head_size]
                key_cache,  # [num_blocks, num_kv_heads, block_size, head_size/x, x]
                value_cache,  # [num_blocks, num_kv_heads, block_size, head_size/x, x]
                block_tables_id_device,
                seq_lens,
                max_num_blocks,
            )
        else:
            max_seq_len = fmha_params.max_seq_len
            scale = 1.0 / (self.head_dim**0.5)
            alibi_slopes = None
            k_scale = (
                kv_cache.k_scale_base
                if kv_cache and kv_cache.k_scale_base is not None
                else torch.tensor(1.0, device=query.device, dtype=query.dtype)
            )
            v_scale = (
                kv_cache.v_scale_base
                if kv_cache and kv_cache.v_scale_base is not None
                else torch.tensor(1.0, device=query.device, dtype=query.dtype)
            )
            num_kv_heads = self.head_num_kv
            num_seqs, num_heads, head_size = query.shape
            block_size = value_cache.shape[2]
            _PARTITION_SIZE_ROCM = 256

            # init output
            output = torch.empty_like(query)

            max_num_partitions = (
                max_seq_len + _PARTITION_SIZE_ROCM - 1
            ) // _PARTITION_SIZE_ROCM
            assert _PARTITION_SIZE_ROCM % block_size == 0
            # init tmp_output
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )

            # init exp_sums
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            fp8_out_scale = None
            cpa_fp8_out = False
            # init max_logits
            max_logits = torch.ones_like(exp_sums)

            kv_cache_dtype = "auto"
            # key_cache_reshaped = key_cache.permute(0, 1, 3, 2)
            # value_cache_reshaped = value_cache.permute(0, 1, 3, 2)

            aiter.paged_attention_rocm(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                float(scale),
                block_tables_id_device,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,  # kv_cache_dtype
                k_scale,
                v_scale,
                fp8_out_scale if cpa_fp8_out else None,
                _PARTITION_SIZE_ROCM,
            )

        output_reshaped = output.view(output.shape[0], -1)
        return output_reshaped


class AiterDecodeImpl(FMHADecodeImplBase):
    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            AiterDecodeAttnOp(config),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )


DECODE_MHA_IMPS.append(AiterDecodeImpl)
