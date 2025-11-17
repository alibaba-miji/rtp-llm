from typing import Dict, List, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.fmha import (
    DECODE_MHA_IMPS,
    DECODE_MLA_IMPS,
    FMHADecodeImplBase,
)
from rtp_llm.models_py.modules.mla import MlaFlashInferDecodeOp, MlaRotaryEmbeddingOp
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (
    FlashInferDecodeOp,
    FusedRopeKVCacheDecodeOp,
    KVCache,
    PyAttentionInputs,
)


class FlashInferDecodeImpl(FMHADecodeImplBase):

    def __init__(
        self, config: GptInitModelParameters, attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            FlashInferDecodeOp(config.gpt_init_params),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )
        self.support_ = self.support_ and (config.use_mla == False)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def support_cuda_graph(self) -> bool:
        return True


class MlaFlashInferDecodeImpl(FMHADecodeImplBase):

    def __init__(
        self,
        config: GptInitModelParameters,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
    ) -> None:
        super().__init__(
            MlaFlashInferDecodeOp(
                config.head_num // config.tp_size,
                config.kv_lora_rank,
                config.rope_head_dim,
                config.nope_head_dim,
                config.seq_size_per_block,
                config.softmax_extra_scale,
                config.use_mla,
                weights,
            ),
            MlaRotaryEmbeddingOp(
                head_size=config.nope_head_dim,
                cos_sin_cache=cos_sin_cache,
                kv_lora_rank=config.kv_lora_rank,
                rope_head_dim=config.rope_head_dim,
                token_per_block=config.seq_size_per_block,
                is_neox_style=False,
            ),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_INFER

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ):
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]
        self.rope_kvcache_impl.forward(
            q_pe, k_pe, compressed_kv, self.rope_params, kv_cache
        )

        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)
        q_nope, q_pe = torch.split(
            q,
            [self.fmha_impl.qk_nope_head_dim, self.fmha_impl.qk_rope_head_dim],
            dim=-1,
        )
        assert self.fmha_impl is not None
        res = self.fmha_impl.forward(q_nope, q_pe, kv_cache, self.fmha_params, layer_id)
        return res


DECODE_MLA_IMPS.append(MlaFlashInferDecodeImpl)
DECODE_MHA_IMPS.append(FlashInferDecodeImpl)
