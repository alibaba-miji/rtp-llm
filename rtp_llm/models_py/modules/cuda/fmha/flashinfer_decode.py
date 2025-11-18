from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.fmha import DECODE_MHA_IMPS, FMHADecodeImplBase
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import (
    FlashInferDecodeOp,
    FusedRopeKVCacheDecodeOp,
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


DECODE_MHA_IMPS.append(FlashInferDecodeImpl)
