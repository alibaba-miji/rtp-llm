"""Adaptive DeepGemm executor that switches between masked and continuous
modes at runtime based on token count."""

import logging
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType

logger = logging.getLogger(__name__)


class AdaptiveDeepGemmExecutor(FusedMoeExpertExecutor):
    """Wraps DeepGemmContinousExecutor and DeepGemmMaskedExecutorV2,
    dispatching at runtime based on token_num vs max_moe_normal_masked_token_num.

    token_num < threshold  → DeepGemmMaskedExecutorV2
    token_num >= threshold → DeepGemmContinousExecutor
    """

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_continous_executor import (
            DeepGemmContinousExecutor,
        )

        DeepGemmContinousExecutor.check_conditions(checker, config)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)
        self.max_moe_normal_masked_token_num = config.max_moe_normal_masked_token_num

        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_continous_executor import (
            DeepGemmContinousExecutor,
        )

        self.continuous_executor = DeepGemmContinousExecutor(
            config, quant_config, weights
        )

        self.masked_executor: Optional[Any] = None
        try:
            from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
                MoeConfigResolver,
            )
            from rtp_llm.models_py.utils.arch import get_sm

            resolver = MoeConfigResolver()
            if resolver.is_bf16(config) and get_sm()[0] >= 9:
                from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor_v2 import (
                    DeepGemmMaskedExecutorV2,
                )

                self.masked_executor = DeepGemmMaskedExecutorV2(
                    config, quant_config, weights
                )
                logger.info(
                    "AdaptiveDeepGemmExecutor: masked executor enabled "
                    "(threshold=%d)",
                    self.max_moe_normal_masked_token_num,
                )
        except Exception:
            logger.warning(
                "AdaptiveDeepGemmExecutor: failed to create masked executor, "
                "falling back to continuous only",
                exc_info=True,
            )

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return self.continuous_executor.topk_ids_dtype

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        token_num = payload.expert_x.shape[0]
        if (
            self.masked_executor is not None
            and token_num <= self.max_moe_normal_masked_token_num
        ):
            return self.masked_executor.execute(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )
        return self.continuous_executor.execute(
            payload,
            activation,
            expert_map,
            a2_scale,
            apply_router_weight_on_input,
            extra_expert_args,
        )
