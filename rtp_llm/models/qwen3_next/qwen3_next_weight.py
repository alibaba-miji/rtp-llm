import functools
from typing import List

import torch

from rtp_llm.config.gpt_init_model_parameters import HybridAttentionType
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWithSharedWeight,
    SharedMoeConfig,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    stack_,
    stack_moe_w1,
    transpose,
)


def split_0(ts: List[torch.Tensor], part: int):
    dim0, dim1 = ts[0].shape
    if part == 0:
        return ts[0][: dim0 // 2, :]
    else:
        return ts[0][dim0 // 2 :, :]


class Qwen3NextWeight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("model.norm.weight", identity)],
            ),
        ]
        all_layer_weights: List[List[WeightModule]] = []
        for idx in range(self._num_layers):
            layer_weight: List[WeightModule] = []
            layer_weight.append(
                AtomicWeight(
                    W.pre_ln_gamma,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.input_layernorm.weight",
                            identity,
                        )
                    ],
                )
            )
            if (
                self.config.hybrid_attention_config.hybrid_attention_types[idx]
                == HybridAttentionType.LINEAR
            ):
                layer_weight.extend(self._create_linear_attention_weight(idx))
            else:
                layer_weight.extend(self._create_mqa_weight(idx))
            layer_weight.append(
                AtomicWeight(
                    W.post_ln_gamma,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.post_attention_layernorm.weight", identity
                        ),
                    ],
                )
            )
            layer_weight.extend(self._create_ffn_weight(idx))
            all_layer_weights.append(layer_weight)
        return ModelWeightInfo(layer_weights=all_layer_weights, weights=weights)

    def _create_ffn_weight(self, idx: int) -> List[WeightModule]:
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            inter_padding_size=self._inter_padding_size,
        )
        shared_moe_config = SharedMoeConfig(
            expert_num=self.expert_num_,
            inter_padding_size=self._inter_padding_size,
        )
        return [
            MoeWithSharedWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [CkptWeightInfo("model.layers.{i}.mlp.gate.weight", identity)],
                        process_fun=transpose,
                        config=moe_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert.gate_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=transpose,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert.down_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=transpose,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert.up_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=transpose,
                    ),
                    FfnAtomicWeight(
                        W.shared_expert_gate,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert_gate.weight",
                                identity,
                            )
                        ],
                        process_fun=transpose,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                                identity,
                            )
                        ]
                        + [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=stack_moe_w1,
                        config=moe_config,
                    ),
                ],
                config=shared_moe_config,
            )
        ]

    def _create_linear_attention_weight(self, idx: int):
        return [
            AtomicWeight(
                W.linear_attn_qkvz_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.linear_attn.in_proj_qkvz.weight", transpose
                    )
                ],
            ),
            AtomicWeight(
                W.linear_attn_ba_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.linear_attn.in_proj_ba.weight", transpose
                    )
                ],
            ),
            AtomicWeight(
                W.linear_attn_norm_w,
                [CkptWeightInfo("model.layers.{i}.linear_attn.norm.weight", identity)],
            ),
            AtomicWeight(
                W.linear_attn_dt_b,
                [CkptWeightInfo("model.layers.{i}.linear_attn.dt_bias", identity)],
            ),
            AtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.linear_attn.conv1d.weight", identity
                    )
                ],
            ),
            AtomicWeight(
                W.linear_attn_alog,
                [CkptWeightInfo("model.layers.{i}.linear_attn.A_log", identity)],
            ),
            AtomicWeight(
                W.linear_attn_out_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.linear_attn.out_proj.weight", transpose
                    )
                ],
            ),
        ]

    def _create_mqa_weight(self, idx: int):
        return [
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_proj.weight",
                        functools.partial(split_0, part=0),
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.k_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.v_proj.weight",
                        identity,
                    ),
                ],
                process_fun=merge_qkv_hf,
                config=self.attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.o_proj.weight",
                        identity,
                    )
                ],
                process_fun=transpose,
                config=self.attn_config,
            ),
            AtomicWeight(
                W.attn_gate_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_proj.weight",
                        functools.partial(split_0, part=1),
                    )
                ],
                process_fun=transpose,
            ),
            AtomicWeight(
                W.q_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.self_attn.q_norm.weight")],
            ),
            AtomicWeight(
                W.k_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.self_attn.k_norm.weight")],
            ),
        ]
