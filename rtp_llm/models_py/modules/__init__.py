# isort:skip_file
from typing import Union

import torch

# Type alias for quantization dtype
QuantDtype = Union[None, torch.dtype, str]


from rtp_llm.ops.compute_ops import DeviceType, get_device

device_type = get_device().get_device_type()
if device_type == DeviceType.ROCm:
    import rtp_llm.models_py.modules.rocm_registry
    from rtp_llm.models_py.modules.rocm.linear import Linear
    from rtp_llm.models_py.modules.rocm.mlp import FusedSiluActDenseMLP
    from rtp_llm.models_py.modules.rocm.norm import (
        AddBiasResLayerNorm,
        FusedQKRMSNorm,
        RMSNorm,
    )
    from rtp_llm.models_py.modules.rocm.select_topk import SelectTopk
else:
    import rtp_llm.models_py.modules.cuda_registry
    from rtp_llm.models_py.modules.cuda.linear import Linear
    from rtp_llm.models_py.modules.common.mlp import FusedSiluActDenseMLP
    from rtp_llm.models_py.modules.cuda.norm import (
        AddBiasResLayerNorm,
        FusedQKRMSNorm,
        RMSNorm,
    )
    from rtp_llm.models_py.modules.cuda.select_topk import SelectTopk

from rtp_llm.models_py.modules.common.embedding import Embedding
from rtp_llm.models_py.modules.common.kvcache_store import WriteCacheStoreOp

__all__ = [
    "Linear",
    "FusedSiluActDenseMLP",
    "AddBiasResLayerNorm",
    "FusedQKRMSNorm",
    "RMSNorm",
    "SelectTopk",
    "Embedding",
    "WriteCacheStoreOp",
]
