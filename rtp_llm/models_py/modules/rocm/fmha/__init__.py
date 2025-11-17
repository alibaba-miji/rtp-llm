# Import all ROCm implementations to register them
# Use relative imports to avoid circular import issues
# Export implementations
import rtp_llm.models_py.modules.rocm.fmha.aiter_decode
import rtp_llm.models_py.modules.rocm.fmha.aiter_prefill

from . import aiter_decode  # noqa: F401
from . import aiter_prefill  # noqa: F401
