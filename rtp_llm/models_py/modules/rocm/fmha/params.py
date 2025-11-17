from typing import Optional

import torch

from rtp_llm.ops.compute_ops import ParamsBase


# Pure Python implementation of FMHAParams
class FMHAParams(ParamsBase):
    """Python implementation of FMHAParams for Aiter attention operations."""

    def __init__(
        self,
        input_lengths: torch.Tensor,
        is_prefill: Optional[bool] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        kv_cache_block_id_device: Optional[torch.Tensor] = None,
        enable_cuda_graph: bool = False,
    ):
        super().__init__()

        # Prefill mode
        if is_prefill is not None and is_prefill:
            self.max_seq_len = input_lengths.max().item()
            batch_size = input_lengths.size(0)
            self.cu_seqlens_q = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=input_lengths.device
            )
            self.cu_seqlens_q[1:] = torch.cumsum(input_lengths, 0)
            self.cu_seqlens_k = self.cu_seqlens_q.clone()
            self.max_seqlen_q = self.max_seq_len
            self.max_seqlen_k = self.max_seq_len
            self.seq_lens = None
            self.kv_cache_block_id_device = None
        # Decode mode
        else:
            self.sequence_lengths = sequence_lengths
            self.kv_cache_block_id_device = kv_cache_block_id_device

            if enable_cuda_graph:
                self.max_seq_len = 8192
            else:
                self.max_seq_len = input_lengths.max().item() + 1

            self.max_seqlen_k = self.max_seq_len
            self.max_seqlen_q = 0
            self.cu_seqlens_q = None
            self.cu_seqlens_k = None

            # Create seq_lens on CUDA
            if sequence_lengths is not None:
                self.seq_lens = (sequence_lengths + 1).to(torch.device("cuda"))
            else:
                self.seq_lens = None

    def update(self):
        """Update parameters for CUDA graph execution."""
        if self.seq_lens is not None and self.sequence_lengths is not None:
            self.seq_lens.copy_((self.sequence_lengths + 1).to(torch.device("cuda")))
            self.max_seq_len = 8192

    def check_recycle(self) -> bool:
        """Check whether the params can be recycled automatically."""
        return True
