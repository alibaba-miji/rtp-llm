import itertools
from typing import Tuple
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import FusedQKRMSNorm, RMSNorm


class QKRMSNorm(torch.nn.Module):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: float = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.q_norm = RMSNorm(q_weight, eps)
        self.k_norm = RMSNorm(k_weight, eps)
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.q_size = self.head_num * self.size_per_head
        self.kv_size = self.kv_head_num * self.size_per_head
        self.variance_epsilon = eps

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.size_per_head)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.size_per_head)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(self, hidden_states):
        q, k, v = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        output = torch.cat([q, k, v], dim=-1)
        return output


class FusedQKRMSNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HEAD_NUM = [40]
    KV_HEAD_NUM = [40, 8, 4]
    SIZE_PER_HEAD = [128]

    # DTYPES = [torch.bfloat16]
    # NUM_TOKENS = [4096]
    # HEAD_NUM = [40]
    # KV_HEAD_NUM =  [40]
    # SIZE_PER_HEAD = [128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_qk_rmsnorm_test(
        self,
        num_tokens: int,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        dtype: _dtype,
    ):
        torch.manual_seed(0)

        hidden_size = head_num * size_per_head + 2 * kv_head_num * size_per_head

        q_weight = torch.randn(size_per_head, dtype=dtype)
        k_weight = torch.randn(size_per_head, dtype=dtype)

        qkrmsnorm = QKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
        fused_qkrmsnorm = FusedQKRMSNorm(
            q_weight, k_weight, head_num, kv_head_num, size_per_head
        )

        x = torch.randn(num_tokens, hidden_size, dtype=dtype)

        # for _ in range(5):
        #     # out = qkrmsnorm(x)
        #     out = fused_qkrmsnorm(x)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     for _ in range(10):
        #         # out = qkrmsnorm(x)
        #         out = fused_qkrmsnorm(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        self.assertTrue(
            torch.allclose(qkrmsnorm(x), fused_qkrmsnorm(x), atol=1e-2, rtol=1e-2)
        )

    def test_fusedqkrmsnorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HEAD_NUM,
            self.KV_HEAD_NUM,
            self.SIZE_PER_HEAD,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                head_num=params[1],
                kv_head_num=params[2],
                size_per_head=params[3],
                dtype=params[4],
            ):
                self._run_fused_qk_rmsnorm_test(*params)


if __name__ == "__main__":
    main()
