import torch

from rtp_llm.ops.compute_ops import layernorm


class LayerNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        output = torch.empty_like(hidden_states)
        layernorm(
            output,
            hidden_states,
            self.weight.data,
            self.beta,
            self.variance_epsilon,
        )
        return output
