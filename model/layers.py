import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import apply_rotary_pos_emb
from typing import Tuple


CosSin = Tuple[torch.tensor, torch.tensor]

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qkv_num_heads = self.num_heads + 2*self.num_key_value_heads

        self.qkv_proj = nn.Linear(self.hidden_size, self.head_dim * self.qkv_num_heads )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def forward(self, hidden_state, cos_sin: CosSin):
        batch_size, seq_len, _ = hidden_state.shape

        qkv = self.qkv_proj(hidden_state)
        qkv = qkv.view(batch_size, seq_len, self.qkv_num_heads, self.head_dim)
        q = qkv[:, :, :self.num_heads, :]
        k = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_heads != self.num_key_value_heads:
            group_size = self.num_heads // self.num_key_value_heads
            k = k.unsqueeze(2).repeat(1, 1, group_size, 1, 1).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.unsqueeze(2).repeat(1, 1, group_size, 1, 1).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)