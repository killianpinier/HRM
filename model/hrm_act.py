import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SwiGLU, RotaryEmbedding, trunc_normal_init_
from layers import Attention, CosSin, rms_norm
from dataclasses import dataclass

from typing import Dict

@dataclass
class HRM_Config:
    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    seq_len: int
    vocab_size: int
    hidden_size: int
    num_heads: int
    pos_encodings: str

    halt_max_steps: int
    halt_exploration_prob: float

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

@dataclass
class HRM_CarryInner:
    z_L: torch.Tensor
    z_H: torch.Tensor

@dataclass
class HRM_Carry:
    inner_carry: HRM_CarryInner

    steps: torch.tensor
    halted: torch.tensor

    current_data: torch.tensor

class HRM_Block(nn.Module):
    def __init__(self, config: HRM_Config):
        super().__init__()

        self.attention_layer = Attention(config.hidden_size, config.hidden_size // config.num_heads, config.num_heads, config.num_heads)
        self.mlp = SwiGLU(config.hidden_size)
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states, cos_sin: CosSin):
        hidden_states = rms_norm(hidden_states + self.attention_layer(hidden_states, cos_sin), variance_epsilon=self.norm_eps)
        return rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
    
class HRM_Module(nn.Module):
    def __init__(self, config: HRM_Config, n_layers: int):
        super().__init__()

        self.layers = nn.ModuleList(HRM_Block(config) for _ in range(n_layers))

    def forward(self, hidden_states, input_injection, **kwargs):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states
    
class HRM_Inner(nn.Module):
    def __init__(self, config: HRM_Config, device: str = None):
        super().__init__()
        self.config = config
        self.device = device or 'cpu'

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.q_head = nn.Linear(self.config.hidden_size, 2)

        self.H_level = HRM_Module(self.config, self.config.H_layers)
        self.L_level = HRM_Module(self.config, self.config.L_layers)

        self.first_token = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size)))

        self.register_buffer('z_H_init', trunc_normal_init_(torch.empty(self.config.hidden_size)))
        self.register_buffer('z_L_init', trunc_normal_init_(torch.empty(self.config.hidden_size)))

    def get_init_carry(self):
        return HRM_CarryInner(
            z_L = self.z_L_init,
            z_H = self.z_H_init,
        )

    def reset_carry(self, reset_flag: torch.tensor, carry: HRM_CarryInner):
        return HRM_CarryInner(
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.z_L_init, carry.z_L),
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.z_H_init, carry.z_H,),
        )

    def input_embeddings(self, input):
        batch_size, seq_len = input.shape
        embedding = self.embed_tokens(input)
        embedding = torch.cat((self.first_token.view(1, 1, -1).expand(batch_size, -1, -1), embedding), dim=1)

        if self.config.pos_encodings == "learned":
            self.positional_enc = nn.Embedding(self.config.seq_len, self.config.hidden_size)
            position_ids = torch.arange(seq_len, device=input.device).expand(batch_size, -1)
            position_embeddings = self.positional_enc(position_ids)  # [batch, seq_len, embed_dim]
            embedding += position_embeddings
        elif self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(self.config.hidden_size // self.config.num_heads, max_position_embeddings=self.config.seq_len + 1, base=self.config.rope_theta, device=self.device)
        else:
            raise ValueError(f"{self.config.pos_encodings} is not supported")
        
        return embedding

    def forward(self, carry: HRM_CarryInner, inputs):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self.input_embeddings(inputs)

        with torch.no_grad():
            z_L, z_H = carry.z_L, carry.z_H

            for cycle in range(1, self.config.H_cycles * self.config.L_cycles - 1):
                input_injection = z_H + input_embeddings
                z_L = self.L_level(z_L, input_injection, **seq_info)

                if cycle % self.config.L_cycles == 0:
                    z_H = self.H_level(z_H, z_L, **seq_info)
        
        assert not z_H.requires_grad and not z_L.requires_grad

        z_L = self.L_level(z_L, input_injection, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        new_carry = HRM_CarryInner(z_L=z_L.detach(), z_H=z_H.detach())
        output = self.lm_head(z_H)[:, 1:]
        q_logits = self.q_head(z_H[:, 0])

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
    
class HRM(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        self.config = config
        self.inner = HRM_Inner(self.config, device=device)

    def initial_carry(self, batch):
        batch_size = batch["inputs"].shape[0]
        return HRM_Carry(
            inner_carry=self.inner.get_init_carry(),
            steps=torch.zeros((batch_size, ), dtype=torch.int16),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    
    def forward(self, carry: HRM_Carry, batch):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data['inputs'])

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                
                halted = halted & (new_steps >= min_halt_steps)

                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data['inputs'])[-1]

                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HRM_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs