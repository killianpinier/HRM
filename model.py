import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        inter = round(expansion * hidden_size * 2 / 3)

        # Project input to 2 * inter for gating + up projection
        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        # Project back down to hidden_size
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)  # split in half
        x = F.silu(gate) * up                             # SwiGLU activation
        return self.down_proj(x)
    

class HRM_Block(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()

        self.attention_layer = nn.MultiheadAttention(hidden_dim, num_heads)
        self.mlp = SwiGLU(hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        attn_output, _ = self.attention_layer(hidden_states, hidden_states, hidden_states, key_padding_mask=key_padding_mask)
        hidden_states = self.norm(hidden_states + attn_output)
        return self.norm(hidden_states + self.mlp(hidden_states))
    
class HRM_Module(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()

        self.layers = nn.ModuleList(HRM_Block(hidden_dim) for _ in range(n_layers))

    def forward(self, hidden_states, input_injection, attention_mask=None):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
    
class HRM(nn.Module):
    def __init__(self, L_cycles, H_cycles, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles
        self.hidden_dim = hidden_dim
        self.H_level = HRM_Module(self.hidden_dim, 4)
        self.L_level = HRM_Module(self.hidden_dim, 4)
        self.input_fnn = nn.Linear(embed_dim, hidden_dim)
        self.output_fnn = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, attention_mask=None):
        input_embeddings = self.input_fnn(x)

        z_L = input_embeddings
        z_H = input_embeddings

        for _ in range(self.H_cycles):
            input_injection = z_H + input_embeddings

            for _ in range(self.L_cycles):
                z_L = self.L_level(z_L, input_injection, attention_mask)
            
            z_H = self.H_level(z_H, z_L, attention_mask)

        y = self.output_fnn(z_H[0])
        return y
    
class TextHRM(nn.Module):
    def __init__(self, L_cycles, H_cycles, vocab_size, embed_dim, hidden_dim, output_dim, max_seq_len=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.hrm = HRM(L_cycles, H_cycles, embed_dim, hidden_dim, output_dim)

    def forward(self, x, attention_mask=None):
        # x = self.embedding(x).transpose(0, 1)
        # return self.hrm(x, attention_mask)

        batch_size, seq_len = x.shape

        # Get token embeddings
        token_embeddings = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # Get position embeddings
        position_ids = torch.arange(seq_len, device=x.device).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)  # [batch, seq_len, embed_dim]
        
        # Add them together
        embeddings = token_embeddings + position_embeddings
        
        # Transpose for the HRM (which expects [seq_len, batch, embed_dim])
        embeddings = embeddings.transpose(0, 1)  # [seq_len, batch, embed_dim]
        return self.hrm(embeddings, attention_mask)