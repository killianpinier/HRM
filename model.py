import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, hidden_states):
        attn_output, _ = self.attention_layer(hidden_states, hidden_states, hidden_states)
        hidden_states = self.norm(hidden_states + attn_output)
        return self.norm(hidden_states + self.mlp(hidden_states))
    
class HRM_Module(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super().__init__()

        self.layers = nn.ModuleList(HRM_Block(hidden_dim) for _ in range(n_layers))

    def forward(self, hidden_states, input_injection):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
    
class HRM(nn.Module):
    def __init__(self, L_cycles, H_cycles, embed_dim, hidden_dim):
        super().__init__()
        self.L_cycles = L_cycles
        self.H_cycles = H_cycles
        self.hidden_dim = hidden_dim
        self.H_level = HRM_Module(self.hidden_dim, 4)
        self.L_level = HRM_Module(self.hidden_dim, 4)
        self.input_fnn = nn.Linear(embed_dim, hidden_dim)
        self.output_fnn = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        input_embeddings = self.input_fnn(x)
        z_L = input_embeddings
        z_H = input_embeddings

        for _ in range(self.H_cycles):
            input_injection = z_H + input_embeddings

            for _ in range(self.L_cycles):
                z_L = self.L_level(z_L, input_injection)
            
            z_H = self.H_level(z_H, z_L)

        return self.output_fnn(z_H)
    

def main():
    model = HRM(4, 4, 16, 32)

if __name__ == "__main__":
    main()