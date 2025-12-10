from dataclasses import dataclass
import torch.nn.functional as F
import torch.nn as nn
import torch


@dataclass
class MicroLLMConfig:
    n_layers: int = 24
    n_embd: int = 1024
    n_head: int = 16
    n_kv_head: int = 4
    attn_drop_p: float = 0.1
    vocab_size: int = 32768
    max_seq_len: int = 1024
    init_std: float = 0.02


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


# swiglu, kv cache, generate fn
class MLPLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.act_fn = nn.ReLU()
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(inputs)))


class AttentionLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, inputs: torch.Tensor, cos_sin: tuple) -> torch.Tensor:
        B, T, C = inputs.shape
        cos, sin = cos_sin

        query = self.q_proj(inputs).view(B, T, self.n_head, self.head_dim)
        key = self.k_proj(inputs).view(B, T, self.n_kv_head, self.head_dim)
        value = self.v_proj(inputs).view(B, T, self.n_kv_head, self.head_dim)

        query, key = apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.config.attn_drop_p if self.training else 0.0,
            is_causal=True,
            enable_gqa=True
        )
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_attn_norm = nn.RMSNorm(config.n_embd)
        self.attn_layer = AttentionLayer(config)
        self.pre_mlp_norm = nn.RMSNorm(config.n_embd)
        self.mlp_layer = MLPLayer(config)

    def forward(self, inputs: torch.Tensor, cos_sin: tuple) -> torch.Tensor:
        x = x + self.attn_layer(self.pre_attn_norm(x), cos_sin)
        x = x + self.mlp_layer(self.pre_mlp_norm(x))
        return x


class MicroLLM(nn.Module):
    def __init__(self, config: MicroLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight
        
        # rope
        rotary_seq_len = config.max_seq_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rope(rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def _precompute_rope(self, max_seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def forward(self, input_ids: torch.Tensor) -> tuple:
        B, T = input_ids.shape
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        
        x = self.wte(input_ids)
        for layer in self.layers:
            x = layer(x, cos_sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
