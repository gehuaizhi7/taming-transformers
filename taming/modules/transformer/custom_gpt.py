"""
Custom GPT implementation for Taming Transformers
Modify this to create your own transformer architecture
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .mingpt import GPTConfig, CausalSelfAttention, Block


class CrossAttention(nn.Module):
    """Cross-attention layer for incorporating latent vectors z"""
    
    def __init__(self, config, z_dim=None):
        super().__init__()
        if z_dim is None:
            z_dim = config.n_embd
        self.n_embd = config.n_embd
        self.z_dim = z_dim
        
        # Projections for cross-attention
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(z_dim, config.n_embd)
        self.v_proj = nn.Linear(z_dim, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x, z):
        """
        x: input sequence (batch, seq_len, n_embd)
        z: latent vector (batch, z_dim)
        """
        B, T, C = x.size()
        
        # Project queries from x, keys and values from z
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(z).unsqueeze(1)  # (B, 1, n_embd)
        v = self.v_proj(z).unsqueeze(1)  # (B, 1, n_embd)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C))  # (B, T, 1)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Apply attention to values
        y = att @ v  # (B, T, 1) @ (B, 1, n_embd) -> (B, T, n_embd)
        y = self.out_proj(y)
        y = self.resid_drop(y)
        
        return y


class CustomGPTConfig(GPTConfig):
    """Custom GPT configuration"""
    def __init__(self, vocab_size, block_size, **kwargs):
        super().__init__(vocab_size, block_size, **kwargs)
        self.z_dim = kwargs.get('z_dim', self.n_embd)  # Latent vector dimension


class CustomBlock(Block):
    """Custom transformer block with cross-attention to latent vectors z"""
    
    def __init__(self, config, z_dim=None):
        super().__init__(config)
        # Add cross-attention layer
        self.cross_attn = CrossAttention(config, z_dim)
        self.ln_cross = nn.LayerNorm(config.n_embd)
        
    def forward(self, x, z=None):
        # Self-attention
        x_attn_input = self.ln1(x)
        attn_output, present = self.attn(x_attn_input)
        x = x + attn_output
        
        
        # Cross-attention with latent vector z (if provided)
        if z is not None:
            x_cross_input = self.ln_cross(x)
            cross_attn_output = self.cross_attn(x_cross_input, z)
            x = x + cross_attn_output
        
        # MLP
        x = x + self.mlp(self.ln2(x))
        
        return x


class CustomGPT(nn.Module):
    """Custom GPT implementation with latent vectors z"""
    
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., z_dim=None):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size
        self.z_dim = z_dim if z_dim is not None else n_embd
        self.L = n_layer  # L = number of transformer layers

        # Create config object for internal use
        config = CustomGPTConfig(
            vocab_size=vocab_size, 
            block_size=block_size,
            n_layer=n_layer, 
            n_head=n_head, 
            n_embd=n_embd,
            embd_pdrop=embd_pdrop, 
            resid_pdrop=resid_pdrop, 
            attn_pdrop=attn_pdrop,
            z_dim=self.z_dim
        )
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([CustomBlock(config, self.z_dim) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None, z=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Sample L latent vectors z from standard normal distribution if not provided
        if z is None:
            z = torch.randn(b, self.L, self.z_dim, device=device)  # Shape: (batch_size, L, z_dim)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Pass through each transformer block with different z for each layer
        for i, block in enumerate(self.transformer.h):
            z_l = z[:, i, :]  # Get z for layer l: (batch_size, z_dim)
            x = block(x, z_l)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
