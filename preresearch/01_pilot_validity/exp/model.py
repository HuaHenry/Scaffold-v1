"""GridFormer (plan 00 section 8): 2x2-patch transformer over the 28x56 grid, two 21-way heads, copy bias."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

H, W, N_MAT, N_ACT = 28, 56, 21, 127


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int = 4):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp * d), nn.Linear(mlp * d, d)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(a)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GridFormer(nn.Module):
    def __init__(self, dim: int = 384, depth: int = 8, heads: int = 6, emb: int = 96):
        super().__init__()
        self.cfg = {"dim": dim, "depth": depth, "heads": heads, "emb": emb}
        self.emb = emb
        self.e_fg, self.e_bg, self.e_act = nn.Embedding(N_MAT, emb), nn.Embedding(N_MAT, emb), nn.Embedding(N_ACT, emb)
        self.patch = nn.Linear(4 * emb, dim)
        self.pos = nn.Parameter(torch.zeros(1, (H // 2) * (W // 2), dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 4 * 2 * N_MAT)
        self.a_fg, self.a_bg = nn.Parameter(torch.tensor(3.0)), nn.Parameter(torch.tensor(3.0))

    def forward(self, fg, bg, act):
        B = fg.shape[0]
        x = self.e_fg(fg) + self.e_bg(bg) + self.e_act(act)                      # B,H,W,E
        x = x.view(B, H // 2, 2, W // 2, 2, self.emb).permute(0, 1, 3, 2, 4, 5).reshape(B, -1, 4 * self.emb)
        x = self.patch(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        h = self.head(self.norm(x)).view(B, H // 2, W // 2, 2, 2, 2 * N_MAT)
        h = h.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, 2 * N_MAT)
        lf = h[..., :N_MAT].permute(0, 3, 1, 2)
        lb = h[..., N_MAT:].permute(0, 3, 1, 2)
        lf = lf + self.a_fg * F.one_hot(fg, N_MAT).permute(0, 3, 1, 2).to(lf.dtype)
        lb = lb + self.a_bg * F.one_hot(bg, N_MAT).permute(0, 3, 1, 2).to(lb.dtype)
        return lf, lb

    @torch.no_grad()
    def step(self, fg, bg, act):
        lf, lb = self(fg, bg, act)
        return lf.argmax(1), lb.argmax(1)


def build_model(cfg: dict | None = None) -> GridFormer:
    return GridFormer(**(cfg or {}))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
