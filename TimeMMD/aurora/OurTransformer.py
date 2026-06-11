import torch
import torch.nn as nn
import torch.nn.functional as F
from .sparse_attention import TopRowSparseAttention
from .QueryBudgetController import AdaptiveBudgetController
from .CrossAttnBudget import PruneRatioEMA
from .skeinformer import SkeinAttention
import numpy as np
import pandas as pd
import os


class SampleAttention(nn.Module):
    def __init__(self, embed_dim, nhead, dropout, prune_q=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.prune_q = prune_q

        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"

        if self.prune_q:
            self.q_proj = nn.Linear(embed_dim, embed_dim + self.nhead)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.budget = AdaptiveBudgetController(ratio_init=0.105)
        self.cross_attn = TopRowSparseAttention(index_type='IVFFlat', budget_ctrl=self.budget,
                                                query_filter=True, dropout=dropout, output_attention=False)

    def forward(self, x, y, prune_ratio=None):
        B, M, D1 = x.shape  # batch, seq_len, embed_dim
        _, N, _ = y.shape

        Q = self.q_proj(x)  # (B, L, D)
        K = self.k_proj(y)
        V = self.v_proj(y)

        K = K.view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        # qselect
        if self.prune_q:
            Q, gate_score = torch.split(Q, [self.embed_dim, self.nhead], dim=-1)
            gate = torch.sigmoid(gate_score).view(B, M, self.nhead, 1)
            Q = Q.view(B, M, self.nhead, -1).transpose(1, 2)
            idx = torch.arange(M, device=x.device)[None, None, :].expand(B, self.nhead, M)

            if prune_ratio > 0:
                with torch.no_grad():
                    gate_p = gate.transpose(1, 2).reshape(B * self.nhead, M)  # (B*H, M)
                    keep = M - int(prune_ratio * M)
                    idx = torch.topk(gate_p, k=keep, dim=1, largest=True).indices  # (B*H, keep)
                    idx = idx.view(B, self.nhead, -1)
                Q = torch.gather(Q, 2, idx.unsqueeze(-1).repeat(1, 1, 1, self.head_dim))

            out, K_sel = self.cross_attn(Q, K, V)
            V_sum = V.mean(dim=2)
            attn_output = V_sum.unsqueeze(-2).expand(B, self.nhead, M, V_sum.shape[-1]).clone()
            attn_output.scatter_(2, idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim), out.type_as(attn_output))
            attn_output = attn_output * gate.transpose(1, 2)  # (B,H,M,hd)
            attn_output = attn_output.transpose(1, 2).reshape(B, M, -1)
            out = self.out_proj(attn_output)

            return out, gate

        else:
            Q = Q.view(B, M, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, L, head_dim)
            out = self.cross_attn(Q, K, V)
            attn_output = out.transpose(1, 2).contiguous().view(B, M, D1)
            out = self.out_proj(attn_output)

            return out


class SimpleCrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        init_ratio: float = 0.4,
        prune_q: bool = True,
        warmup_steps: int = 30,
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.activation = F.gelu
        self.cross_attn = SampleAttention(embed_dim=d_model, nhead=nhead, dropout=dropout, prune_q=prune_q)
        self.ratio_ema = PruneRatioEMA()
        self.register_buffer("_global_ratio", torch.tensor(init_ratio), persistent=True)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=True)
        self._ratio_inited = False
        self.warmup_steps = warmup_steps
        self.prune_q = prune_q

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ):

        residual = tgt
        tgt_norm = self.norm1(tgt)

        if self.prune_q:
            use_prune = (not self.training) or (self._step.item() >= self.warmup_steps)

            if use_prune:
                prune_ratio_to_use = self._global_ratio.to(device=tgt.device, dtype=tgt.dtype)
            else:
                prune_ratio_to_use = tgt.new_tensor(0.0)

            attn_output, _ = self.cross_attn(tgt_norm, memory, prune_ratio_to_use)

            if self.training:
                with torch.no_grad():
                    batch_ratio = self.ratio_ema.batch_prune_ratio(tgt_norm, memory, attn_output)
                    if not self._ratio_inited:
                        new_ratio = batch_ratio
                        self._ratio_inited = True
                    else:
                        new_ratio = self.ratio_ema.beta * self._global_ratio + (1.0 - self.ratio_ema.beta) * batch_ratio

                    self._global_ratio.copy_(new_ratio.clamp(0.0, 0.5).to(self._global_ratio.device))
                    self._step.add_(1)
        else:
            attn_output = self.cross_attn(tgt_norm, memory)

        tgt = residual + self.dropout(attn_output)

        residual = tgt
        tgt_norm = self.norm2(tgt)

        ffn_output = self.linear2(
            self.dropout_ffn(self.activation(self.linear1(tgt_norm)))
        )

        tgt = residual + self.dropout(ffn_output)

        return tgt


class SimpleCrossTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [encoder_layer for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


def main(self, config, target_vision_tokens, patch_features):
    encoder_layer = SimpleCrossTransformerEncoderLayer(
        d_model=config.hidden_size,
        nhead=config.num_attention_heads,
        dim_feedforward=config.intermediate_size,
        dropout=config.dropout_rate,
        prune_q=config.prune_q,
    )

    self.cross_vision = SimpleCrossTransformerEncoder(
        encoder_layer,
        num_layers=config.num_vision_cross_layers,
        norm=nn.LayerNorm(config.hidden_size),
    )

    output_tokens = self.cross_vision(
        target_vision_tokens,
        patch_features,
    )

