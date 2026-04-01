import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt

import os


class ProbAttention(nn.Module):
    def __init__(self, factor=30, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()

        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.contiguous(), attn


class InformerCrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        d_keys = (d_model // nhead)
        d_values = (d_model // nhead)

        self.query_projection = nn.Linear(d_model, d_keys * nhead)
        self.key_projection = nn.Linear(d_model, d_keys * nhead)
        self.value_projection = nn.Linear(d_model, d_values * nhead)
        self.out_projection = nn.Linear(d_values * nhead, d_model)
        self.n_heads = nhead

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.activation = F.gelu
        self.cross_attn = ProbAttention(attention_dropout=dropout, output_attention=False)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ):

        B, L, _ = tgt.shape
        _, S, _ = memory.shape
        H = self.n_heads

        residual = tgt
        tgt_norm = self.norm1(tgt)

        queries = self.query_projection(tgt_norm).view(B, L, H, -1)
        keys = self.key_projection(memory).view(B, S, H, -1)
        values = self.value_projection(memory).view(B, S, H, -1)

        out, attn = self.cross_attn(queries, keys, values)
        out = out.view(B, L, -1)

        attn_output = self.out_projection(out)
        tgt = residual + self.dropout(attn_output)

        residual = tgt
        tgt_norm = self.norm2(tgt)

        ffn_output = self.linear2(
            self.dropout_ffn(self.activation(self.linear1(tgt_norm)))
        )

        tgt = residual + self.dropout(ffn_output)

        return tgt


class InformerCrossTransformerEncoder(nn.Module):
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
    encoder_layer = InformerCrossTransformerEncoderLayer(
        d_model=config.hidden_size,
        nhead=config.num_attention_heads,
        dim_feedforward=config.intermediate_size,
        dropout=config.dropout_rate,
    )

    self.cross_vision = InformerCrossTransformerEncoder(
        encoder_layer,
        num_layers=config.num_vision_cross_layers,
        norm=nn.LayerNorm(config.hidden_size),
    )

    output_tokens = self.cross_vision(
        target_vision_tokens,
        patch_features,
    )