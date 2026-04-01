import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


def kernel_RS_SM1(X1, X2=None, X2_accu=False, random_sign=None):
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...nmd', X1, X2)
        result = torch.exp(product)
        result = torch.einsum('bhnmd,...bmd->...bhnd', result, random_sign)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)
    return result


class SkeinAttention(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.kernel_fn = kernel_RS_SM1
        self.factor = factor

    @torch.no_grad()
    def uniform_sketching(self, non_padding_num, nb_rows, nb_columns, device):
        # non_padding_num: (b,) long
        S = torch.rand(nb_rows, nb_columns, device=device)
        S = torch.einsum("b,md->bmd", non_padding_num, S).long()  # bmd in [0, non_padding_num)
        random_sign = torch.ones(S.shape, device=device)
        random_sign = torch.einsum(
            'bmd,b->bmd',
            random_sign,
            torch.sqrt(non_padding_num / (nb_rows * nb_columns))
        )
        return S, random_sign

    @torch.no_grad()
    def importance_sketching(self, prob, nb_rows, nb_columns):
        # prob: (b,h,n)
        B, H, n = prob.shape
        sample_shape = (B, H, nb_rows, nb_columns)

        prob2 = prob.reshape(B * H, n)
        w = torch.einsum('...n,...->...n', prob2, 1 / prob2.sum(-1))
        S = torch.multinomial(w, nb_rows * nb_columns, replacement=True).reshape(sample_shape)

        w = w.reshape(B, H, -1)[
            torch.arange(B)[:, None, None, None],
            torch.arange(H)[None, :, None, None],
            S
        ]
        random_sign = torch.ones(S.shape, device=prob.device) / torch.sqrt(w * nb_rows * nb_columns)
        return S, random_sign

    def forward(self, q, k, v):
        """
        Cross-attention, 无 mask.
        q: (b,h,nq,p)
        k: (b,h,nk,p)
        v: (b,h,nk,p)
        return: (b,h,nq,p)
        """
        b, h, nq, p = q.shape
        _, _, nk, pk = k.shape
        assert pk == p, f"head_dim mismatch: q:{p}, k:{pk}"
        assert v.shape[:3] == (b, h, nk), f"v shape must be (b,h,nk,p), got {v.shape}"

        device = q.device
        data_normalizer = (p ** -0.25)

        q = q * data_normalizer
        k = k * data_normalizer

        non_padding_num = torch.full((b,), nk, device=device, dtype=torch.float32)
        nb_features = int(self.factor * math.log(nk))
        nb_features = min(nb_features, nk - 1)

        S0, rs0 = self.uniform_sketching(non_padding_num, 1, nb_features, device=device)  # (b,1,d)
        T0 = torch.rand(1, nb_features, device=device)
        T0 = torch.einsum("b,md->bmd", torch.full((b,), nq, device=device, dtype=torch.float32), T0).long()  # (b,1,d)

        QS0 = q.transpose(1, 2)[torch.arange(b)[:, None, None], T0].permute(0, 3, 1, 2, 4)  # bhmdp (m=1)
        ATS0 = self.kernel_fn(k, QS0, True, rs0).reshape(b, h, nk, -1)  # bhkd (k=key length)
        D_inv0_partial = 1 / ATS0.sum(-2)
        Dinv_S0TA = torch.einsum('...d,...kd->...dk', D_inv0_partial, ATS0)
        out0 = torch.matmul(Dinv_S0TA, v)  # (d,nk)@(nk,p)->(d,p)

        prob_AV = torch.sqrt((Dinv_S0TA * Dinv_S0TA).sum(-2) * (v * v).sum(-1))  # (b,h,nk)

        S1, rs1 = self.importance_sketching(prob=prob_AV, nb_rows=3, nb_columns=nb_features)  # (b,h,m,d) int(self.factor * math.log(nq)) S1 K index
        #print('s1',S1.shape)
        S1TV = v[
            torch.arange(b)[:, None, None, None],
            torch.arange(h)[None, :, None, None],
            S1
        ]  # (b,h,m,d,p)

        K1 = k[
            torch.arange(b)[:, None, None, None],
            torch.arange(h)[None, :, None, None],
            S1
        ]  # (b,h,m,d,p)
        qK1 = torch.einsum('...np,...mdp->...nmd', q, K1)

        AS1 = torch.exp(qK1)  # (b,h,nq,m,d)

        AV1 = torch.einsum('...nmd,...mdp->...np', AS1, S1TV)  # (b,h,nq,p)
        A1_sum = AS1.reshape(b, h, nq, -1).sum(-1)  # (b,h,nq)

        model_column = torch.exp(qK1.reshape(b, h, nq, -1).mean(-1))  # (b,h,nq)

        V_sum = torch.einsum("...n,...p->...np", model_column, v.sum(-2))  # (b,h,nq,p)
        V1_sum = torch.einsum(
            "...n,...p->...np",
            model_column,
            S1TV.reshape(b, h, -1, p).sum(-2)
        )  # (b,h,nq,p)

        D1 = A1_sum + torch.einsum('bhn,b->bhn', model_column, non_padding_num - nb_features)  # (b,h,nq)

        out1 = AV1 + (V_sum - V1_sum)  # (b,h,nq,p)
        out1 = torch.einsum('...n,...np->...np', 1 / D1, out1)

        out1 = out1.transpose(1, 2)  # (b,nq,h,p)
        out1[torch.arange(b)[:, None], T0.reshape(b, -1)] = out0.transpose(1, 2)  # (b,d,h,p) -> 对齐
        out1 = out1.transpose(1, 2)  # (b,h,nq,p)




        return out1


class skeinCrossTransformerEncoderLayer(nn.Module):
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
        self.cross_attn = SkeinAttention(factor=30)

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

        queries = self.query_projection(tgt_norm).view(B, L, H, -1).transpose(1, 2)
        keys = self.key_projection(memory).view(B, S, H, -1).transpose(1, 2)
        values = self.value_projection(memory).view(B, S, H, -1).transpose(1, 2)

        out = self.cross_attn(queries, keys, values)
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


class skeinCrossTransformerEncoder(nn.Module):
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



if __name__ == "__main__":
    depth = 16
    head = 8
    rows = 4096
    rows1 = 4096
    cols = 32
    Q = -1 + 2 * torch.rand(depth, head, rows, cols)
    K = -1 + 2 * torch.rand(depth, head, rows1, cols)
    V = -1 + 2 * torch.rand(depth, head, rows1, cols)
    torch.cuda.synchronize()
    start_time = time.time()
    attn = SkeinAttention(factor=30)
    out = attn(Q, K, V)
    torch.cuda.synchronize()
    end_time = time.time() - start_time
    print('time study:', end_time)

    '''original_attention = torch.matmul(Q, K.transpose(-2, -1))
    original_attn = torch.softmax(original_attention, dim=-1)
    context = torch.matmul(original_attn, V)
    mse_loss = nn.MSELoss()
    skeinformer_loss = mse_loss(out, context)
    print(f"loss:{skeinformer_loss:.10f}")'''
