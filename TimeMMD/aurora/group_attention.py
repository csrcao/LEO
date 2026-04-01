import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_u_and_upart(L_Q: int, L_K: int, factor: int):
    # c*ceil(log(L))
    # safety: at least 1
    u = max(1, min(factor * math.ceil(math.log(max(L_Q, 2))), L_Q))
    U_part = max(1, min(factor * math.ceil(math.log(max(L_K, 2))), L_K))
    return u, U_part


def group_attention_op(Q, V, R, BLG, CNT, eps=1e-9):
    """
    Q:   (B,H,LQ,d)
    V:   (B,H,LK,dv)
    R:   (B,H,N,d)     group representative keys (centroids)
    BLG: (B,H,LK)      group id for each key/value token
    CNT: (B,H,N)       counts per group
    Return:
      O: (B,H,LQ,dv)
    """
    B, H, LQ, d = Q.shape
    _, _, LK, dv = V.shape
    _, _, N, _ = R.shape

    # (1) aggregate values per group: eV_g = sum_{k in group g} V_k
    eV = torch.zeros((B, H, N, dv), device=Q.device, dtype=Q.dtype)
    idx = BLG.unsqueeze(-1).expand(B, H, LK, dv)
    eV.scatter_add_(dim=2, index=idx, src=V)

    # (2) scores between each query and each group rep key: (B,H,LQ,N)
    scores = torch.einsum("bhqd,bhnd->bhqn", Q, R) / math.sqrt(d)

    # (3) group softmax with CNT: a ∝ exp(score) * CNT
    max_scores = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_scores)
    weights = exp_scores * CNT.unsqueeze(2).to(dtype=Q.dtype)  # (B,H,LQ,N)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    attn = weights / denom

    # (4) output: (B,H,LQ,dv)
    O = torch.einsum("bhqn,bhnd->bhqd", attn, eV)
    return O


class KMeansGrouperWithPart(nn.Module):
    """
    KMeans on a sampled subset of keys (U_part), then assign full keys to groups,
    optional refinement.
    """
    def __init__(self, iters_on_part: int = 5, refine_iters: int = 1):
        super().__init__()
        self.iters_on_part = iters_on_part
        self.refine_iters = refine_iters

    @torch.no_grad()
    def forward(self, K: torch.Tensor, num_groups: int, U_part: int):
        """
        K: (B,H,LK,d)
        Returns: R (B,H,N,d), BLG (B,H,LK), CNT (B,H,N)
        """
        B, H, LK, d = K.shape
        N = int(num_groups)
        U_part = int(min(max(U_part, 1), LK))

        device = K.device
        dtype = K.dtype

        # sample subset of keys
        part_idx = torch.randint(0, LK, (B, H, U_part), device=device)
        K_part = K.gather(dim=2, index=part_idx.unsqueeze(-1).expand(B, H, U_part, d)).contiguous()

        # init centroids from subset
        init_idx = torch.randint(0, U_part, (B, H, N), device=device)
        R = K_part.gather(dim=2, index=init_idx.unsqueeze(-1).expand(B, H, N, d)).contiguous()

        # kmeans on subset
        for _ in range(self.iters_on_part):
            k2 = (K_part * K_part).sum(dim=-1, keepdim=True)          # (B,H,U,1)
            r2 = (R * R).sum(dim=-1).unsqueeze(2)                     # (B,H,1,N)
            kr = torch.einsum("bhud,bhnd->bhun", K_part, R)            # (B,H,U,N)
            dist = k2 + r2 - 2.0 * kr
            BLG_part = torch.argmin(dist, dim=-1)                     # (B,H,U)

            CNT = torch.zeros((B, H, N), device=device, dtype=torch.long)
            CNT.scatter_add_(2, BLG_part, torch.ones_like(BLG_part, dtype=torch.long))

            sumK = torch.zeros((B, H, N, d), device=device, dtype=dtype)
            sumK.scatter_add_(2, BLG_part.unsqueeze(-1).expand(B, H, U_part, d), K_part)

            cnt_f = CNT.clamp_min(1).to(dtype=dtype).unsqueeze(-1)
            newR = sumK / cnt_f
            empty = (CNT == 0).unsqueeze(-1)
            R = torch.where(empty, R, newR)

        # assign full keys to groups + optional refinement
        for _ in range(self.refine_iters):
            k2 = (K * K).sum(dim=-1, keepdim=True)                    # (B,H,LK,1)
            r2 = (R * R).sum(dim=-1).unsqueeze(2)                     # (B,H,1,N)
            kr = torch.einsum("bhkd,bhnd->bhkn", K, R)                 # (B,H,LK,N)
            dist = k2 + r2 - 2.0 * kr
            BLG = torch.argmin(dist, dim=-1)                          # (B,H,LK)

            CNT = torch.zeros((B, H, N), device=device, dtype=torch.long)
            CNT.scatter_add_(2, BLG, torch.ones_like(BLG, dtype=torch.long))

            sumK = torch.zeros((B, H, N, d), device=device, dtype=dtype)
            sumK.scatter_add_(2, BLG.unsqueeze(-1).expand(B, H, LK, d), K)

            cnt_f = CNT.clamp_min(1).to(dtype=dtype).unsqueeze(-1)
            R = sumK / cnt_f

        return R, BLG, CNT


class MultiheadGroupCrossAttention(nn.Module):
    """
    Cross-Attention: Q from x_q, K/V from x_kv.
    Grouping is performed on K (length LK), groups count u depends on LQ.
    """
    def __init__(self, embed_dim: int, num_heads: int, factor: int = 30,
                 dv: int | None = None, iters_on_part: int = 5, refine_iters: int = 1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dv = dv if dv is not None else self.head_dim
        self.factor = factor

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, num_heads * self.dv, bias=False)
        self.Wo = nn.Linear(num_heads * self.dv, embed_dim, bias=False)

        self.grouper = KMeansGrouperWithPart(iters_on_part=iters_on_part, refine_iters=refine_iters)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor):
        """
        x_q:  (B,LQ,D)
        x_kv: (B,LK,D)
        """
        B, LQ, D = x_q.shape
        _, LK, _ = x_kv.shape
        H = self.num_heads
        d = self.head_dim
        dv = self.dv

        Q = self.Wq(x_q).view(B, LQ, H, d).transpose(1, 2)      # (B,H,LQ,d)
        K = self.Wk(x_kv).view(B, LK, H, d).transpose(1, 2)     # (B,H,LK,d)
        V = self.Wv(x_kv).view(B, LK, H, dv).transpose(1, 2)    # (B,H,LK,dv)

        # u depends on LQ, U_part depends on LK
        u, U_part = calc_u_and_upart(L_Q=LQ, L_K=LK, factor=self.factor)

        # group K -> R, BLG, CNT
        R, BLG, CNT = self.grouper(K, num_groups=u, U_part=U_part)

        O = group_attention_op(Q=Q, V=V, R=R, BLG=BLG, CNT=CNT)  # (B,H,LQ,dv)

        y = O.transpose(1, 2).contiguous().view(B, LQ, H * dv)
        y = self.Wo(y)

        return y


class GroupCrossTransformerEncoderLayer(nn.Module):
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
        self.cross_attn = MultiheadGroupCrossAttention(embed_dim=d_model, num_heads=nhead, iters_on_part=10,
                                                       refine_iters=10)  #10,10

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

        out= self.cross_attn(tgt_norm, memory)
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


class GroupCrossTransformerEncoder(nn.Module):
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
    encoder_layer = GroupCrossTransformerEncoderLayer(
        d_model=config.hidden_size,
        nhead=config.num_attention_heads,
        dim_feedforward=config.intermediate_size,
        dropout=config.dropout_rate,
    )

    self.cross_vision = GroupCrossTransformerEncoder(
        encoder_layer,
        num_layers=config.num_vision_cross_layers,
        norm=nn.LayerNorm(config.hidden_size),
    )

    output_tokens = self.cross_vision(
        target_vision_tokens,
        patch_features,
    )


def demo_cross_attention():
    torch.manual_seed(0)

    B = 32
    LQ = 1024     # query length (e.g., text tokens)
    LK = 1024    # key/value length (e.g., image patches or long TS)
    D = 256
    H = 8
    factor = 20

    x_q = torch.randn(B, LQ, D)
    x_kv = torch.randn(B, LK, D)

    attn = MultiheadGroupCrossAttention(
        embed_dim=D,
        num_heads=H,
        factor=factor,
        iters_on_part=10,
        refine_iters=2,
    )

    torch.cuda.synchronize()
    start = time.time()
    y= attn(x_q, x_kv)
    end = time.time() - start
    print(end)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    demo_cross_attention()

