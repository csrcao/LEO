import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import faiss
import faiss.contrib.torch_utils
from .QueryBudgetController import AdaptiveBudgetController
import math


@torch.no_grad()
def build_faiss_index(col_feat, index='IVFFlat', step=6):
    """
    col_feat: [N, K]
    """
    col_feat, _ = torch.sort(col_feat, dim=1, descending=True)
    idx = np.linspace(0, col_feat.shape[1] - 1, num=step, dtype=int)
    col_np = col_feat.detach().cpu().numpy().astype(np.float16)[:, idx]
    col_np = np.ascontiguousarray(col_np)
    dim = col_np.shape[1]
    N = col_np.shape[0]

    if index == 'l2':
        res = faiss.StandardGpuResources()
        res.setTempMemory(1 << 30)
        opts = faiss.GpuClonerOptions()
        opts.useFloat16 = True
        cpu_index = faiss.IndexFlatL2(dim)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index, opts)
    elif index == 'IVFFlat':  # IVFFlat
        res = faiss.StandardGpuResources()
        res.setTempMemory(1 << 30)
        opts = faiss.GpuClonerOptions()
        opts.useFloat16 = True
        nlist = max(1, N // 40)
        quantizer = faiss.IndexFlatL2(dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        cpu_index.train(col_np)
        cpu_index.add(col_np)
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index, opts)
    else:
        M = 8
        index = faiss.IndexHNSWFlat(dim, M)  # L2
        index.hnsw.efSearch = 16
        index.hnsw.efConstruction = 40
        index.add(col_np)

    return index


@torch.no_grad()
def faiss_query(index, x, batch_size=2048, step=6):
    """
    x: [Q, K]
    return: [Q, 1]
    """
    x = x.reshape(-1, x.shape[-1])
    idx = np.linspace(0, x.shape[1] - 1, num=step, dtype=int)

    all_I = []
    for i in range(0, x.shape[0], batch_size):
        batch = x[i:i + batch_size]
        batch_np = batch.detach().cpu().numpy().astype(np.float16)[:, idx]
        batch_np = np.ascontiguousarray(batch_np)
        _, I = index.search(batch_np, 1)
        all_I.append(I)

    I = np.vstack(all_I)
    I = np.asarray(I, dtype=np.int64)

    return torch.tensor(I, dtype=torch.long, device=x.device)


def AV_matmul(att_pre_nor, att_1_nor, att_2_nor, K_indices, K2_indices, V):
    batch, head, M, D = V.shape
    batch_indices = torch.arange(batch, device=V.device)[:, None, None]
    head_indices = torch.arange(head, device=V.device)[None, :, None]

    major_v = V[batch_indices, head_indices, K_indices, :]
    major2_v = V[batch_indices, head_indices, K2_indices, :]

    major_context = torch.matmul(att_pre_nor, major_v)

    mask = torch.zeros(batch, head, M, dtype=torch.bool, device=V.device)
    indices = torch.cat([K_indices, K2_indices], dim=-1)
    mask.scatter_(dim=2, index=indices, value=True)

    V_in = V[~mask]
    V_remaining = V_in.reshape(batch, head, -1, D)

    V_sum = V_remaining.sum(dim=2)
    Vm2_sum = major2_v.sum(dim=2)

    context = (
        major_context
        + torch.matmul(att_1_nor.unsqueeze(-1), Vm2_sum.unsqueeze(2))
        + torch.matmul(att_2_nor.unsqueeze(-1), V_sum.unsqueeze(2))
    )

    return context



def css_probe(Q, K, s, budget_ctrl):
    """
    Q: [B, H, Lq, D]
    K: [B, H, Lk, D]
    """
    with torch.no_grad():
        B, H, Lq, D = Q.shape
        _, _, Lk, _ = K.shape
        T = B * H

        Qf = Q.reshape(T, Lq, D)
        Kf = K.reshape(T, Lk, D)

        sampled_idx = torch.stack(
            [torch.randperm(Lq, device=Q.device)[:s] for _ in range(T)]
        )

        Q_sub = Qf.gather(1, sampled_idx.unsqueeze(-1).expand(-1, -1, D))
        scores = torch.softmax(torch.bmm(Q_sub, Kf.transpose(1, 2)) / math.sqrt(D), dim=-1)
        votes = scores.sum(dim=1)

        K_sel = budget_ctrl.compute_K(int(s))
        K_sel = min(K_sel, Lk)
        top2k = min(2 * K_sel, Lk)
        top2k_idx = torch.topk(votes, k=top2k, dim=1, largest=True).indices
        K_indices = top2k_idx[:, :K_sel].reshape(B, H, -1)
        K2_indices = top2k_idx[:, K_sel:2 * K_sel].reshape(B, H, -1)

    K_top = Kf.gather(1, K_indices.reshape(T, K_sel).unsqueeze(-1).expand(-1, -1, D))
    att_pre = torch.softmax(torch.bmm(Qf, K_top.transpose(1, 2)) / math.sqrt(D), dim=-1)
    att_pre = att_pre.clone()

    return K_indices, K2_indices, K_sel, att_pre, {
        "scores": scores,
        "sampled_idx": sampled_idx,
        "top2k_idx": top2k_idx,
        "Lk": Lk,
    }

@torch.no_grad()
def restore_attention_order_perkey_mean(
    M: int,
    att_pre: torch.Tensor,       # (B,H,Lq,K_sel)
    K_indices: torch.Tensor,     # (B,H,Lq,K_sel) in [0,M)
    att_1_nor: torch.Tensor,     # (B,H,Lq)  per-key mean for K2 group
    K2_indices: torch.Tensor,    # (B,H,Lq,K2) in [0,M)
    att_2_nor: torch.Tensor,     # (B,H,Lq)  per-key mean for remaining keys
):

    B, H, Lq = att_1_nor.shape
    W = att_2_nor.unsqueeze(-1).expand(B, H, Lq, M).clone()

    K2_idx = K2_indices.unsqueeze(2).expand(B, H, Lq, -1)  # (B,H,Lq,K2)
    src2 = att_1_nor.unsqueeze(-1).expand(B, H, Lq, K2_idx.shape[-1])
    W.scatter_(dim=-1, index=K2_idx, src=src2)

    K1_idx = K_indices.unsqueeze(2).expand(B, H, Lq, -1)  # (B,H,Lq,u)
    W.scatter_(dim=-1, index=K1_idx, src=att_pre)
    Z = W.sum(-1, keepdim=True)
    W = W / Z

    return W



@torch.no_grad()
def build_index_table(att_pre, K_sel, stage1_out, index_type, step):
    scores = stage1_out["scores"]
    sampled_idx = stage1_out["sampled_idx"]
    top2k_idx = stage1_out["top2k_idx"]
    Lk = stage1_out["Lk"]
    s = sampled_idx.shape[1]
    top2k = min(2 * K_sel, Lk)

    topk_sum = scores.gather(2, top2k_idx[:, :K_sel].unsqueeze(1).expand(-1, s, -1)).sum(dim=2, keepdim=True)
    top2k_mean = scores.gather(2, top2k_idx[:, K_sel:].unsqueeze(1).expand(-1, s, -1)).mean(dim=2, keepdim=True)
    top2k_sum = scores.gather(2, top2k_idx.unsqueeze(1).expand(-1, s, -1)).sum(dim=2, keepdim=True)
    remaining_mean = (scores.sum(dim=2, keepdim=True) - top2k_sum) / (Lk - top2k)
    index_table = torch.cat([topk_sum, top2k_mean, remaining_mean], dim=2).reshape(-1, 3)
    global_mean = index_table.mean(dim=0, keepdim=True)
    safe_row = torch.zeros((1, 3), device=index_table.device, dtype=index_table.dtype)
    mu = global_mean[:, 0]  # μ
    safe_row[:, 0] = mu
    rem_mean = (1.0 - mu) / (Lk - K_sel)
    safe_row[:, 1] = rem_mean  # top2k_mean
    safe_row[:, 2] = rem_mean  # remaining_mean
    index_table = torch.cat([safe_row, index_table], dim=0)
    topk = att_pre.gather(1, sampled_idx.unsqueeze(-1).expand(-1, -1, K_sel))
    faiss_index = build_faiss_index(topk.reshape(-1, K_sel), index=index_type, step=step)

    return index_table, faiss_index


@torch.no_grad()
def query_energy_filter_and_faiss_route(
    att_pre,                # [T, Lq, K]
    faiss_index,
    final_K,
    tail_ratio,
    step,
    budget_ctrl,
    device,
):
    """
        nearest: [T, 1] LongTensor
    """
    att_pre_flat = att_pre.reshape(-1, final_K)  # [T*Lq, K]
    D = att_pre_flat.shape[1]
    tail = int(D * (1 - tail_ratio))

    att_pre_flat_sorted, _ = torch.sort(att_pre_flat, dim=1, descending=True)
    head_energy = (att_pre_flat_sorted[:, :tail] ** 2).sum(dim=1)
    tail_energy = (att_pre_flat_sorted[:, tail:] ** 2).sum(dim=1)
    score = tail_energy / (head_energy + 1e-9)

    energy_ratio = budget_ctrl.get_ratio()

    base_mask = score > energy_ratio
    base_ratio = base_mask.float().mean().item()

    max_ratio = 0.11
    N = score.shape[0]
    max_k = max(1, int(N * max_ratio))

    if base_ratio > max_ratio:
        topk_idx = torch.topk(score, max_k, largest=True).indices
        valid_mask = torch.zeros_like(score, dtype=torch.bool)
        valid_mask[topk_idx] = True
    else:
        valid_mask = base_mask

    valid_ratio = valid_mask.float().mean().item()

    if budget_ctrl is not None:
        budget_ctrl.update(valid_ratio)

    T = att_pre_flat.shape[0]

    if valid_ratio == 0:
        nearest = torch.zeros((T, 1), dtype=torch.long, device=device)
    else:
        att_pre_valid = att_pre_flat_sorted[valid_mask]
        nearest_valid = faiss_query(
            faiss_index,
            att_pre_valid,
            step=step
        )

        nearest = torch.zeros((T, 1), dtype=torch.long, device=device)
        nearest[valid_mask] = nearest_valid + 1

    return nearest


class IndexManager:
    def __init__(self, step, index_type, update_interval):
        self.step = step
        self.index_type = index_type
        self.update_interval = update_interval
        self.index_table = None
        self.faiss_index = None
        self.step_counter = 0

    def need_update(self):
        return self.faiss_index is None or self.step_counter % self.update_interval == 0

    @torch.no_grad()
    def update(self, att_pre, K_sel, stage1):
        self.index_table, self.faiss_index = build_index_table(att_pre, K_sel, stage1_out=stage1, index_type=self.index_type, step=self.step)
        self.step_counter += 1


class TopRowSparseAttention(nn.Module):
    def __init__(
        self,
        factor=10,
        step=8,
        index_type='IVFFlat',
        update_interval=30,
        budget_ctrl=None,
        tail_ratio=0.1,
        query_filter=True,
        dropout=0.05,
        output_attention=False
    ):
        super().__init__()

        self.index_mgr = IndexManager(
            step=step,
            index_type=index_type,
            update_interval=update_interval,
        )
        self.budget_ctrl = budget_ctrl
        self.query_filter = query_filter
        self.tail_ratio = tail_ratio
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

    def forward(self, Q, K, V, index_use=True):
        """
        Q, K, V: [B, H, L, D]
        """
        _, _, M, _ = K.shape
        u = self.factor * np.ceil(np.log(M)).astype('int').item()
        u = u if u < M else M
        K_indices, K2_indices, K_sel, att_pre, stage1 = css_probe(Q, K, u, self.budget_ctrl)  # kproject
        B, H, Lq, D = Q.shape

        if self.index_mgr.need_update():
            self.index_mgr.update(att_pre, K_sel, stage1)

        # ---- attnapprox ----
        if index_use:
            if self.query_filter:
                nearest = query_energy_filter_and_faiss_route(
                    att_pre=att_pre,
                    faiss_index=self.index_mgr.faiss_index,
                    final_K=K_sel,
                    tail_ratio=self.tail_ratio,
                    step=self.index_mgr.step,
                    budget_ctrl=self.budget_ctrl,
                    device=att_pre.device,
                )
            else:
                nearest = faiss_query(
                    self.index_mgr.faiss_index,
                    att_pre,
                    step=self.index_mgr.step
                )

            result = self.index_mgr.index_table[nearest.squeeze(1)]
            topk_sum_part = result[..., 0].view(B, H, -1, 1)
            att_pre = att_pre.view(B, H, -1, K_sel)
            att_pre.mul_(topk_sum_part)
            att_pre = self.dropout(att_pre)
            att_1_nor = result[..., 1].view(B, H, Lq)
            att_2_nor = result[..., 2].view(B, H, Lq)
            context = AV_matmul(att_pre, att_1_nor, att_2_nor, K_indices, K2_indices, V)  # attention reconstruction
        else:
            batch, head, M, D = V.shape
            batch_indices = torch.arange(batch, device=V.device)[:, None, None]
            head_indices = torch.arange(head, device=V.device)[None, :, None]
            major_v = V[batch_indices, head_indices, K_indices, :]
            context = torch.matmul(att_pre.view(B, H, -1, K_sel), major_v)

        if self.output_attention:
            att = restore_attention_order_perkey_mean(M, att_pre, K_indices, att_1_nor, K2_indices, att_2_nor)
            return context, att
        else:
            return context, K_sel


