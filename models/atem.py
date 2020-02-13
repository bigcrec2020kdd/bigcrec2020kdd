import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import embedding


class ATEM(nn.Module):

    def __init__(self, n_items, n_buckets, emb_size, n_samples, tie_emb=True):
        super().__init__()
        self.n_items = n_items
        self.n_buckets = n_buckets
        self.n_samples = n_samples
        self.tie_emb = tie_emb

        # self.item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)
        # self.temporal_emb_table = nn.Embedding(n_buckets, emb_size, padding_idx=0)
        self.item_emb_table = embedding(n_items, emb_size, padding_idx=0)
        self.temporal_emb_table = embedding(n_buckets, emb_size, padding_idx=0)
        if self.tie_emb:
            self.target_item_emb_table = self.item_emb_table
        else:
            self.target_item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.attn_layer = nn.Sequential(nn.Linear(emb_size, 1, bias=False),
                                        nn.Softmax(dim=1))

    def get_emb(self, h_iids, bucket_ids):
        item_emb = self.item_emb_table(h_iids)
        temporal_emb = self.temporal_emb_table(bucket_ids)
        return item_emb + temporal_emb

    def nce_loss(self, h_iids, bucket_ids, t_iids):
        bs = h_iids.size(0)
        h = self.transform_input(h_iids, bucket_ids)  # [bs, dim]
        neg_iids = torch.randint(low=1, high=self.n_items,
                                 size=(1, self.n_samples),
                                 device=h_iids.device)
        pos_y = self.target_item_emb_table(t_iids)  # [bs, dim]
        neg_y = self.target_item_emb_table(neg_iids)  # [N, dim]
        y = torch.cat([pos_y.unsqueeze(1), neg_y.expand(bs, -1, -1)], dim=1)  # [bs, N+1, dim]
        logits = (y @ h.unsqueeze(-1)).squeeze(-1)  # [bs, N+1]
        logits = F.log_softmax(logits, dim=1)
        xentropy = -logits[:, 0].mean()
        return xentropy

    def forward(self, h_iids, bucket_ids):
        h = self.transform_input(h_iids, bucket_ids)
        logits = h @ self.target_item_emb_table.weight.t()
        return logits

    def transform_input(self, h_iids, bucket_ids):
        x = self.get_emb(h_iids, bucket_ids)  # [bs, k, dim]
        a = self.attn_layer(x)  # [bs, k, 1]
        h = (x * a).sum(dim=1)
        return h
