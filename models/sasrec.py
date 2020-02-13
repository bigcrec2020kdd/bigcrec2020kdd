import math
import torch
from torch import nn
from torch.nn import functional as F


class SASRec(nn.Module):

    def __init__(self, n_items, hist_max_len, emb_size, n_heads, n_blocks,
                 n_samples, dropout, attn_dropout):
        super().__init__()
        self.n_items = n_items
        self.hist_max_len = hist_max_len
        self.n_samples = n_samples
        self.item_emb_table = nn.Embedding(n_items, emb_size)
        self.temporal_emb_table = nn.Embedding(hist_max_len, emb_size)
        self.target_item_emb_table = self.item_emb_table
        self.blocks = nn.ModuleList([Block(emb_size, n_heads, dropout, attn_dropout)
                                     for _ in range(n_blocks)])

    def nce_loss(self, h_iids, t_iids):
        bs = h_iids.size(0)
        h = self.transform_input(h_iids)  # [bs, len, dim]

        neg_iids = torch.randint(low=1, high=self.n_items,
                                 size=(1, self.hist_max_len, self.n_samples),
                                 device=h_iids.device)
        pos_y = self.target_item_emb_table(t_iids)  # [bs, len, dim]
        neg_y = self.target_item_emb_table(neg_iids)  # [1, len, N, dim]
        y = torch.cat([pos_y.unsqueeze(-2), neg_y.expand(bs, -1, -1, -1)], dim=2)  # [bs, len, N+1, dim]
        logits = (y @ h.unsqueeze(-1)).squeeze(-1)  # [bs, len, N+1]
        logits = F.log_softmax(logits, -1)
        xentropy = -logits[:, :, 0]
        xentropy = xentropy.masked_fill(h_iids.eq(0), 0).mean()
        return xentropy

    def get_emb(self, h_iids):
        item_emb = self.item_emb_table(h_iids)
        temporal_emb = self.temporal_emb_table.weight.unsqueeze(0)
        mask = h_iids.unsqueeze(-1).eq(0)
        return item_emb + temporal_emb.masked_fill(mask, 0)

    def transform_input(self, h_iids):
        x = self.get_emb(h_iids)
        padding_mask = h_iids.ne(0)
        for l in self.blocks:
            x = l(x, padding_mask)
        # [bs, length, dim]
        return x

    def forward(self, h_iids):
        output = self.get_emb(h_iids)  # [bs, len, dim]
        last_idx = h_iids.ne(0).sum(1) - 1
        h = output[range(output.size(0)), last_idx]  # [bs, dim]
        logits = h @ self.target_item_emb_table.weight.t()
        return logits


class Block(nn.Module):

    def __init__(self, input_size, n_head, dropout, attn_dropout):
        super().__init__()
        self.mh_attn_layer = MultiHeadAttention(input_size, n_head, dropout, attn_dropout)
        self.fc = nn.Sequential(nn.Linear(input_size, input_size),
                                nn.ReLU(),
                                nn.Linear(input_size, input_size))
        self.connector_attn = Connector(input_size, dropout)
        self.connector_fc = Connector(input_size, dropout)

    def forward(self, x, padding_mask):
        h = self.connector_attn(x, lambda v: self.mh_attn_layer(v, padding_mask))
        z = self.connector_fc(h, self.fc)
        return z


class MultiHeadAttention(nn.Module):

    def __init__(self, input_size, n_head, dropout, attn_dropout):
        super().__init__()
        self.attn_layers = nn.ModuleList([SelfAttentionLayer(input_size, input_size // n_head, dropout, attn_dropout)
                                          for _ in range(n_head)])
        # In the paper and the code on GitHub, there is no linear projection after MH attention.

    def forward(self, x, padding_mask):
        outs = [l(x, padding_mask) for l in self.attn_layers]  # [ [bs, length, d / nh] ]
        z = torch.cat(outs, -1)  # [bs, length, dim]
        return z


class SelfAttentionLayer(nn.Module):

    def __init__(self, input_size, hidden_size, dropout, attn_dropout):
        super().__init__()
        self.linear_mappings = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for _ in range(3)])
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, padding_mask):
        # x : [bs, length, dim]
        # padding_mask: [bs, length]
        length = x.size(1)
        subseq_mask = make_subsequent_mask(length, x.device)  # [1, length, length]
        mask = padding_mask.unsqueeze(1) & subseq_mask  # [bs, length, length]
        q, k, v = [l(x) for l in self.linear_mappings]
        z = scaled_dot_attention(q, k, v, mask=mask, dropout_fn=self.attn_dropout)
        return z


class Connector(nn.Module):

    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x, fn):
        return x + self.dropout(fn(self.layer_norm(x)))


def make_subsequent_mask(size, device):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    ones = torch.ones(attn_shape, device=device, dtype=torch.uint8)
    subsequent_mask = ~torch.triu(ones, 1)
    return subsequent_mask


def scaled_dot_attention(q, k, v, mask=None, dropout_fn=None):
    # q: [bs, m, dim]
    # k/v: [bs, n, dim]
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # [bs, m, n]
    if mask is not None:
        scores = scores.masked_fill(mask.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)  # [bs, *, n]
    if dropout_fn is not None:
        p_attn = dropout_fn(p_attn)
    z = p_attn @ v  # [bs, *, dim]
    return z
