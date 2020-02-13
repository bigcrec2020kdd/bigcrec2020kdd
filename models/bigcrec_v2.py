import math

import torch
from torch import nn, optim
from torch.nn import functional as F

from .model_utils import embedding, trunc_normal_


class MultiHeadAttention(nn.Module):

    def __init__(self, input_size, output_size, n_heads, attn_dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = output_size // n_heads
        self.output_size = output_size
        self.linear_mappings = nn.ModuleList([nn.Linear(input_size, output_size, bias=False) for _ in range(3)])
        self.output_proj = nn.Linear(output_size, output_size)
        self.attn_dropout_fn = nn.Dropout(attn_dropout)

        self.p_attn = None  # for visualization

    def forward(self, q, k, v, mask=None):
        """
        Parameters
        ----------
        q : [bs, m, in_size], Source items of information
        k : [bs, n, in_size], target items
        v : [bs, n, in_size], target items
        mask : [bs, 1, n] or [bs, m, n]

        Returns
        -------
        output: [bs, m, out_size]
        """
        bs = q.size(0)
        q, k, v = [l(t).view(bs, -1, self.n_heads, self.head_size).transpose(1, 2)  # !!!IMPORTANT transpose(1, 2)
                   for l, t in zip(self.linear_mappings, [q, k, v])]
        # [bs, n_heads, m/n, hidden_size // n_heads]
        if mask is not None:
            mask = mask.unsqueeze(1)
        output, p_attn = scaled_dot_attention(q, k, v, mask)  # p_attn: [bs, h, m, n]
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.output_size)  # [bs, h, m, d//h] -> [bs, m, d]
        output = self.output_proj(output)
        self.p_attn = p_attn
        return output


class Connector(nn.Module):

    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x, fn):
        return x + self.dropout(fn(self.layer_norm(x)))


class PointWiseFC(nn.Module):

    def __init__(self, input_size, dropout):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, input_size),
                                nn.ReLU(),
                                nn.Linear(input_size, input_size))
        self.connector = Connector(input_size, dropout)

    def forward(self, x):
        output = self.connector(x, self.fc)
        return output


class ConvLayer(nn.Module):

    def __init__(self, input_size, n_topics, n_heads, dropout, attn_dropout, use_mid_layer):
        super().__init__()
        self.use_mid_layer = use_mid_layer
        self.topic_nodes = nn.Parameter(torch.Tensor(n_topics, input_size))
        trunc_normal_(self.topic_nodes.data, 0, 0.01)
        # x2y
        self.x2y_layer = MultiHeadAttention(input_size, input_size, n_heads=n_heads, attn_dropout=attn_dropout)
        self.y_connector = Connector(input_size, dropout)
        self.y_fc = PointWiseFC(input_size, dropout)
        # y2y
        if use_mid_layer:
            self.y2y_layer = MultiHeadAttention(input_size, input_size, n_heads=n_heads, attn_dropout=attn_dropout)
            self.mid_connector = Connector(input_size, dropout)
            self.mid_fc = PointWiseFC(input_size, dropout)
        # y2z
        self.y2z_layer = MultiHeadAttention(input_size, input_size, n_heads=n_heads, attn_dropout=attn_dropout)
        self.z_connector = Connector(input_size, dropout)
        self.z_fc = PointWiseFC(input_size, dropout)

    def forward(self, x, z, mask=None):
        bs = x.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        y = self.topic_nodes.expand(bs, -1, -1)
        # x -> y
        y_hat = self.y_connector(y, lambda t: self.x2y_layer(t, x, x, mask))
        y_hat = self.y_fc(y_hat)
        # y -> y
        if self.use_mid_layer:
            y_hat = self.mid_connector(y_hat, lambda t: self.y2y_layer(t, t, t))
            y_hat = self.mid_fc(y_hat)
        # y -> z
        z_hat = self.z_connector(z, lambda t: self.y2z_layer(t, y_hat, y_hat))
        z_hat = self.z_fc(z_hat)
        return z_hat


class BiGCrec_v2(nn.Module):

    def __init__(self, n_items, n_buckets, emb_size, n_topics, n_layers, n_heads,
                 n_samples_train, n_samples_eval, dropout, attn_dropout,
                 use_mid_layer=True, use_full=False):
        super().__init__()
        self.n_items = n_items
        self.n_buckets = n_buckets
        self.emb_size = emb_size
        self.n_topics = n_topics
        self.n_layers = n_layers
        self.n_samples_train = n_samples_train
        self.n_samples_eval = n_samples_eval
        self.use_full = use_full
        # Embeddings
        self.item_emb_table = embedding(n_items, emb_size, padding_idx=0)
        self.temp_emb_table = embedding(n_buckets, emb_size, padding_idx=0)

        # Convolutional Layers
        self.conv_layers = nn.ModuleList([ConvLayer(input_size=emb_size, n_topics=n_topics, n_heads=n_heads,
                                                    dropout=dropout, attn_dropout=attn_dropout,
                                                    use_mid_layer=use_mid_layer)
                                          for _ in range(n_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        # Output Layer
        # self.output_transform = nn.Linear(emb_size * (1 + n_layers), emb_size)
        self.score_layer = nn.Linear(emb_size * (1 + n_layers), 1, bias=False)

    def get_embedding(self, seq_item_ids, seq_temp_ids):
        item_embs = self.item_emb_table(seq_item_ids)
        temp_embs = self.item_emb_table(seq_temp_ids)
        return item_embs + temp_embs

    def get_score(self, outputs):
        outputs = torch.cat(outputs, dim=-1)  # [bs, 1+n_samples, dim * (1+n_layers)]
        scores = self.score_layer(outputs).squeeze(-1)
        return scores

    def forward(self, seq_item_ids, seq_temp_ids):
        if self.use_full:
            return self.full_forward(seq_item_ids, seq_temp_ids)
        else:
            return self.sample_forward(seq_item_ids, seq_temp_ids)

    def nce_loss(self, seq_item_ids, seq_temp_ids, target_item_ids):
        scores = self.train_sample_forward(seq_item_ids, seq_temp_ids, target_item_ids)
        logits = torch.log_softmax(scores, dim=1)
        xentropy = -logits[:, 0].mean()
        return xentropy

    def train_sample_forward(self, seq_item_ids, seq_temp_ids, target_item_ids):
        bs = seq_temp_ids.size(0)
        seq_embs = self.get_embedding(seq_item_ids, seq_temp_ids)  # [bs, len, dim]
        seq_mask = seq_item_ids.ne(0)  # 1 if padded, 0 otherwise.

        total_n_samples = self.n_layers * self.n_samples_train
        sample_item_ids = torch.randint(low=1, high=self.n_items, size=(total_n_samples,), device=seq_embs.device)
        sample_embs = self.item_emb_table(sample_item_ids)  # [total_n_samples, dim]
        target_embs = self.item_emb_table(target_item_ids)  # [bs, dim]

        x = seq_embs
        z = torch.cat([target_embs.unsqueeze(1), sample_embs.expand(bs, -1, -1)], dim=1)  # [bs, 1+samples, dim]
        outputs = [z[:, :1 + self.n_samples_train]]  # target + negative samples
        for i, (conv_f, norm_f) in enumerate(zip(self.conv_layers, self.norm_layers)):
            mask = None if i else seq_mask
            z_new = conv_f(norm_f(x), z, mask)
            # Log Layer outputs
            outputs.append(z_new[:, :1 + self.n_samples_train])
            # Partition Z
            x = z_new[:, -self.n_samples_train:]  # only use last n_samples items to propagate information
            z = z_new[:, :-self.n_samples_train]  # drop last n_samples items
        # [[bs, 1+n_samples, dim]]
        scores = self.get_score(outputs)  # [bs, 1 + n_samples]
        return scores

    def sample_forward(self, seq_item_ids, seq_temp_ids):
        bs = seq_temp_ids.size(0)
        seq_embs = self.get_embedding(seq_item_ids, seq_temp_ids)  # [bs, len, dim]
        seq_mask = seq_item_ids.ne(0)  # 1 if padded, 0 otherwise.

        x = seq_embs  # initial source of information
        z = self.item_emb_table.weight.expand(bs, -1, -1)  # [bs, n_items, dim]
        outputs = [z]
        for i, (conv_f, norm_f) in enumerate(zip(self.conv_layers, self.norm_layers)):
            mask = None if i else seq_mask
            z_new = conv_f(norm_f(x), z, mask)
            outputs.append(z_new)
            # Partition Z
            seed_idx = torch.randint(low=1, high=self.n_items, size=(self.n_samples_eval,), device=seq_embs.device)
            x = z_new[:, seed_idx]  # new source of information
            z = z_new
        scores = self.get_score(outputs)
        return scores

    def full_forward(self, seq_item_ids, seq_temp_ids):
        bs = seq_temp_ids.size(0)
        seq_embs = self.get_embedding(seq_item_ids, seq_temp_ids)  # [bs, len, dim]
        seq_mask = seq_item_ids.ne(0)  # 1 if padded, 0 otherwise.

        x = seq_embs  # initial source of information
        z = self.item_emb_table.weight.expand(bs, -1, -1)  # [bs, n_items, dim]
        outputs = [z]
        for i, (conv_f, norm_f) in enumerate(zip(self.conv_layers, self.norm_layers)):
            mask = None if i else seq_mask
            z_new = conv_f(norm_f(x), z, mask)
            outputs.append(z_new)
            # Partition Z
            x = z_new[:, 1:]  # yeah, all items are used as source
            z = z_new
        scores = self.get_score(outputs)
        return scores


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
    return z, p_attn
