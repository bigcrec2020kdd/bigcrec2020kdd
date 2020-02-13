import torch
from torch import nn, optim
from torch.nn import functional as F

from .model_utils import embedding, trunc_normal_


class BiGCNLayer(nn.Module):
    """Bipartite Graph Convolutional Layer."""

    def __init__(self, in_size, *, dropout=0.5, include_y=True, bias=False, slope=0.2, dot_attn=True):
        super().__init__()
        self.p_attn = None  # log attention score
        self.include_y = include_y
        self.out_size = in_size
        self.FILL_VALUE = -9e15
        self.in_proj = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(in_size, in_size, bias=bias), )
        #                                      nn.LeakyReLU(slope))
        if dot_attn:
            self.attn_score = None
        else:
            self.attn_score = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(2 * in_size, 1, bias=bias),
                                            nn.LeakyReLU(slope))

        self.calc_attention = self.scaled_dot_attention if dot_attn else self.cat_affine_attention

    def scaled_dot_attention(self, hx, hy, mask):
        # hx : [bs, nx, dim]
        # hy : [bs, ny, dim]
        # mask : [bs, nx]
        bs = hx.size(0)
        n_x = hx.size(1)
        temperature = self.out_size ** .5
        # x -> y
        a_xy = hx @ hy.transpose(1, 2) / temperature  # [bs, nx, ny]
        if mask is not None:
            a_xy.masked_fill_(mask.unsqueeze(-1), self.FILL_VALUE)
        if self.include_y:
            # y self attention
            a_yy = (hy ** 2).sum(dim=2) / temperature  # [bs, ny]
            # stack
            a = torch.cat([a_xy, a_yy.unsqueeze(1)], dim=1)  # [bs, nx+1, ny]
            a = torch.softmax(a, dim=1)
            a_xy = a[:, :n_x, :]  # [bs, nx, ny]
            a_yy = a[:, n_x:, :]  # [bs, 1, ny]
            z = a_xy.transpose(1, 2) @ hx + a_yy @ hy
        else:
            a = torch.softmax(a_xy, dim=1)  # [bs, nx, ny]
            z = a.transpose(1, 2) @ hx  # [bs, ny, dim]
        return z

    def cat_affine_attention(self, hx, hy, mask):
        n_x, n_y = hx.size(1), hy.size(1)

        h = torch.cat([hx.unsqueeze(2).expand(-1, -1, n_y, -1),
                       hy.unsqueeze(1)], dim=1)  # [bs, n_x+1, n_y, dim]
        cat_vec = torch.cat([h, hy.unsqueeze(1).expand(-1, n_x + 1, -1, -1)], dim=-1)  # [bs, n_x+1, n_y, 2*dim]
        # x -> y attention
        a = self.attn_score(cat_vec).squeeze(-1)  # [bs, n_x+1, n_y]
        if mask is not None:
            a[:, :-1].masked_fill_(mask.unsqueeze(-1), self.FILL_VALUE)
        # y self attention
        if not self.include_y:
            a[:, -1].fill_(self.FILL_VALUE)
        # sum
        a = torch.softmax(a, dim=1)
        self.p_attn = a
        z = (h * a.unsqueeze(-1)).sum(dim=1)
        return z

    def forward(self, x, y, mask=None):
        # x -> y
        # x : [bs, n_x, in_size]
        # y : [bs, n_y, in_size] or [n_y, in_size]
        # mask : [bs, n_x]
        bs = x.size(0)
        n_x, n_y = x.size(1), y.size(-2)  # to support rank-2 and 3 tensor
        hx = self.in_proj(x)
        if x is y:
            hy = hx
        elif y.dim() == 2:
            hy = self.in_proj(y).expand(bs, -1, -1)  # for computation efficiency
        else:
            hy = self.in_proj(y)
        z = self.calc_attention(hx, hy, mask)
        return z


class SymBiGCNLayer(nn.Module):
    """Symmetric Bipartite Graph Convolutional Layer."""

    def __init__(self, in_size, *, dot_attn=True,
                 use_mid_layer=True, dropout=0.5, bias=True, slope=0.2):
        super().__init__()
        self.use_mid_layer = use_mid_layer
        kwargs = dict(dropout=dropout, bias=bias, slope=slope, dot_attn=dot_attn)
        #         self.act = nn.LeakyReLU(slope)
        self.act = lambda x: x
        self.x2y_layer = BiGCNLayer(in_size, include_y=True, **kwargs)
        if use_mid_layer:
            self.y2y_layer = BiGCNLayer(in_size, include_y=False, **kwargs)
        self.y2z_layer = BiGCNLayer(in_size, include_y=True, **kwargs)

    def forward(self, x, y, z, mask=None):
        # x -> y
        y_hat = self.act(self.x2y_layer(x, y, mask))
        # y -> y
        if self.use_mid_layer:
            y_hat = self.act(self.y2y_layer(y_hat, y_hat))
        # y -> z
        z_hat = self.act(self.y2z_layer(y_hat, z))
        return z_hat


class MultiHeadSymBiGCNLayer(nn.Module):
    """Multi-Head Symmetric Bipartite Graph Convolutional Layer"""

    def __init__(self, in_size, n_topics, n_heads, *,
                 use_mid_layer=True, dropout=0.5, bias=True, slope=0.2, dot_attn=True):
        super().__init__()

        self.topic_nodes = nn.Parameter(torch.Tensor(n_topics, in_size))
        nn.init.xavier_uniform_(self.topic_nodes.data)
        # trunc_normal_(self.topic_nodes.data, 0, 0.01)

        self.heads = [SymBiGCNLayer(in_size,
                                    dot_attn=dot_attn,
                                    use_mid_layer=use_mid_layer,
                                    dropout=dropout,
                                    bias=bias,
                                    slope=slope)
                      for _ in range(n_heads)]
        self.heads = nn.ModuleList(self.heads)

    def forward(self, x, z, mask=None):
        rets = [f(x, self.topic_nodes, z, mask) for f in self.heads]
        rets = torch.cat(rets, dim=-1)
        return rets


class MultiBranchSymBiGCNLayer(nn.Module):
    "Multi-Branch Multi-Head Symmetric Bipartite Graph Convolutional Layer."

    def __init__(self, n_branches, in_size, n_topics, n_heads, **kwargs):
        super().__init__()

        self.branches = nn.ModuleList(
            [MultiHeadSymBiGCNLayer(in_size, n_topics, n_heads, **kwargs) for _ in range(n_branches)])
        self.out_proj = nn.Sequential(nn.Dropout(.1),
                                      nn.Linear(n_heads * n_branches * in_size, in_size),
                                      nn.LeakyReLU(.2))

    def forward(self, x, z, mask=None):
        rets = [f(x, z, mask) for f in self.branches]
        rets = torch.cat(rets, dim=-1)
        rets = self.out_proj(rets)
        return rets


class BiGCrec_v3(nn.Module):

    def __init__(self, n_items, n_buckets, emb_size, n_topics, n_heads, n_branches, n_layers,
                 n_samples_per_layer=16, *, dropout=0.5, slope=0.2, use_mid_layer=True, bias=True):
        super().__init__()
        self.n_items = n_items
        self.n_buckets = n_buckets
        self.n_samples_per_layer = n_samples_per_layer
        self.dim = emb_size
        self.n_layers = n_layers
        self.item_emb_table = embedding(n_items, emb_size, padding_idx=0)
        self.temporal_emb_table = embedding(n_buckets, emb_size, padding_idx=0)
        self.bigc_layers = [MultiBranchSymBiGCNLayer(n_branches, emb_size, n_topics, n_heads,
                                                     dot_attn=False,
                                                     use_mid_layer=use_mid_layer,
                                                     dropout=dropout,
                                                     bias=bias,
                                                     slope=slope)
                            for _ in range(n_layers)]
        self.bigc_layers = nn.ModuleList(self.bigc_layers)
        self.score_layer = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(emb_size * (1 + n_layers), emb_size),
                                         # nn.LeakyReLU(slope)
                                         )
        self.attn_layer = nn.Sequential(nn.Linear(emb_size, 1, bias=False),
                                        nn.Softmax(dim=1))

    def in_proj(self, item_emb, temp_emb):
        return item_emb + temp_emb

    def get_score(self, outputs, seq_emb):
        # seq_emb : [bs, len, dim]
        a = self.attn_layer(seq_emb)  # [bs, len, 1]
        context = (seq_emb * a).sum(dim=1)  # [bs, dim]
        item_vecs = self.score_layer(outputs)  # [bs, 1+n_samples, dim]
        scores = item_vecs @ context.unsqueeze(-1)
        return scores.squeeze(-1)

    def forward(self, h_iids, buckets_ids):
        bs = h_iids.size(0)
        item_emb = self.item_emb_table(h_iids)
        temp_emb = self.temporal_emb_table(buckets_ids)
        seq_emb = self.in_proj(item_emb, temp_emb)

        x = seq_emb
        z = self.item_emb_table.weight.expand(bs, -1, -1)
        outputs = [z]
        for i, bigc_f in enumerate(self.bigc_layers):
            mask = None if i else h_iids.eq(0)
            z1 = bigc_f(x, z, mask)
            outputs.append(z1)
            x = z1
            z = z1

        outputs = torch.cat(outputs, dim=-1)
        logits = self.get_score(outputs, seq_emb)
        return logits

    def nce_loss(self, logits):
        # NCE loss
        logits = torch.log_softmax(logits, dim=1)
        xentropy = -logits[:, 0].mean()
        return xentropy

    def sample_forward_full_return(self, h_iids, buckets_ids):
        # h_iids [bs, len]
        # bucket_ids [bs, len]
        bs = h_iids.size(0)
        item_emb = self.item_emb_table(h_iids)
        temp_emb = self.temporal_emb_table(buckets_ids)
        seq_emb = self.in_proj(item_emb, temp_emb)  # [bs, len, dim]

        x = seq_emb
        z = self.item_emb_table.weight.expand(bs, -1, -1)
        outputs = [z]
        for i, bigc_f in enumerate(self.bigc_layers):
            mask = None if i else h_iids.eq(0)
            z1 = bigc_f(x, z, mask)
            outputs.append(z1)
            # split
            idx = torch.randint(low=1, high=self.n_items, size=(self.n_samples_per_layer,), device=h_iids.device)
            x = z1[:, idx]
            z = z1

        outputs = torch.cat(outputs, dim=-1)  # [bs, n_samples+1, (1+n_layers) * dim]
        logits = self.get_score(outputs, seq_emb)
        return logits

    def mini_batch_forward(self, h_iids, buckets_ids, t_iids):
        # h_iids [bs, len]
        # bucket_ids [bs, len]
        # t_iids [bs]
        # samples_iids [n_layers * n_samples]
        bs = h_iids.size(0)
        n_samples = self.n_samples_per_layer * self.n_layers
        n_samples_per_layer = self.n_samples_per_layer
        samples_iids = torch.randint(low=1, high=self.n_items, size=(n_samples,), device=h_iids.device)

        item_emb = self.item_emb_table(h_iids)
        temp_emb = self.temporal_emb_table(buckets_ids)
        seq_emb = self.in_proj(item_emb, temp_emb)  # [bs, len, dim]

        x = seq_emb
        target_emb = self.item_emb_table(t_iids)  # [bs, dim]
        samples_emb = self.item_emb_table(samples_iids)  # [n_samples, dim]
        z = torch.cat([target_emb.unsqueeze(1), samples_emb.expand(bs, -1, -1)], dim=1)  # [bs, n_samples+1, dim]

        outputs = [z[:, :1 + n_samples_per_layer]]  # true target + negative samples
        for i, bigc_f in enumerate(self.bigc_layers):
            mask = None if i else h_iids.eq(0)
            z1 = bigc_f(x, z, mask)
            outputs.append(z1[:, :1 + n_samples_per_layer])
            # split
            x = z1[:, -n_samples_per_layer:]  # [bs, n_samples_per_layer, dim]
            z = z1[:, :-n_samples_per_layer]  # [bs, n_samples - n_samples_per_layer * layer_id, dim]

        outputs = torch.cat(outputs, dim=-1)  # [bs, n_samples+1, (1+n_layers) * dim]
        logits = self.get_score(outputs, seq_emb)
        loss = self.nce_loss(logits)
        return loss
