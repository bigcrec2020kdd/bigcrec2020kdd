import torch
from torch import nn
from torch.nn import functional as F


class Caser(nn.Module):

    def __init__(self, n_items, emb_size, n_v_convs, n_h_convs, hist_max_len, dropout, n_samples):
        super().__init__()
        self.n_items = n_items
        self.n_samples = n_samples

        self.item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)

        self.conv_v = nn.Conv2d(1, n_v_convs, (hist_max_len, 1))
        lengths = [i + 1 for i in range(hist_max_len)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, n_h_convs, (i, emb_size)) for i in lengths])

        self.fc1_dim_v = n_v_convs * emb_size
        self.fc1_dim_h = n_h_convs * len(lengths)
        self.fc1 = nn.Linear(self.fc1_dim_v + self.fc1_dim_h, emb_size)
        self.target_item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.target_item_bias_table = nn.Embedding(n_items, 1, padding_idx=0)

        self.dropout = nn.Dropout(dropout)

        self.item_emb_table.weight.data.normal_(0, 1.0 / self.item_emb_table.embedding_dim)
        self.target_item_emb_table.weight.data.normal_(0, 1.0 / self.target_item_emb_table.embedding_dim)
        self.target_item_bias_table.weight.data.zero_()

    def transform_input(self, h_iids):
        item_embs = self.item_emb_table(h_iids).unsqueeze(1)  # [bs, 1, len, dim]

        out_v = self.conv_v(item_embs).view(-1, self.fc1_dim_v)  # [bs, nv x dim]
        out_hs = []
        for conv in self.conv_h:
            conv_out = F.relu(conv(item_embs).squeeze(3))  # [bs, nv, ?]
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [bs, nv]
            out_hs.append(pool_out)
        out_h = torch.cat(out_hs, 1)

        out = self.dropout(torch.cat([out_v, out_h], 1))
        z = F.relu(self.fc1(out))
        return z

    def nce_loss(self, h_iids, t_iids):
        bs = h_iids.size(0)
        h = self.transform_input(h_iids)
        neg_iids = torch.randint(low=1, high=self.n_items,
                                 size=(1, self.n_samples),
                                 device=h_iids.device)
        pos_y = self.target_item_emb_table(t_iids)  # [bs, dim]
        pos_b = self.target_item_bias_table(t_iids)  # [bs, 1]
        neg_y = self.target_item_emb_table(neg_iids)  # [N, dim]
        neg_b = self.target_item_bias_table(neg_iids)  # [N, 1]
        y = torch.cat([pos_y.unsqueeze(1), neg_y.expand(bs, -1, -1)], dim=1)  # [bs, N+1, dim]
        b = torch.cat([pos_b.unsqueeze(1), neg_b.expand(bs, -1, -1)], dim=1)  # [bs, N+1, 1]
        logits = (y @ h.unsqueeze(-1)).squeeze(-1)  # [bs, N+1]
        logits += b.squeeze(2)
        logits = F.log_softmax(logits, dim=1)
        xentropy = -logits[:, 0].mean()
        return xentropy

    def forward(self, h_iids):
        h = self.transform_input(h_iids)  # [bs, d]
        logits = h @ self.target_item_emb_table.weight.t()  # [bs, N]
        logits += self.target_item_bias_table.weight.squeeze(1)
        return logits
