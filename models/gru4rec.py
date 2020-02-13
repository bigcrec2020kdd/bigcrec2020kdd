import torch
from torch import nn
from torch.nn import functional as F


class GRU4RecSimple(nn.Module):

    def __init__(self, n_items, emb_size, hidden_size, n_layers, emb_dropout, hidden_dropout, n_samples):
        super().__init__()
        self.n_items = n_items
        self.n_samples = n_samples

        self.item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=hidden_dropout, batch_first=True)

        self.target_item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout)

    def transform_input(self, h_iids):
        item_embs = self.item_emb_table(h_iids)  # [bs, len, dim]
        item_embs = self.emb_dropout(item_embs)
        output, state = self.gru(item_embs)

        last_idx = h_iids.ne(0).sum(1) - 1
        z = output[range(output.size(0)), last_idx]
        return z

    def nce_loss(self, h_iids, t_iids):
        bs = h_iids.size(0)
        h = self.transform_input(h_iids)  # [bs, dim]
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

    def forward(self, h_iids):
        h = self.transform_input(h_iids)  # [bs, dim]
        logits = h @ self.target_item_emb_table.weight.t()
        return logits


class GRU4RecSeq(nn.Module):

    def __init__(self, n_items, emb_size, hidden_size, n_layers, emb_dropout, hidden_dropout, n_samples):
        super().__init__()
        self.n_items = n_items
        self.n_samples = n_samples

        self.item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=hidden_dropout, batch_first=True)

        self.target_item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout)

    def transform_input(self, h_iids):
        item_embs = self.item_emb_table(h_iids)  # [bs, len, dim]
        item_embs = self.emb_dropout(item_embs)
        output, _ = self.gru(item_embs)
        return output

    def nce_loss(self, h_iids, t_iids):
        bs = h_iids.size(0)
        h = self.transform_input(h_iids)

        neg_iids = torch.randint(low=1, high=self.n_items,
                                 size=(1, h_iids.size(1), self.n_samples),
                                 device=h_iids.device)
        pos_y = self.target_item_emb_table(t_iids)  # [bs, len, dim]
        neg_y = self.target_item_emb_table(neg_iids)  # [1, len, N, dim]
        y = torch.cat([pos_y.unsqueeze(-2), neg_y.expand(bs, -1, -1, -1)], dim=2)  # [bs, len, N+1, dim]
        logits = (y @ h.unsqueeze(-1)).squeeze(-1)  # [bs, len, N+1]
        logits = F.log_softmax(logits, -1)
        xentropy = -logits[:, :, 0]
        xentropy = xentropy.masked_fill(h_iids.eq(0), 0).mean()
        return xentropy

    def forward(self, h_iids):
        output = self.transform_input(h_iids)
        last_idx = h_iids.ne(0).sum(1) - 1
        h = output[range(output.size(0)), last_idx]
        logits = h @ self.target_item_emb_table.weight.t()
        return logits
