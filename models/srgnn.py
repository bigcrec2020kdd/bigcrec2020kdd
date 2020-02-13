import math

import torch
from torch import nn
from torch.nn import functional as F


class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super().__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNN(nn.Module):
    def __init__(self, n_items, emb_size, n_samples, nonhybrid=False, step=1):
        super().__init__()
        self.hidden_size = emb_size
        self.n_items = n_items
        self.n_samples = n_samples
        # self.batch_size = opt.batchSize
        self.nonhybrid = nonhybrid
        self.item_emb_table = nn.Embedding(n_items, self.hidden_size, padding_idx=0)
        self.target_item_emb_table = self.item_emb_table
        self.gnn = GNN(self.hidden_size, step=step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, encoded_idx, unique_items, A):
        h = self.transform_input(encoded_idx, unique_items, A)
        logits = h @ self.target_item_emb_table.weight.t()
        return logits

    def nce_loss(self, encoded_idx, unique_items, A, t_iids):
        bs = encoded_idx.size(0)
        h = self.transform_input(encoded_idx, unique_items, A)  # [bs, dim]
        neg_iids = torch.randint(low=1, high=self.n_items,
                                 size=(1, self.n_samples),
                                 device=t_iids.device)
        pos_y = self.target_item_emb_table(t_iids)  # [bs, dim]
        neg_y = self.target_item_emb_table(neg_iids)  # [N, dim]
        y = torch.cat([pos_y.unsqueeze(1), neg_y.expand(bs, -1, -1)], dim=1)  # [bs, N+1, dim]
        logits = (y @ h.unsqueeze(-1)).squeeze(-1)  # [bs, N+1]
        logits = F.log_softmax(logits, dim=1)
        xentropy = -logits[:, 0].mean()
        return xentropy

    def transform_input(self, encoded_idx, unique_items, A):
        bs = encoded_idx.size(0)
        unique_hidden_states = self.item_emb_table(unique_items)
        unique_hidden_states = self.gnn(A, unique_hidden_states)
        seq_hidden_states = torch.stack([unique_hidden_states[i][encoded_idx[i]] for i in range(bs)])
        raw_iids = torch.stack([unique_items[i][encoded_idx[i]] for i in range(bs)])
        mask = raw_iids.ne(0)
        last_idx = mask.sum(1) - 1

        ht = seq_hidden_states[range(bs), last_idx]  # states of the last items

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(seq_hidden_states)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        z = torch.sum(alpha * seq_hidden_states * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            z = self.linear_transform(torch.cat([z, ht], 1))
        return z

