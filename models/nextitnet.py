import torch
from torch import nn
from torch.nn import functional as F


class NextItNet(nn.Module):

    def __init__(self, n_items, emb_size, dilations, n_samples):
        super().__init__()
        self.n_items = n_items
        self.n_samples = n_samples

        self.item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)
        self.conv_layers = nn.Sequential(*[ConvBlock(emb_size, 3, d) for d in dilations])
        self.target_item_emb_table = nn.Embedding(n_items, emb_size, padding_idx=0)

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
        output = self.transform_input(h_iids)  # [bs, length, dim]
        last_idx = h_iids.ne(0).sum(1) - 1
        h = output[range(output.size(0)), last_idx]
        logits = h @ self.target_item_emb_table.weight.t()
        return logits

    def transform_input(self, h_iids):
        item_embs = self.item_emb_table(h_iids).transpose(1, 2)  # [bs, dim, length]
        z = self.conv_layers(item_embs).transpose(1, 2)
        return z


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              dilation=dilation, padding=self.padding, **kwargs)

    def forward(self, x):
        z = self.conv(x)
        if self.padding > 0:
            z = z[:, :, :-self.padding]
        return z


class ChannelFirstLayerNorm(nn.Module):

    def __init__(self, normalized_shape, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, **kwargs)

    def forward(self, x):
        h = x.transpose(1, -1)
        h = self.layer_norm(h).transpose(1, -1)
        return h


class ConvBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, dilation):
        super().__init__()
        self.in_layer_norm = ChannelFirstLayerNorm(in_channels)
        self.in_conv1x1 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1)
        self.mid_layer_norm = ChannelFirstLayerNorm(in_channels // 2)
        self.causal_conv = CausalConv1d(in_channels // 2, in_channels // 2,
                                        kernel_size=kernel_size, dilation=dilation)
        self.out_layer_norm = ChannelFirstLayerNorm(in_channels // 2)
        self.out_conv1x1 = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        # x : [bs, in_channels, length]
        h = F.relu(self.in_layer_norm(x))
        h = F.relu(self.mid_layer_norm(self.in_conv1x1(h)))
        h = self.causal_conv(h)
        h = self.out_conv1x1(F.relu(self.out_layer_norm(h)))

        return x + h
