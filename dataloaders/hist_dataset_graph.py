import numpy as np
import pandas as pd
import pathlib

import torch
from torch.utils.data import Dataset

from .prefetch_dataloader import DataLoaderX
from .hist_dataset_v2 import get_idx


class GraphHistDataset(Dataset):

    def __init__(self, df, idx_list):
        self.data = df.values
        self.idx_list = idx_list

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        l, r, t = self.idx_list[i]
        submat = self.data[l:r]
        h_uid, h_items, _ = submat.T
        uid, t_item, _ = self.data[t]
        assert np.all(h_uid == uid)
        return uid, h_items, t_item


def graph_collate_fn(batch):
    uids, h_iids, t_iids = zip(*batch)
    uids = torch.LongTensor(uids)
    t_iids = torch.LongTensor(t_iids)
    lengths = [len(x) for x in h_iids]
    max_length = max(lengths)
    batch_adj_mats = []
    batch_encoded_items = []
    batch_unique_items = []
    for i, line in enumerate(h_iids):
        padded_line = np.pad(line, [0, max_length - len(line)], 'constant', constant_values=0)
        unique_items = np.unique(padded_line)
        batch_unique_items.append(np.pad(unique_items, [0, max_length - len(unique_items)],
                                         'constant', constant_values=0))
        item2idx = {v: i for i, v in enumerate(unique_items)}
        indices = [item2idx[v] for v in line]
        # encoded_items = [item2idx[v] for v in line]  # for recovery
        encoded_item_set = np.pad(indices, [0, max_length - len(indices)],
                                  'constant', constant_values=0)  # for recovery
        batch_encoded_items.append(encoded_item_set)
        adj_mat = np.zeros([max_length, max_length])
        adj_mat[indices[:-1], indices[1:]] = 1
        adj_in = adj_mat / (np.clip(adj_mat.sum(0, keepdims=True), a_min=1., a_max=None))
        adj_out = adj_mat.T / (np.clip(adj_mat.sum(1, keepdims=True), a_min=1., a_max=None))
        adj_merged = np.concatenate([adj_in, adj_out]).T
        batch_adj_mats.append(adj_merged)

    b_encoded_items = torch.from_numpy(np.asarray(batch_encoded_items)).long()
    b_unique_items = torch.from_numpy(np.asarray(batch_unique_items)).long()
    adj_merged = torch.from_numpy(np.asarray(batch_adj_mats)).float()

    return uids, b_encoded_items, b_unique_items, adj_merged, t_iids


def load_graph_dataloader(args):
    root = pathlib.Path('data/')
    df = pd.read_csv(root / f'{args.dataset}.csv', header=None, names=['user', 'item', 'timestamp'])
    df = df.sort_values(['user', 'timestamp'], ascending=[True, True])

    n_users, n_items, _ = df.max(0) + 1
    black_list = df.groupby('user').apply(lambda subdf: subdf.item.values).to_dict()

    train_idx_list, valid_idx_list, test_idx_list = get_idx(df, args.hist_min_len, args.hist_max_len, 1)

    train_ds = GraphHistDataset(df, train_idx_list)
    valid_ds = GraphHistDataset(df, valid_idx_list)
    test_ds = GraphHistDataset(df, test_idx_list)
    train_dl = DataLoaderX(train_ds, args.train_bs, pin_memory=True, collate_fn=graph_collate_fn,
                           shuffle=True, drop_last=True, num_workers=args.n_workers)
    valid_dl = DataLoaderX(valid_ds, args.eval_bs, pin_memory=True, collate_fn=graph_collate_fn,
                           num_workers=args.n_workers)
    test_dl = DataLoaderX(test_ds, args.eval_bs, pin_memory=True, collate_fn=graph_collate_fn,
                          num_workers=args.n_workers)
    return train_dl, valid_dl, test_dl, n_items, black_list
