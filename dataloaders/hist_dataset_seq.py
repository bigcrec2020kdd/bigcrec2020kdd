import numpy as np
import pandas as pd
import pathlib

from torch.utils.data import Dataset

from .prefetch_dataloader import DataLoaderX

from .hist_dataset_v2 import get_idx


class SeqHistDataset(Dataset):

    def __init__(self, df, idx_list, hist_max_len):
        self.data = df.values  # [user, item, timestamp]
        self.idx_list = idx_list
        self.hist_max_len = hist_max_len
        self.max_timestamp = self.data[:, 2].max()

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        l, r, t = self.idx_list[i]
        submat = self.data[l:r+1]  # include target
        h_uid, h_items, _ = submat.T
        hist_items = h_items[:-1]
        target_items = h_items[1:]
        t_id = h_items[-1]
        n = len(hist_items)
        uid = h_uid[0]
        assert np.all(h_uid == uid)
        if n < self.hist_max_len:
            hist_items, target_items = [np.pad(i, [0, self.hist_max_len - n], 'constant', constant_values=0)
                    for i in [hist_items, target_items]]
        return uid, hist_items, target_items, t_id


def load_seq_dataloader(args):
    root = pathlib.Path('data/')
    df = pd.read_csv(root / f'{args.dataset}.csv', header=None, names=['user', 'item', 'timestamp'])
    df = df.sort_values(['user', 'timestamp'], ascending=[True, True])

    n_users, n_items, _ = df.max(0) + 1
    black_list = df.groupby('user').apply(lambda subdf: subdf.item.values).to_dict()

    train_idx_list, valid_idx_list, test_idx_list = get_idx(df, args.hist_min_len, args.hist_max_len, 1)

    train_ds = SeqHistDataset(df, train_idx_list, args.hist_max_len)
    valid_ds = SeqHistDataset(df, valid_idx_list, args.hist_max_len)
    test_ds = SeqHistDataset(df, test_idx_list, args.hist_max_len)
    train_dl = DataLoaderX(train_ds, args.train_bs, pin_memory=True,
                           shuffle=True, drop_last=True, num_workers=args.n_workers)
    valid_dl = DataLoaderX(valid_ds, args.eval_bs, pin_memory=True, num_workers=args.n_workers)
    test_dl = DataLoaderX(test_ds, args.eval_bs, pin_memory=True, num_workers=args.n_workers)
    return train_dl, valid_dl, test_dl, n_items, black_list
