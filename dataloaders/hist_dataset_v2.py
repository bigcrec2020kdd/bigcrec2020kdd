import numpy as np
import pandas as pd
import pathlib

from torch.utils.data import Dataset

from .prefetch_dataloader import DataLoaderX


def extract_subseq(n, hist_min_len, hist_max_len, n_step):
    idx_list = []
    for step in range(1, n_step + 1):
        for right in range(hist_min_len, n + 1 - step):
            left = max(0, right - hist_max_len)
            target = right + step - 1
            idx_list.append([left, right, target])
    return np.asarray(idx_list)


def get_idx(df, hist_min_len, hist_max_len, n_step):
    offset = 0
    train_idx_list = []
    valid_idx_list = []
    test_idx_list = []
    for n in df.groupby('user').size():
        train_idx_list.append(extract_subseq(n-2, hist_min_len, hist_max_len, n_step) + offset)
        valid_idx_list.append(np.add([max(0, n-2 - hist_max_len), n-2, n-2], offset))
        test_idx_list.append(np.add([max(0, n-1 - hist_max_len), n-1, n-1], offset))
        offset += n
    train_idx_list = np.concatenate(train_idx_list)
    valid_idx_list = np.stack(valid_idx_list)
    test_idx_list = np.stack(test_idx_list)
    return train_idx_list, valid_idx_list, test_idx_list


class HistDatasetv2(Dataset):

    def __init__(self, df, idx_list, hist_max_len, reverse=True):
        self.reverse = reverse
        self.data = df.values  # [user, item, timestamp]
        self.idx_list = idx_list
        self.hist_max_len = hist_max_len
        self.max_timestamp = self.data[:, 2].max()

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        l, r, t = self.idx_list[item]
        submat = self.data[l:r]
        h_uid, h_items, h_ts = submat.T
        n = len(h_items)
        uid, t_item, t_t = self.data[t]
        assert np.all(h_uid == uid)

        if self.reverse:
            if n < self.hist_max_len:
                h_items = np.pad(h_items, [self.hist_max_len - n, 0], 'constant', constant_values=0)
                h_ts = np.pad(h_ts, [self.hist_max_len - n, 0], 'constant', constant_values=self.max_timestamp)
            return uid, h_items[::-1].copy(), h_ts[::-1].copy(), t_item, t_t
        else:
            if n < self.hist_max_len:
                h_items = np.pad(h_items, [0, self.hist_max_len - n], 'constant', constant_values=0)
                h_ts = np.pad(h_ts, [0, self.hist_max_len - n], 'constant', constant_values=self.max_timestamp)
            return uid, h_items, h_ts, t_item, t_t


def load_dataloader_v2(args, reverse=True):
    root = pathlib.Path('data/')
    df = pd.read_csv(root / f'{args.dataset}.csv', header=None, names=['user', 'item', 'timestamp'])
    df = df.sort_values(['user', 'timestamp'], ascending=[True, True])

    n_users, n_items, _ = df.max(0) + 1
    black_list = df.groupby('user').apply(lambda subdf: subdf.item.values).to_dict()

    train_idx_list, valid_idx_list, test_idx_list = get_idx(df, args.hist_min_len, args.hist_max_len, args.n_step)
    train_ds = HistDatasetv2(df, train_idx_list, args.hist_max_len, reverse)
    valid_ds = HistDatasetv2(df, valid_idx_list, args.hist_max_len, reverse)
    test_ds = HistDatasetv2(df, test_idx_list, args.hist_max_len, reverse)
    train_dl = DataLoaderX(train_ds, args.train_bs, pin_memory=True,
                           shuffle=True, drop_last=True, num_workers=args.n_workers)
    valid_dl = DataLoaderX(valid_ds, args.eval_bs, pin_memory=True, num_workers=args.n_workers)
    test_dl = DataLoaderX(test_ds, args.eval_bs, pin_memory=True, num_workers=args.n_workers)
    return train_dl, valid_dl, test_dl, n_items, black_list

