import pandas as pd
import numpy as np
import tqdm

from collections import Counter


def encode_dataframe(input_df):
    df = input_df.copy()
    df['user'] = df.user.astype('category').cat.codes + 1
    df['item'] = df.item.astype('category').cat.codes + 1

    df = df[['user', 'item', 'timestamp']]
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['user', 'item'], inplace=True)
    return df


def remove_low_freq(input_df, min_freqs):
    df = input_df.copy()
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['user', 'item'], inplace=True)
    # drop items
    shape = df.shape
    while True:
        for col in ['item', 'user']:
            counter = Counter(df[col])
            ids, freqs = zip(*counter.items())
            ids = np.array(ids)
            freqs = np.array(freqs)
            df = df[~df[col].isin(ids[freqs < min_freqs])]
        if df.shape == shape:
            break
        shape = df.shape
    # re-index
    for col in ['item', 'user']:
        df[col] = df[col].astype('category').cat.codes + 1
    df.reset_index(drop=True, inplace=True)
    print(df.user.max(), df.item.max())
    return df
