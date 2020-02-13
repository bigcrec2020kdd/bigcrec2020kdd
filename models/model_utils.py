import torch
from torch import nn
import numpy as np


def bucketize(h_ts, t_ts, tau=21600, vmax=25):
    diff = (t_ts.unsqueeze(-1) - h_ts).float() / tau
    diff = diff.masked_fill_(diff < 0, -1) + 1
    bucket_ids = torch.floor(torch.log2(diff))
    bucket_ids += 1
    # bucket_ids = bucket_ids.masked_fill(~torch.isfinite(bucket_ids), 0)  # padding index = 0
    bucket_ids = torch.clamp_min(bucket_ids, 0)
    bucket_ids = torch.clamp_max(bucket_ids.long(), vmax)
    return bucket_ids


def prepare_batch(batch, device, args):
    uids, h_iids, h_ts, t_iids, t_ts = [x.to(device) for x in batch]
    if args.use_time == 'bucket':
        time_ids = bucketize(h_ts, t_ts, tau=args.tau, vmax=args.n_buckets - 1)
    elif args.use_time == 'simple':
        time_ids = (h_iids > 0).long() * torch.arange(1, 1+h_ts.size(1), device=device).unsqueeze(0)
    elif args.use_time == 'none':
        time_ids = torch.zeros_like(h_ts)
    else:
        raise NotImplementedError
    return uids, h_iids, t_iids, time_ids


def get_ranks(scores, t_iids, blacklist=None):
    scores = scores.clone()
    if blacklist is not None:
        rows = np.concatenate([[i for _ in range(len(x))] for i, x in enumerate(blacklist)])
        cols = np.concatenate(blacklist)
        scores[rows, cols] = float('-Inf')
    scores[:, 0] = float('-Inf')
    ranks = (-scores).argsort().argsort()
    return ranks[range(scores.size(0)), t_iids] + 1


def trunc_normal_(x, mean=0., std=1.):
    # From Fast.ai
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def embedding(ni, nf, padding_idx=None):
    # From Fast.ai
    emb = nn.Embedding(ni, nf, padding_idx=padding_idx)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad():
        trunc_normal_(emb.weight, std=0.01)
    return emb
