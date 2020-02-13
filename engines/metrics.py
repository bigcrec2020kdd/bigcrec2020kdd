import torch
import numpy as np

from ignite.metrics.metric import Metric


class BaseRankMetric(Metric):

    def __init__(self, k=None, output_transform=lambda x: x):
        self.k = float('Inf') if k is None else k
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._cum_score = 0.0
        self._n_examples = 1e-9

    def compute(self):
        return self._cum_score / self._n_examples


class NDCG(BaseRankMetric):

    def update(self, output):
        ranks = output[output < self.k + 1]
        self._cum_score += (1. / torch.log2(ranks + 1.)).sum().item()
        self._n_examples += output.size(0)


class MRR(BaseRankMetric):

    def update(self, output):
        ranks = output[output < self.k + 1]
        self._cum_score += (1. / ranks).sum().item()
        self._n_examples += output.size(0)


class HitRate(BaseRankMetric):

    def update(self, output):
        ranks = output[output < self.k + 1]
        self._cum_score += ranks.size(0)
        self._n_examples += output.size(0)


# For Test
def _update_metric_info(info, rank):
    for m in info.keys():
        if m.startswith('MRR'):
            k = int(m[4:])
            if rank <= k:
                info[m] += 1 / rank
        elif m.startswith('HitRate'):
            k = int(m[8:])
            if rank <= k:
                info[m] += 1
        elif m.startswith('NDCG'):
            k = int(m[5:])
            if rank <= k:
                info[m] += 1 / np.log2(rank + 1)
    return info


def test():
    outputs = [torch.rand(100, 100).cuda().argsort()[:, 0].float() + 1 for _ in range(5)]
    metric_classes = [NDCG, MRR, HitRate]
    for k in [1, 5, 20, 100000]:
        info = {f'{m.__name__}@{k}': 0.0 for m in metric_classes}
        for r in torch.cat(outputs):
            _update_metric_info(info, r.item())
        for key, v in info.items():
            info[key] /= sum([x.size(0) for x in outputs])
        metrics_batch = [m(k) for m in metric_classes]
        metrics_full = [m(k) for m in metric_classes]
        for m_batch, m_full in zip(metrics_batch, metrics_full):
            for o in outputs:
                m_batch.update(o)
            m_full.update(torch.cat(outputs))
        vals_batch = [m_batch.compute() for m_batch in metrics_batch]
        vals_full = [m_full.compute() for m_full in metrics_full]
        vals_np = [info[f'{m.__name__}@{k}'] for m in metric_classes]
        print(np.allclose(vals_batch, vals_full) , np.allclose(vals_batch, vals_np))


if __name__ == '__main__':
    test()
