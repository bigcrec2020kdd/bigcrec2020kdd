import argparse

from ignite.engine import Engine
from torch.nn.utils.clip_grad import clip_grad_norm_

from models.model_utils import *
from args_utils import *
from exp_utils import run


def create_bigcrec_trainer(model, optimizer, device, args):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        _, h_iids, t_iids, time_ids = prepare_batch(batch, device, args)
        loss = model.mini_batch_forward(h_iids, time_ids, t_iids)
        loss.backward()
        if args.gradient_clip > 0:
            clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        return loss.item()
    return Engine(_update)


def create_bigcrec_evaluator(model, metrics, device, valid, black_list, args):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            uids, h_iids, t_iids, time_ids = prepare_batch(batch, device, args)
            # sample
            logits = model.sample_forward_full_return(h_iids, time_ids)
            # logits = model(h_iids, time_ids)
            # Get Ranks
            if valid:
                batch_black_list = [black_list[u.item()][:-2] for u in uids]
            else:
                batch_black_list = [black_list[u.item()][:-1] for u in uids]
            ranks = get_ranks(logits, t_iids, batch_black_list)
            ranks_raw = get_ranks(logits, t_iids)
            return logits, t_iids, ranks.float(), ranks_raw.float()
    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def get_bigcrec_parser():
    parser = argparse.ArgumentParser()

    add_common_env_args(parser)
    add_hist_dataloader_args(parser)
    add_optim_args(parser)
    add_time_processing_args(parser)

    # Model
    parser.add_argument('--emb_size', type=int, default=16)
    parser.add_argument('--n_topics', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--n_branches', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_samples_per_layer', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--slope', type=float, default=0.2)
    parser.add_argument('--bias', type=t_or_f, default=False)
    parser.add_argument('--use_mid_layer', type=t_or_f, default=True)

    return parser


if __name__ == '__main__':
    parser = get_bigcrec_parser()
    args = parser.parse_args()
    state = run('BiGCrecV3', args, create_bigcrec_trainer, create_bigcrec_evaluator)
