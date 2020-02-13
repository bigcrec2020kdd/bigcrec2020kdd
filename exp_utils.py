import argparse

from ignite.engine import Engine
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch import optim

from engines.engine_utils import *
from models import *
from dataloaders.hist_dataset_seq import load_seq_dataloader
from dataloaders.hist_dataset_v2 import load_dataloader_v2
from dataloaders.hist_dataset_graph import load_graph_dataloader
from args_utils import *

SEQUENTIAL_MODELS = {'SASRec', 'NextItNet', 'GRU4RecSeq'}


def get_model(name, n_items, args):
    if name in {'ATEM', 'BiGCrec', 'BiGCrecV2', 'BiGCrecV3', 'BiGCrecV4'}:
        if args.use_time == 'bucket':
            n_buckets = args.n_buckets
        elif args.use_time == 'simple':
            n_buckets = args.hist_max_len + 1
        elif args.use_time == 'none':
            n_buckets = 1
        else:
            raise NotImplementedError

    if name == 'SASRec':
        model = SASRec(n_items, args.hist_max_len, emb_size=args.emb_size,
                       n_heads=args.n_heads, n_blocks=args.n_blocks, n_samples=args.n_samples,
                       dropout=args.dropout, attn_dropout=args.attn_dropout)
    elif name == 'ATEM':
        model = ATEM(n_items=n_items, n_buckets=n_buckets, emb_size=args.emb_size,
                     n_samples=args.n_samples, tie_emb=args.tie_emb)
    elif name == 'Caser':
        model = Caser(n_items, emb_size=args.emb_size, n_v_convs=args.n_v_convs,
                      n_h_convs=args.n_h_convs, hist_max_len=args.hist_max_len,
                      dropout=args.dropout, n_samples=args.n_samples)
    elif name == 'GRU4RecSimple':
        model = GRU4RecSimple(n_items, emb_size=args.emb_size, hidden_size=args.hidden_size,
                              n_layers=args.n_layers, emb_dropout=args.emb_dropout,
                              hidden_dropout=args.hidden_dropout, n_samples=args.n_samples)
    elif name == 'GRU4RecSeq':
        model = GRU4RecSeq(n_items, emb_size=args.emb_size, hidden_size=args.hidden_size,
                           n_layers=args.n_layers, emb_dropout=args.emb_dropout,
                           hidden_dropout=args.hidden_dropout, n_samples=args.n_samples)
    elif name == 'NextItNet':
        model = NextItNet(n_items, emb_size=args.emb_size,
                          dilations=args.dilations, n_samples=args.n_samples)
    elif name == 'SRGNN':
        model = SRGNN(n_items, emb_size=args.emb_size, n_samples=args.n_samples,
                      nonhybrid=args.nonhybrid, step=args.step)
    elif name == 'BiGCrec':
        model = BiGCNet(n_items=n_items, n_buckets=n_buckets, emb_size=args.emb_size,
                        n_topics=args.n_topics, n_heads=args.n_heads, n_branches=args.n_branches,
                        n_layers=args.n_layers, n_samples_per_layer=args.n_samples_per_layer, dropout=args.dropout,
                        slope=args.slope, temporal_concat=args.temporal_concat,
                        score_layer_hidden=args.score_layer_hidden, dot_attn=args.dot_attn,
                        use_mid_layer=args.use_mid_layer, bias=args.bias)
    elif name == 'BiGCrecV2':
        model = BiGCrec_v2(n_items=n_items, n_buckets=n_buckets, emb_size=args.emb_size, n_layers=args.n_layers,
                           n_topics=args.n_topics, n_heads=args.n_heads, n_samples_train=args.n_samples_train,
                           n_samples_eval=args.n_samples_eval, dropout=args.dropout, attn_dropout=args.attn_dropout,
                           use_mid_layer=args.use_mid_layer, use_full=args.use_full)
    elif name == 'BiGCrecV3':
        model = BiGCrec_v3(n_items=n_items, n_buckets=n_buckets, emb_size=args.emb_size,
                           n_topics=args.n_topics, n_heads=args.n_heads, n_branches=args.n_branches,
                           n_layers=args.n_layers, n_samples_per_layer=args.n_samples_per_layer, dropout=args.dropout,
                           slope=args.slope, use_mid_layer=args.use_mid_layer, bias=args.bias)
    elif name == 'BiGCrecV4':
        model = BiGCrec_v4(n_items=n_items, n_buckets=n_buckets, emb_size=args.emb_size,
                           n_topics=args.n_topics, n_heads=args.n_heads, n_branches=args.n_branches,
                           n_layers=args.n_layers, n_samples_per_layer=args.n_samples_per_layer, dropout=args.dropout,
                           slope=args.slope, use_mid_layer=args.use_mid_layer, bias=args.bias)
    else:
        raise NotImplementedError

    return model


def get_dataloader(name, args):
    if name in SEQUENTIAL_MODELS:
        train_dl, valid_dl, test_dl, n_items, black_list = load_seq_dataloader(args)
    elif name == 'GRU4RecSimple':
        train_dl, valid_dl, test_dl, n_items, black_list = load_dataloader_v2(args, reverse=False)
    elif name == 'SRGNN':
        train_dl, valid_dl, test_dl, n_items, black_list = load_graph_dataloader(args)
    else:
        train_dl, valid_dl, test_dl, n_items, black_list = load_dataloader_v2(args)
    return train_dl, valid_dl, test_dl, n_items, black_list


def create_default_trainer_seq(model, optimizer, device, args):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        _, h_iids, seq_t_iids, _2 = [x.to(device) for x in batch]
        loss = model.nce_loss(h_iids, seq_t_iids)
        loss.backward()
        if args.gradient_clip > 0:
            clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_default_evaluator_seq(model, metrics, device, valid, black_list, args):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            uids, h_iids, _, t_iids = [x.to(device) for x in batch]
            logits = model(h_iids)
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


def run(name, args, trainer_fn=None, evaluator_fn=None):
    save_dir, device = init_experiment(args, name)

    train_dl, valid_dl, test_dl, n_items, black_list = get_dataloader(name, args)
    model = get_model(name, n_items, args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if trainer_fn is None:
        if name in SEQUENTIAL_MODELS:
            trainer_fn = create_default_trainer_seq
    if evaluator_fn is None:
        if name in SEQUENTIAL_MODELS:
            evaluator_fn = create_default_evaluator_seq

    trainer = trainer_fn(model, optimizer, device=device, args=args)
    valid_evaluator = evaluator_fn(model, metrics=make_metric_handlers(), device=device, valid=True,
                                   black_list=black_list, args=args)
    test_evaluator = evaluator_fn(model, metrics=make_metric_handlers(), device=device, valid=False,
                                  black_list=black_list, args=args)

    add_events(engines=[trainer, valid_evaluator, test_evaluator],
               dataloaders=[train_dl, valid_dl, test_dl],
               model=model, optimizer=optimizer,
               device=device, save_dir=save_dir, args=args)
    state = trainer.run(train_dl, args.n_epochs)

    save_state(save_dir, state, args)

    return state
