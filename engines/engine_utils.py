import datetime
import json
import pathlib
import random

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from ignite.handlers import TerminateOnNan, EarlyStopping, ModelCheckpoint, Timer
from ignite.metrics import Loss, RunningAverage
from ignite.contrib.handlers import CosineAnnealingScheduler, CustomPeriodicEvent, ConcatScheduler, LinearCyclicalScheduler
from ignite.contrib.handlers.tensorboard_logger import *

from engines.metrics import MRR, NDCG, HitRate


def make_slanted_triangular_lr_scheduler(optimizer, n_events, lr_max, frac=0.1, ratio=32):
    n1 = int(n_events * frac)
    n2 = n_events - n1
    scheduler_1 = LinearCyclicalScheduler(optimizer, 'lr', start_value=lr_max / ratio, end_value=lr_max,
                                          cycle_size=n1 * 2)
    scheduler_2 = LinearCyclicalScheduler(optimizer, 'lr', start_value=lr_max, end_value=lr_max / ratio,
                                          cycle_size=n2 * 2)
    return ConcatScheduler([scheduler_1, scheduler_2], durations=[n1, ])


def make_metric_handlers():
    metrics = dict()
    # With Black List
    metrics['MRR'] = MRR(output_transform=lambda o: o[-2])
    metrics['NDCG'] = NDCG(output_transform=lambda o: o[-2])
    for k in [1, 5, 10, 20, 25, 50, 100]:
        metrics[f'MRR@{k}'] = MRR(k, output_transform=lambda o: o[-2])
        metrics[f'NDCG@{k}'] = NDCG(k, output_transform=lambda o: o[-2])
        metrics[f'HR@{k}'] = HitRate(k, output_transform=lambda o: o[-2])
    # Without Black List
    metrics['MRR_raw'] = MRR(output_transform=lambda o: o[-1])
    metrics['NDCG_raw'] = NDCG(output_transform=lambda o: o[-1])
    for k in [1, 5, 10, 20, 25, 50, 100]:
        metrics[f'MRR@{k}_raw'] = MRR(k, output_transform=lambda o: o[-1])
        metrics[f'NDCG@{k}_raw'] = NDCG(k, output_transform=lambda o: o[-1])
        metrics[f'HR@{k}_raw'] = HitRate(k, output_transform=lambda o: o[-1])

    def loss_fn(logits, ids):
        logits = torch.log_softmax(logits, dim=1)
        xentropy = -logits[range(logits.size(0)), ids].mean()
        return xentropy

    metrics['Loss'] = Loss(loss_fn, output_transform=lambda o: (o[0], o[1]))
    return metrics


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_experiment(args, name):
    seed_all(args.seed)
    device = torch.device('cuda', args.cuda_id)
    time_info = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    save_dir = pathlib.Path(f'./results/{args.dataset}/{name}/{time_info}')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(args)
    print(f'{args.dataset}/{name}/{time_info}')

    with open(save_dir / 'settings.json', 'w') as fh:
        arg_dict = vars(args)
        arg_dict['name'] = name
        json.dump(arg_dict, fh, sort_keys=True, indent=4)
    return save_dir, device


def save_state(save_dir, state, args):
    data = {'valid': state.valid_metrics_history,
            'test': state.test_metrics}
    json.dump(data, open(save_dir / 'metrics.json', 'w'), sort_keys=True, indent=4)


def add_events(engines, dataloaders, model, optimizer, device, save_dir, args):
    trainer, valid_evaluator, test_evaluator = engines
    train_dl, valid_dl, test_dl = dataloaders

    if args.valid_on == 'Loss':
        score_fn = lambda engine: -engine.state.metrics[args.valid_on]
    elif args.valid_on == 'Product':
        score_fn = lambda engine: engine.state.metrics['MRR'] * engine.state.metrics['HR@10']
    elif args.valid_on == 'RMS':
        score_fn = lambda engine: engine.state.metrics['MRR'] ** 2 + engine.state.metrics['HR@10'] ** 2
    else:
        score_fn = lambda engine: engine.state.metrics[args.valid_on]

    # LR Scheduler
    if args.lr_scheduler == 'restart':
        scheduler = CosineAnnealingScheduler(optimizer, 'lr',
                                             start_value=args.lr,
                                             end_value=args.lr*0.01,
                                             cycle_size=len(train_dl),
                                             cycle_mult=args.cycle_mult)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler, 'lr_scheduler')
    elif args.lr_scheduler == 'triangle':
        scheduler = make_slanted_triangular_lr_scheduler(optimizer, n_events=args.n_epochs * len(train_dl),
                                                         lr_max=args.lr)
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler, 'lr_scheduler')
    elif args.lr_scheduler == 'none':
        pass
    else:
        raise NotImplementedError

    # EarlyStopping
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    valid_evaluator.add_event_handler(Events.COMPLETED,
                                      EarlyStopping(args.patience, score_function=score_fn, trainer=trainer) )
    # Training Loss
    RunningAverage(output_transform=lambda x: x, alpha=args.avg_alpha).attach(trainer, 'loss')
    # Checkpoint
    ckpt_handler = ModelCheckpoint(save_dir, 'best', score_function=score_fn, score_name=args.valid_on, n_saved=1)
    valid_evaluator.add_event_handler(Events.COMPLETED, ckpt_handler, {'model': model})
    # Timer
    timer = Timer(average=True)
    timer.attach(trainer, resume=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)
    # Progress Bar
    if args.pbar:
        pbar = ProgressBar()
        pbar.attach(trainer, ['loss'])
        log_msg = pbar.log_message
    else:
        log_msg = print

    cpe_valid = CustomPeriodicEvent(n_epochs=args.valid_every)
    cpe_valid.attach(trainer)
    valid_metrics_history = []
    @trainer.on(getattr(cpe_valid.Events, f'EPOCHS_{args.valid_every}_COMPLETED'))
    def evaluate_on_valid(engine):
        state = valid_evaluator.run(valid_dl)
        metrics = state.metrics
        valid_metrics_history.append(metrics)
        msg = f'Epoch: {engine.state.epoch:3d} AvgTime: {timer.value():3.1f}s TrainLoss: {engine.state.metrics["loss"]:.4f} '
        msg += ' '.join([f'{k}: {v:.4f}' for k, v in metrics.items() if k in ['Loss', 'MRR', 'HR@10']])
        log_msg(msg)

    @trainer.on(Events.COMPLETED)
    def evaluate_on_test(engine):
        pth_file = [f for f in pathlib.Path(save_dir).iterdir() if f.name.endswith('pth')][0]
        log_msg(f'Load Best Model: {str(pth_file)}')
        model.load_state_dict(torch.load(pth_file, map_location=device))
        # Rerun on Valid for log.
        valid_state = valid_evaluator.run(valid_dl)
        engine.state.valid_metrics = valid_state.metrics
        # Test
        test_state = test_evaluator.run(test_dl)
        engine.state.test_metrics = test_state.metrics
        engine.state.valid_metrics_history = valid_metrics_history
        msg = f'[Test] '
        msg += ' '.join([f'{k}: {v:.4f}' for k, v in test_state.metrics.items() if k in ['Loss', 'MRR', 'HR@10']])
        log_msg(msg)

    # Tensorboard
    if args.tensorboard:
        tb_logger = TensorboardLogger(log_dir=str(save_dir / 'tb_log'))
        # Loss
        tb_logger.attach(trainer,
                         log_handler=OutputHandler(tag='training', output_transform=lambda x: x),
                         event_name=Events.ITERATION_COMPLETED)
        # Metrics
        tb_logger.attach(valid_evaluator,
                         log_handler=OutputHandler(tag='validation',
                                                   metric_names=['Loss', 'MRR', 'HR@10'],
                                                   another_engine=trainer),
                         event_name=Events.EPOCH_COMPLETED)
        # Optimizer
        tb_logger.attach(trainer,
                         log_handler=OptimizerParamsHandler(optimizer),
                         event_name=Events.ITERATION_STARTED)
        # Parameters
        # tb_logger.attach(trainer,
        #                  log_handler=WeightsScalarHandler(model),
        #                  event_name=Events.ITERATION_COMPLETED)
        # tb_logger.attach(trainer,
        #                  log_handler=GradsScalarHandler(model),
        #                  event_name=Events.ITERATION_COMPLETED)

        @trainer.on(Events.COMPLETED)
        def close_tb(engine):
            tb_logger.close()
