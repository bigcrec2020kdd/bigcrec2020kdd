import argparse


def add_common_env_args(parser: argparse.ArgumentParser):
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--avg_alpha', type=float, default=0.8)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--pbar', type=t_or_f, default=True)
    parser.add_argument('--tensorboard', type=t_or_f, default=True)


def add_optim_args(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--valid_on', type=str, default='RMS')
    parser.add_argument('--lr_scheduler', type=str, default='triangle')
    parser.add_argument('--cycle_mult', type=int, default=2)
    parser.add_argument('--valid_every', type=int, default=10)
    parser.add_argument('--gradient_clip', type=float, default=5.)


def add_hist_dataloader_args(parser: argparse.ArgumentParser):
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='garden_5')
    parser.add_argument('--hist_min_len', type=int, default=2)
    parser.add_argument('--hist_max_len', type=int, default=20)
    parser.add_argument('--n_step', type=int, default=1)
    parser.add_argument('--train_bs', type=int, default=512)
    parser.add_argument('--eval_bs', type=int, default=128)


def add_time_processing_args(parser: argparse.ArgumentParser):
    parser.add_argument('--tau', type=int, default=21600)
    parser.add_argument('--n_buckets', type=int, default=10)
    parser.add_argument('--use_time', type=str, default='bucket')


def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass  #error condition maybe?
