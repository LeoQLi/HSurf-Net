import os, sys
import argparse
import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_

from utils.misc import *
from net.network import Network
from dataset import PointCloudDataset, PatchDataset, RandomPointcloudPatchSampler


def parse_arguments():
    parser = argparse.ArgumentParser()
    ### Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_gamma', type=float, default=0.2)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--nepoch', type=int, default=999999)
    parser.add_argument('--scheduler_epoch', type=int, nargs='+', default=[200,400,600,800])
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
    ### Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--data_set', type=str, default='PCPNet', choices=['PCPNet'])
    parser.add_argument('--trainset_list', type=str, default='trainingset_whitenoise.txt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--num_knn', type=int, default=16)
    parser.add_argument('--decode_knn', type=int, default=16)
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='The number of patches sampled from each shape in an epoch')
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed)
        np.random.seed(args.seed)

    # g = torch.Generator()
    # g.manual_seed(args.seed)

    train_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='train',
            data_set=args.data_set,
            data_list=args.trainset_list,
        )
    train_set = PatchDataset(
            datasets=train_dset,
            patch_size=args.patch_size,
        )
    train_datasampler = RandomPointcloudPatchSampler(train_set, patches_per_shape=args.patches_per_shape, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            sampler=train_datasampler,
            batch_size=args.batch_size,
            num_workers=int(args.num_workers),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # generator=g,
        )

    return train_dataloader, train_datasampler


### Arguments
args = parse_arguments()
seed_all(args.seed)

assert args.gpu >= 0, "ERROR GPU ID!"
_device = torch.device('cuda:%d' % args.gpu)
PID = os.getpid()

### Datasets and loaders
print('Loading datasets ...')
train_dataloader, train_datasampler = get_data_loaders(args)
train_num_batch = len(train_dataloader)

### Model
print('Building model ...')
model = Network(num_in=args.patch_size,
                num_knn=args.num_knn,
                decode_knn=args.decode_knn,
            ).to(_device)

### Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)

### Logging
if args.logging:
    log_path, log_dir_name = get_new_log_dir(args.log_root, prefix='',
                                            postfix='_' + args.tag if args.tag is not None else '')
    sub_log_dir = os.path.join(log_path, 'log')
    os.makedirs(sub_log_dir)
    logger = get_logger(name='train(%d)(%s)' % (PID, log_dir_name), log_dir=sub_log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(sub_log_dir)
    log_hyperparams(writer, sub_log_dir, args)
    ckpt_mgr = CheckpointManager(os.path.join(log_path, 'ckpts'))
    # git_commit(logger=logger, log_dir=sub_log_dir, git_name=log_dir_name)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()

refine_epoch = -1
if args.resume != '':
    assert os.path.exists(args.resume), 'ERROR path: %s' % args.resume
    logger.info('Resume from: %s' % args.resume)

    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['state_dict'])
    refine_epoch = ckpt['others']['epoch']

    logger.info('Load pretrained mode: %s' % args.resume)

if args.logging:
    code_dir = os.path.join(log_path, 'code')
    os.makedirs(code_dir, exist_ok=True)
    os.system('cp %s %s' % ('*.py', code_dir))
    os.system('cp -r %s %s' % ('net', code_dir))
    os.system('cp -r %s %s' % ('utils', code_dir))


### Arguments
logger.info('Command: {}'.format(' '.join(sys.argv)))
arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
logger.info('Arguments:\n' + arg_str)
logger.info(repr(model))
logger.info('training set: %d patches (in %d batches)' %
                (len(train_datasampler), len(train_dataloader)))


def train(epoch):
    for train_batchind, batch in enumerate(train_dataloader, 0):
        pcl_pat = batch['pcl_pat'].to(_device)
        center_normal = batch['center_normal'].to(_device)  # (B, 1, 3)

        ### Reset grad and model state
        model.train()
        optimizer.zero_grad()

        ### Forward
        pred_point, weights, pred_neighbor = model(pcl_pat)
        loss, loss_tuple = model.get_loss(q_target=center_normal.squeeze(), q_pred=pred_point,
                                            pred_weights=weights, pcl_in=pcl_pat)

        ### Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        ### Logging
        s = ''
        for l in loss_tuple:
            s += '%.5f+' % l.item()
        logger.info('[Train] [%03d: %03d/%03d] | Loss: %.6f(%s) | Grad: %.6f' % (
                    epoch, train_batchind, train_num_batch-1, loss.item(), s[:-1], orig_grad_norm)
                )


def scheduler_fun():
    pre_lr = optimizer.param_groups[0]['lr']
    current_lr = pre_lr * args.lr_gamma
    if current_lr < args.lr_min:
        current_lr = args.lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    logger.info('Update learning rate: %f => %f \n' % (pre_lr, current_lr))


if __name__ == '__main__':
    logger.info('Start training ...')
    try:
        for epoch in range(1, args.nepoch+1):
            logger.info('### Epoch %d ###' % epoch)
            if epoch <= refine_epoch:
                # scheduler.step()
                if epoch in args.scheduler_epoch:
                    scheduler_fun()
                continue

            start_time = time.time()
            train(epoch)
            end_time = time.time()
            logger.info('Time cost: %.1f s \n' % (end_time-start_time))

            # scheduler.step()
            if epoch in args.scheduler_epoch:
                scheduler_fun()

            if epoch % args.interval == 0 or epoch == args.nepoch-1:
                opt_states = {
                    'epoch': epoch,
                    # 'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                }

                if args.logging:
                    ckpt_mgr.save(model, args, others=opt_states, step=epoch)

    except KeyboardInterrupt:
        logger.info('Terminating ...')
