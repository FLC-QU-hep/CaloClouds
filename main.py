from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.dataset import *
from utils.misc import *
from models.CaloClouds import *
from configs import Configs

cfg = Configs()

if cfg.log_comet:
    experiment = Experiment(
        project_name="point-cloud",
    )
    experiment.log_parameters(cfg.__dict__)


seed_all()

# Logging
log_dir = get_new_log_dir(cfg.logdir, prefix=cfg.name, postfix='_' + cfg.tag if cfg.tag is not None else '')
ckpt_mgr = CheckpointManager(log_dir)

# Datasets and loaders
if cfg.dataset == 'x36_grid' or cfg.dataset ==  'clustered':
    train_dset = PointCloudDataset(
        file_path=cfg.dataset_path,
        bs=cfg.train_bs,
        quantized_pos=cfg.quantized_pos
    )
dataloader = DataLoader(
    train_dset,
    batch_size=1,
    num_workers=cfg.workers,
    shuffle=cfg.shuffle
)
val_dset = []


# Model
model = CaloClouds(cfg).to(cfg.device)

optimizer = torch.optim.Adam(
            [
            {'params': model.encoder.parameters()}, 
            {'params': model.diffusion.parameters()},
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )
optimizer_flow = torch.optim.Adam(
            [
            {'params': model.flow.parameters()}, 
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )

scheduler = get_linear_scheduler(optimizer, start_epoch=cfg.sched_start_epoch, end_epoch=cfg.sched_end_epoch, start_lr=cfg.lr, end_lr=cfg.end_lr)
scheduler_flow = get_linear_scheduler(optimizer_flow, start_epoch=cfg.sched_start_epoch, end_epoch=cfg.sched_end_epoch, start_lr=cfg.lr, end_lr=cfg.end_lr)


# Train, validate and test
def train(batch, it):
    # Load data
    x = batch['event'][0].float().to(cfg.device)
    e = batch['energy'][0].float().to(cfg.device)
    n = batch['points'][0].float().to(cfg.device)
    # Reset grad and model state
    optimizer.zero_grad()
    optimizer_flow.zero_grad()
    model.train()

    # Normalize conditioning features
    if cfg.norm_cond:
        e = e / 100 * 2 -1   # max incident energy: 100 GeV
        n = n / cfg.max_points * 2  - 1
    cond_feats = torch.cat([e,n], -1) 
    
    # Forward
    if cfg.log_comet:
        loss, loss_flow = model.get_loss(x, cond_feats, kl_weight=cfg.kl_weight, writer=experiment, it=it, kld_min=cfg.kld_min)
    else:
        loss, loss_flow = model.get_loss(x, cond_feats, kl_weight=cfg.kl_weight, writer=None, it=it, kld_min=cfg.kld_min)

    # Backward and optimize
    loss.backward()
    loss_flow.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()
    optimizer_flow.step()
    scheduler.step()
    scheduler_flow.step()

    if it % 10 == 0:
        print('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
            it, loss.item(), orig_grad_norm, cfg.kl_weight
        ))
        if cfg.log_comet:
            experiment.log_metric('train/loss', loss, it)
            experiment.log_metric('train/loss_flow', loss_flow, it)
            experiment.log_metric('train/kl_weight', cfg.kl_weight, it)
            experiment.log_metric('train/lr', optimizer.param_groups[0]['lr'], it)
            experiment.log_metric('train/lr_flow', optimizer_flow.param_groups[0]['lr'], it)
            experiment.log_metric('train/grad_norm', orig_grad_norm, it)

# Main loop
print('Start training...')

stop = False
it = 1
while not stop:
    for batch in dataloader:
        it += 1
        train(batch, it)
        if it % cfg.val_freq == 0 or it == cfg.max_iters:
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'optimizer_flow': optimizer_flow.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scheduler_flow': scheduler_flow.state_dict(),
            }
            ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)
        if it >= cfg.max_iters:
            stop = True
            break
