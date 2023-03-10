import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb
import shutil
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import MultiClassMetric, AverageMeter
from models import *

import tqdm
import logging
import importlib
from utils.logger import config_logger
from utils import builder


# import torch.backends.cudnn as cudnn
# cudnn.deterministic = True
# cudnn.benchmark = False


def train_fp16(epoch, end_epoch, args, model, train_loader, optimizer, scheduler, logger, log_frequency):
    scaler = torch.cuda.amp.GradScaler()
    rank = torch.distributed.get_rank()
    model.train()

    losses = AverageMeter()
    print('FP16 Train mode!')
    for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target) in tqdm.tqdm(enumerate(train_loader)):
        # pdb.set_trace()
        with torch.cuda.amp.autocast():
            loss = model(pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        torch.distributed.reduce(loss, 0)
        if rank == 0:
            losses.update(loss.item() / torch.distributed.get_world_size())
            if i % log_frequency == 0 or i == len(train_loader) - 1:
                string = 'Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}'.format(epoch, end_epoch, \
                                                                                 i, len(train_loader),
                                                                                 optimizer.state_dict()['param_groups'][
                                                                                     0]['lr'])
                string = string + '; loss: {:.6f} / {:.6f}'.format(losses.val, losses.avg)
                logger.info(string)


def train(epoch, end_epoch, args, model, train_loader, optimizer, scheduler, logger, log_frequency):
    rank = torch.distributed.get_rank()
    model.train()

    losses = AverageMeter()
    for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target) in tqdm.tqdm(enumerate(train_loader)):
        # pdb.set_trace()
        loss = model(pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.distributed.reduce(loss, 0)
        if rank == 0:
            losses.update(loss.item() / torch.distributed.get_world_size())
            if i % log_frequency == 0 or i == len(train_loader) - 1:
                string = 'Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}'.format(epoch, end_epoch, \
                                                                                 i, len(train_loader),
                                                                                 optimizer.state_dict()['param_groups'][
                                                                                     0]['lr'])
                string = string + '; loss: {:.6f} / {:.6f}'.format(losses.val, losses.avg)
                logger.info(string)


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()

    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    os.system('mkdir -p {}'.format(model_prefix))

    # start logging
    shutil.copyfile('config' + '/' + prefix + ".py", save_path + '/' + prefix + ".py")
    config_logger(os.path.join(save_path, "log.txt"))
    logger = logging.getLogger()

    # reset dist
    device = torch.device('cuda:{}'.format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # reset random seed
    seed = rank * pDataset.Train.num_workers + 50051
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # define dataloader
    train_dataset = eval('datasets.{}.DataloadTrain'.format(pDataset.Train.data_src))(pDataset.Train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=pGen.batch_size_per_gpu,
                              shuffle=(train_sampler is None),
                              num_workers=pDataset.Train.num_workers,
                              sampler=train_sampler,
                              pin_memory=True)

    print("rank: {}/{}; batch_size: {}".format(rank, world_size, pGen.batch_size_per_gpu))

    # define model
    base_net = eval(pModel.prefix)(pModel)

    base_net = nn.SyncBatchNorm.convert_sync_batchnorm(base_net)
    model = torch.nn.parallel.DistributedDataParallel(base_net.to(device),
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)

    # define optimizer
    optimizer = builder.get_optimizer(pOpt, model)

    # define scheduler
    per_epoch_num_iters = len(train_loader)
    scheduler = builder.get_scheduler(optimizer, pOpt, per_epoch_num_iters)

    if rank == 0:
        logger.info(model)
        logger.info(optimizer)
        logger.info(scheduler)

    # load pretrain model
    pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(pModel.pretrain.pretrain_epoch))
    if os.path.exists(pretrain_model):
        checkpoint = torch.load(pretrain_model, map_location='cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        pOpt.schedule.begin_epoch = checkpoint['epoch'] + 1

        # base_net.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
        logger.info("Load model from {}".format(pretrain_model))

    # start training
    for epoch in range(pOpt.schedule.begin_epoch, pOpt.schedule.end_epoch):
        train_sampler.set_epoch(epoch)
        if pGen.fp16:
            train_fp16(epoch, pOpt.schedule.end_epoch, args, model, train_loader, optimizer, scheduler, logger,
                       pGen.log_frequency)
        else:
            train(epoch, pOpt.schedule.end_epoch, args, model, train_loader, optimizer, scheduler, logger,
                  pGen.log_frequency)

        # save model
        if rank == 0:
            save_dict = {
                'epoch': epoch,  # after training one epoch, the start_epoch should be epoch+1
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model_state_dict': model.module.state_dict()
            }
            torch.save(save_dict, os.path.join(model_prefix, '{}-model.pth'.format(epoch)))

            # torch.save(model.module.state_dict(), os.path.join(model_prefix, '{}-model.pth'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', default='config/wce.py', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)
