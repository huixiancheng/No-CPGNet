import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import MultiClassMetric
from models import *

import tqdm
import importlib
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True


def val_fp16(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    print('FP16 inference mode!')
    model.eval()
    f = open(os.path.join(save_path, 'record_fp16_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, fname_pcd) in tqdm.tqdm(enumerate(val_loader)):
            with torch.cuda.amp.autocast():
                # pred_cls = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(), pcds_sphere_coord.squeeze(0).cuda())
                pred_cls = model.infer(pcds_xyzi.cuda(), pcds_coord.cuda(), pcds_sphere_coord.cuda())
            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pcds_target = pcds_target[0, :, 0].contiguous()
            
            valid_point_num = pcds_target.shape[0]
            criterion_cate.addBatch(pcds_target, pred_cls[:valid_point_num])
        
        #record segmentation metric
        metric_cate = criterion_cate.get_metric()
        string = 'Epoch {}'.format(epoch)
        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])
        
        f.write(string + '\n')
        f.close()


def val(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    
    model.eval()
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, fname_pcd) in tqdm.tqdm(enumerate(val_loader)):
            # pred_cls = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(), pcds_sphere_coord.squeeze(0).cuda())
            pred_cls = model.infer(pcds_xyzi.cuda(), pcds_coord.cuda(),pcds_sphere_coord.cuda())
            
            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pcds_target = pcds_target[0, :, 0].contiguous()
            
            valid_point_num = pcds_target.shape[0]
            criterion_cate.addBatch(pcds_target, pred_cls[:valid_point_num])
        
        #record segmentation metric
        metric_cate = criterion_cate.get_metric()
        string = 'Epoch {}'.format(epoch)
        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])
        
        f.write(string + '\n')
        f.close()


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    # reset dist
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # reset random seed
    seed = rank * pDataset.Val.num_workers + 50051
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # define dataloader
    val_dataset = eval('datasets.{}.DataloadVal'.format(pDataset.Val.data_src))(pDataset.Val)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=pDataset.Val.num_workers,
                            pin_memory=True)
    
    # define model
    model = eval(pModel.prefix)(pModel)
    model.cuda()
    model.eval()


    for epoch in range(args.start_epoch, args.end_epoch + 1, world_size):
        if (epoch + rank) < (args.end_epoch + 1):
            pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(epoch + rank))
            print(pretrain_model)
            checkpoint = torch.load(pretrain_model, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            # model.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
            if pGen.fp16:
                val_fp16(epoch + rank, model, val_loader, pGen.category_list, save_path, rank)
            else:
                val(epoch + rank, model, val_loader, pGen.category_list, save_path, rank)

def seed_torch(seed=1024):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = False
    print("We use the seed: {}".format(seed))

if __name__ == '__main__':
    seed_torch(seed=520)
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', default='config/wce.py', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=49)
    
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)