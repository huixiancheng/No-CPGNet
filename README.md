
# ~~CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation~~


- ~~This is **N**on-**O**fficial implementation of [CPGNet](https://arxiv.org/abs/2204.09914).~~ Just a simple try and just simple reproduction based on [SMVF](https://github.com/GangZhang842/SMVF). :joy::joy::joy:
- Waiting for training results. :sleeping: :zzz:

## Environment Setup
Please refer to [SMVF](https://github.com/GangZhang842/SMVF) repo.

## Prepare Data
Download SemanticKITTI from [official web](http://www.semantic-kitti.org/dataset.html).

## Training
~~~
### Multi-gpus ###
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --config config/config_cpg_sgd_ohem_fp16_48epoch.py

### Single-gpus ###
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --config config/config_cpg_sgd_ohem_fp16_48epoch.py
~~~

## Evaluation
~~~
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 evaluate.py --config config/config_cpg_sgd_ohem_fp16_48epoch.py --start_epoch 0 --end_epoch 47
~~~

## Find best epoch
~~~
python find_best_metric.py --config config/config_cpg_sgd_ohem_fp16_48epoch.py
~~~

## Pretrained Models and Logs:
Models have been uploaded to this [Google Drive folder](https://drive.google.com/drive/folders/18DsT-int3XuNRmQ1W0FkNnZ3PaGRohpn?usp=sharing).

| Batch_Size * GPUS | mIoU |
| :---------------: | :---------------: |
|       4 * 1       |       61.5        |
|       6 * 2       |       Soon        |

**Note:** 
- The model corresponding to the codebase should be CPGNet (stage=1), which the original paper reported a performance of **62.5**.
- The performance gap may come from two reasons: 1) The original paper uses a **2 * 8** batch size for training while we use smaller batches. 2). The original paper uses the **CutMix data-aug**, which should correspond to [this part](https://github.com/huixiancheng/No-CPGNet/blob/e161450f6f81d0bed8e03ae59fbcabeb03602458/datasets/data.py#L183-L184) of the codebase. Since I do not yet understand the ops, it is not activated. This also has a performance impact.