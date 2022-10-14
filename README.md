
# ~~CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation~~


- ~~This is **N**on-**O**fficial implementation of [CPGNet](https://arxiv.org/abs/2204.09914).~~ Just a simple try and just simple reproduction based on [SMVF](https://github.com/GangZhang842/SMVF). :joy::joy::joy:
- Waiting for training results. :sleeping: :zzz:
- Here is the official [Repo](https://github.com/GangZhang842/CPGNet).

## Environment Setup
Please refer to [SMVF](https://github.com/GangZhang842/SMVF) repo. ~~**Note:** Make sure deep_point is installed.~~
## Prepare Data
Download SemanticKITTI from [official web](http://www.semantic-kitti.org/dataset.html).
## Training
~~~
### Multi-gpus ###
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --config config/config_cpg_sgd_ohem_fp16_48epoch.py

### Single-gpu ###
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

## Pretrained Models and Logs
Models have been uploaded to this [Google Drive folder](https://drive.google.com/drive/folders/18DsT-int3XuNRmQ1W0FkNnZ3PaGRohpn?usp=sharing).

| CPGNet (stage=1) | Batch_Size * GPUS | mIoU |
| :---------------: | :---------------: | :---------------: |
| Our Reproduced |       4 * 1       |       61.5        |
| Our Reproduced |       6 * 2       |       Soon        |
| Paper Reported |       2 * 8       |       62.5        |

**Note:** 
- The model corresponding to the codebase should be CPGNet (stage=1), which the original paper reported a performance of **62.5**.
- The performance gap may come from two reasons: 1) The original paper uses a **2 * 8** batch size for training while we use smaller batches. 2). The original paper uses the **CutMix** data-aug, which should correspond to [this part](https://github.com/huixiancheng/No-CPGNet/blob/e161450f6f81d0bed8e03ae59fbcabeb03602458/datasets/data.py#L183-L184) of the codebase. Since I do not yet understand the ops, it is not activated. This also has a performance impact.
- Maybe I'm missing some other details, welcome to discuss and PR if you find them.

## Citation
It should be considered to cite:
~~~
@inproceedings{li2022cpgnet,
  title={CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation},
  author={Li, Xiaoyan and Zhang, Gang and Pan, Hongyu and Wang, Zhenhua},
  booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022},
  organization={IEEE}
}
~~~