
# ~~CPGNet: Cascade Point-Grid Fusion Network for Real-Time LiDAR Semantic Segmentation~~

<div align="center">
  <img src="assert/CPGNet.png"/>
</div>

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
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --config config/ohem.py

### Single-gpu ###
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --config config/ohem.py
~~~

## Evaluation
~~~
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 evaluate.py --config config/ohem.py --start_epoch 0 --end_epoch 47
~~~

## Find best epoch
~~~
python find_best_metric.py --config config/ohem.py
~~~

## Pretrained Models and Logs
Models have been uploaded to this [Google Drive folder](https://drive.google.com/drive/folders/18DsT-int3XuNRmQ1W0FkNnZ3PaGRohpn?usp=sharing).

| CPGNet (stage=1) | Loss | Batch_Size * GPUS | mIoU |
| :---------------: | :---------------: | :---------------: | :---------------: |
| Our Reproduced | OHEM |      4 * 1 (FP16 on 3090)       |       61.5        |
| Our Reproduced | OHEM |      6 * 2 (FP16 on 3090)       |       61.5        |
| Our Reproduced | WCE |      6 * 2 (FP16 on 3090)       |       60.1        |
| Paper Reported | -- |       ~~2 * 8 (FP32 on 2080ti)~~     |       **62.5**        |

**Note:** 
- The model corresponding to the codebase should be CPGNet (stage=1), which the original paper reported a performance of **62.5**.
- The performance gap may come from two reasons: 1) The original paper uses a **2 * 8** batch size for training while we use smaller batches. 2). The original paper uses the **CutMix** data-aug, which should correspond to [this part](https://github.com/huixiancheng/No-CPGNet/blob/e161450f6f81d0bed8e03ae59fbcabeb03602458/datasets/data.py#L183-L184) of the codebase. Since I do not yet understand the ops, it is not activated. This also has a performance impact.

Below are known issues listed:
- Training with `OHEM` loss, the loss converges slowly. The situation is similar after switching to `WCE`. 
- From the validation results (Oscillation at a value), the model seems to be under-fitted.
- Not sure if the loss of the [BEV part](https://github.com/huixiancheng/No-CPGNet/blob/4e55053f2a883de7aa0158ed3fccb9b5f89c46c8/models/cpgnet.py#L164-L167) is used in CPGNet.
- During validation, TTA is on by default, so the accuracy will be higher.
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