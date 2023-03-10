import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import backbone, bird_view, range_view
from networks.backbone import get_module
import deep_point

from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax

import yaml
import copy
import pdb


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size,
                                         scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat


class AttNet(nn.Module):
    def __init__(self, pModel):
        super(AttNet, self).__init__()
        self.pModel = pModel

        self.bev_shape = list(pModel.Voxel.bev_shape)
        self.rv_shape = list(pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.dx = (pModel.Voxel.range_x[1] - pModel.Voxel.range_x[0]) / (pModel.Voxel.bev_shape[0])
        self.dy = (pModel.Voxel.range_y[1] - pModel.Voxel.range_y[0]) / (pModel.Voxel.bev_shape[1])
        self.dz = (pModel.Voxel.range_z[1] - pModel.Voxel.range_z[0]) / (pModel.Voxel.bev_shape[2])

        self.point_feat_out_channels = pModel.point_feat_out_channels

        self.build_network()
        self.build_loss()

    def build_loss(self):
        self.criterion_seg_cate = None
        print("Loss mode: {}".format(self.pModel.loss_mode))
        if self.pModel.loss_mode == 'ce':
            self.criterion_seg_cate = nn.CrossEntropyLoss(ignore_index=0)
        elif self.pModel.loss_mode == 'ohem':
            self.criterion_seg_cate = CE_OHEM(top_ratio=0.2, top_weight=4.0, ignore_index=0)
        elif self.pModel.loss_mode == 'wce':
            content = torch.zeros(self.pModel.class_num, dtype=torch.float32)
            with open('datasets/semantic-kitti.yaml', 'r') as f:
                task_cfg = yaml.load(f)
                for cl, freq in task_cfg["content"].items():
                    x_cl = task_cfg['learning_map'][cl]
                    content[x_cl] += freq

            loss_w = 1 / (content + 0.001)
            loss_w[0] = 0

            print("Loss weights from content: ", loss_w)
            self.criterion_seg_cate = nn.CrossEntropyLoss(weight=loss_w)
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')

    def build_network(self):
        # build network
        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        rv_layers = copy.deepcopy(self.pModel.RVParam.layers)
        rv_base_block = self.pModel.RVParam.base_block
        rv_grid2point = self.pModel.RVParam.rv_grid2point

        fusion_mode = self.pModel.fusion_mode
        fusion_way = self.pModel.fusion_way

        rv_feature_channel0 = rv_context_layer[0]
        rv_context_layer[0] = rv_context_layer[0]
        # network
        self.point_pre = backbone.PointNetStacker(5, rv_feature_channel0, pre_bn=True, stack_num=2)
        self.rv_net = range_view.RVNet(rv_base_block, rv_context_layer, rv_layers, use_att=True)
        self.rv_grid2point = get_module(rv_grid2point, in_dim=self.rv_net.out_channels)

        point_fusion_channels = (rv_feature_channel0, self.rv_net.out_channels)
        self.point_post = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels,
                                                                  out_channel=self.point_feat_out_channels,
                                                                  way=fusion_way)

        self.pred_layer = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)

    def stage_forward(self, point_feat, pcds_coord, pcds_sphere_coord):
        BS, C, N, _ = point_feat.shape
        pcds_sphere_coord_cur = pcds_sphere_coord.contiguous()

        # range-view
        point_feat = self.point_pre(point_feat)
        rv_input = VoxelMaxPool(pcds_feat=point_feat, pcds_ind=pcds_sphere_coord, output_size=self.rv_shape,
                                scale_rate=(1.0, 1.0))
        rv_feat = self.rv_net(rv_input)
        point_rv_feat = self.rv_grid2point(rv_feat, pcds_sphere_coord_cur)

        # merge multi-view
        point_feat_out = self.point_post(point_feat, point_rv_feat)

        # pred
        pred_cls = self.pred_layer(point_feat_out).float()
        return pred_cls

    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target):
        pcds_xyzi = pcds_xyzi[:, :5, :].contiguous()
        pred_cls = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        loss = self.criterion_seg_cate(pred_cls, pcds_target) + 2 * lovasz_softmax(pred_cls, pcds_target, ignore=0)

        return loss

    def infer(self, pcds_xyzi, pcds_coord, pcds_sphere_coord):
        pcds_xyzi = pcds_xyzi[:, :5, :].contiguous()
        pred_cls = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        return pred_cls
