# ------------------------------------------------------------------------------------------------
# Modified from mmsegmentation (https://github.com/open-mmlab/mmsegmentation) 
# Apache-2.0 License
# Copyright (c) Open-MMLab.  
# and 
# openseg.pytorch (https://github.com/openseg-group/openseg.pytorch)
# MIT License
# ------------------------------------------------------------------------------------------------

import math
import numpy as np
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.ops import resize
from mmcv.runner import force_fp32
from mmseg.models.builder import build_loss, build_backbone
from mmseg.models.losses import accuracy

from .vision_transformer import deit_base_distilled_patch16_384
from .spatial_branch import spatial_branch
from .context_branch import context_branch
from .base import base_segmentor

import sys
sys.path.append("..")
from builder import SEGMENTORS
   
   
@SEGMENTORS.register_module()
class UN_EPT(base_segmentor):
    def __init__(self, 
                 heads, 
                 feat_dim, 
                 k, 
                 L,
                 dropout,
                 depth,
                 hidden_dim,
                 backbone_cfg,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 activation="relu",
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 auxiliary_head=None):

        """
        params:
        heads: head number of the transformer in the context branch;
        feat_dim: input feature dimension of the context branch;
        k: #points for each scale;
        L: #scale;
        depth: transformer encoder/decoder number in the context branch;
        hidden_dim: transforme hidden dimension in the context branch.

        """

        super(UN_EPT, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        if self.test_cfg.mode == 'stride':
            self.test_cfg.stride = test_cfg.stride
            self.test_cfg.crop_size = test_cfg.crop_size
        self.num_classes = self.test_cfg.num_classes
        self.ignore_index = ignore_index
        self.align_corners = False
        self.feat_dim = feat_dim

        self.loss_decode = build_loss(loss_decode)

        if pretrained is not None:
            logger = logging.getLogger()
            logger.info(f'load model from: {pretrained}')

        if backbone_cfg.type == 'DeiT':
            self.backbone = deit_base_distilled_patch16_384(
                            img_size=backbone_cfg.img_size,
                            patch_size=backbone_cfg.patch_size,
                            embed_dim=backbone_cfg.embed_dim,
                            depth=backbone_cfg.bb_depth,
                            num_heads=backbone_cfg.num_heads,
                            mlp_ratio=backbone_cfg.mlp_ratio,
                            pretrained=pretrained)
        elif backbone_cfg.type == 'ResNetV1c':
            self.backbone = build_backbone(backbone_cfg)
            self.backbone.init_weights(pretrained=pretrained)
        
       
        self.cls = nn.Conv2d(feat_dim, self.num_classes, kernel_size=1)

        # get pyramid features
        self.layers = nn.ModuleList([])
        self.backbone_type = backbone_cfg.type
        if self.backbone_type == 'DeiT':
            self.layers.append(nn.Conv2d(backbone_cfg.embed_dim, feat_dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(backbone_cfg.embed_dim, feat_dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(backbone_cfg.embed_dim, feat_dim, kernel_size=1, stride=1))
        elif self.backbone_type == 'ResNetV1c':
            self.layers.append(nn.Conv2d(512, feat_dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(1024, feat_dim, kernel_size=1, stride=1))
            self.layers.append(nn.Conv2d(2048, feat_dim, kernel_size=1, stride=1))

        self.context_branch = context_branch(d_model=feat_dim, nhead=heads,
                    num_encoder_layers=depth, num_decoder_layers=depth, dim_feedforward=hidden_dim, dropout=dropout,
                    activation=activation, num_feature_levels=L, dec_n_points=k,  enc_n_points=k)
        
        self.num_queries = self.test_cfg.num_queries
        self.query_embed = nn.Embedding(self.num_queries, feat_dim)
        self.spatial_branch = spatial_branch()

        self.dir_head = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.SyncBatchNorm(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 8, kernel_size=1, stride=1, padding=0, bias=False))

        self.mask_head = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.SyncBatchNorm(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=False))


    def encode_decode(self, x):

        bsize, c, h, w = x.shape
        backbone_feats = self.backbone(x)
        if self.backbone_type == 'ResNetV1c':
            backbone_feats = backbone_feats[1:]
       
        context = self.spatial_branch(x)
        
        mask_map = self.mask_head(context)
        dir_map = self.dir_head(context)

        context = context.flatten(2).permute(2, 0, 1)

        pyramid_feats = [] 
        for i, conv_layer in enumerate(self.layers):
            feature = conv_layer(backbone_feats[i]) 
            pyramid_feats.append(feature)
    
        out = self.context_branch(pyramid_feats, context, self.query_embed.weight) 
        
        out = out.unsqueeze(0).reshape([h//8, w//8, bsize, self.feat_dim]).permute(2, 3, 0, 1)

        out = self.cls(out)

        seg_logits = resize(
            input=out,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return seg_logits, mask_map, dir_map

    def distance_to_mask_label(self, distance_map, seg_label_map, return_tensor=False):

        if return_tensor:
            assert isinstance(distance_map, torch.Tensor)
            assert isinstance(seg_label_map, torch.Tensor)
        else:
            assert isinstance(distance_map, np.ndarray)
            assert isinstance(seg_label_map, np.ndarray)

        if return_tensor:
            mask_label_map = torch.zeros_like(seg_label_map).long().to(distance_map.device)
        else:
            mask_label_map = np.zeros(seg_label_map.shape, dtype=np.int)

        keep_mask = (distance_map <= 5) & (distance_map >= 0)
        mask_label_map[keep_mask] = 1
        mask_label_map[seg_label_map == -1] = -1

        return mask_label_map

    def calc_weights(self, label_map, num_classes):

        weights = []
        for i in range(num_classes):
            weights.append((label_map == i).sum().data)
        weights = torch.FloatTensor(weights)
        weights_sum = weights.sum()
        return (1 - weights / weights_sum).cuda()  

    def angle_to_direction_label(self, angle_map, seg_label_map=None, distance_map=None, num_classes=8, extra_ignore_mask=None, return_tensor=False):

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
            assert isinstance(seg_label_map, torch.Tensor) or seg_label_map is None
        else:
            assert isinstance(angle_map, np.ndarray)
            assert isinstance(seg_label_map, np.ndarray) or seg_label_map is None

        _, label_map = self.align_angle(angle_map, num_classes=num_classes, return_tensor=return_tensor)
        if distance_map is not None:
            label_map[distance_map > 5] = num_classes
        if seg_label_map is None:
            if return_tensor:
                ignore_mask = torch.zeros(angle_map.shape, dtype=torch.uint8).to(angle_map.device)
            else:
                ignore_mask = np.zeros(angle_map.shape, dtype=np.bool)
        else:
            ignore_mask = seg_label_map == -1

        if extra_ignore_mask is not None:
            extra_ignore_mask = extra_ignore_mask.unsqueeze(1)
            ignore_mask = ignore_mask | extra_ignore_mask
        label_map[ignore_mask] = -1

        return label_map

    def align_angle(self, angle_map, 
                    num_classes=8, 
                    return_tensor=False):

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
        else:
            assert isinstance(angle_map, np.ndarray)

        step = 360 / num_classes
        if return_tensor:
            new_angle_map = torch.zeros(angle_map.shape).float().to(angle_map.device)
            angle_index_map = torch.zeros(angle_map.shape).long().to(angle_map.device)
        else:
            new_angle_map = np.zeros(angle_map.shape, dtype=np.float)
            angle_index_map = np.zeros(angle_map.shape, dtype=np.int)
        mask = (angle_map <= (-180 + step/2)) | (angle_map > (180 - step/2))
        new_angle_map[mask] = -180
        angle_index_map[mask] = 0

        for i in range(1, num_classes):
            middle = -180 + step * i
            mask = (angle_map > (middle - step / 2)) & (angle_map <= (middle + step / 2))
            new_angle_map[mask] = middle
            angle_index_map[mask] = i

        return new_angle_map, angle_index_map

    def shift(self, x, offset):
        """
        x: b x c x h x w
        offset: b x 2 x h x w
        """
        def gen_coord_map(H, W):
            coord_vecs = [torch.arange(length, dtype=torch.float) for length in (H, W)]
            coord_h, coord_w = torch.meshgrid(coord_vecs)
            coord_h = coord_h.cuda()
            coord_w = coord_w.cuda()
            return coord_h, coord_w
        
        b, c, h, w = x.shape
        
        coord_map = gen_coord_map(h, w)
        norm_factor = torch.FloatTensor([(w-1)/2, (h-1)/2]).cuda()
        grid_h = offset[:, 0]+coord_map[0]
        grid_w = offset[:, 1]+coord_map[1]
        grid = torch.stack([grid_w, grid_h], dim=-1) / norm_factor - 1

        x = F.grid_sample(x.float(), grid, padding_mode='border', mode='bilinear', align_corners=True)
       
        return x

    def _get_offset(self, mask_logits, dir_logits):
        
        edge_mask = mask_logits[:, 1] > 0.5
        dir_logits = torch.softmax(dir_logits, dim=1)
        n, _, h, w = dir_logits.shape

        keep_mask = edge_mask

        dir_label = torch.argmax(dir_logits, dim=1).float()
        offset = self.label_to_vector(dir_label)
        offset = offset.permute(0, 2, 3, 1)
        offset[~keep_mask, :] = 0
        
        return offset
    
    def label_to_vector(self, labelmap, 
                        num_classes=8):

        assert isinstance(labelmap, torch.Tensor)

        label_to_vector_mapping = {    
            8: [
                [0, -1], [-1, -1], [-1, 0], [-1, 1],
                [0, 1], [1, 1], [1, 0], [1, -1]
            ],
            16: [
                [0, -2], [-1, -2], [-2, -2], [-2, -1], 
                [-2, 0], [-2, 1], [-2, 2], [-1, 2],
                [0, 2], [1, 2], [2, 2], [2, 1],
                [2, 0], [2, -1], [2, -2], [1, -2]
            ]
        }

        mapping = label_to_vector_mapping[num_classes]
        offset_h = torch.zeros_like(labelmap).long()
        offset_w = torch.zeros_like(labelmap).long()

        for idx, (hdir, wdir) in enumerate(mapping):
            mask = labelmap == idx
            offset_h[mask] = hdir
            offset_w[mask] = wdir

        return torch.stack([offset_h, offset_w], dim=-1).permute(0, 3, 1, 2).to(labelmap.device)

    def forward_train(self, img, img_metas, gt_semantic_seg, distance_map, angle_map):
        
        seg_logits, pred_mask, pred_direction = self.encode_decode(img)
        losses = dict()

        loss_decode = self.losses(seg_logits, pred_mask, pred_direction, gt_semantic_seg, distance_map, angle_map)
        losses.update(loss_decode)

        return losses
    
    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, pred_mask, pred_direction, seg_label, distance_map, angle_map):
        """Compute segmentation loss."""
        loss = dict()
        
        seg_weight = None

        gt_mask = self.distance_to_mask_label(distance_map, seg_label, return_tensor=True)
        gt_size = gt_mask.shape[2:]
        mask_weights = self.calc_weights(gt_mask, 2)

        pred_direction = F.interpolate(pred_direction, size=gt_size, mode="bilinear", align_corners=True)
        pred_mask = F.interpolate(pred_mask, size=gt_size, mode="bilinear", align_corners=True)
        mask_loss = F.cross_entropy(pred_mask, gt_mask[:,0], weight=mask_weights, ignore_index=-1)
        
        mask_threshold = 0.5
        binary_pred_mask = torch.softmax(pred_mask, dim=1)[:, 1, :, :] > mask_threshold
        
        gt_direction = self.angle_to_direction_label(
            angle_map,
            seg_label_map=seg_label,
            extra_ignore_mask=(binary_pred_mask == 0),
            return_tensor=True
        )

        direction_loss_mask = gt_direction != -1
        direction_weights = self.calc_weights(gt_direction[direction_loss_mask], pred_direction.size(1))
        direction_loss = F.cross_entropy(pred_direction, gt_direction[:,0], weight=direction_weights, ignore_index=-1)

        offset = self._get_offset(pred_mask, pred_direction)
        refine_map = self.shift(seg_logit, offset.permute(0,3,1,2))

        seg_label = seg_label.squeeze(1)

        loss['loss_seg'] = 0.8*self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index) + 5*mask_loss + 0.6*direction_loss + \
            self.loss_decode(
            refine_map,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        return loss

