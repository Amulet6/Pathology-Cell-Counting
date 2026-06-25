# reference to song2021rethinking.
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
import numpy as np
import time
from util.misc import (accuracy, get_world_size, interpolate, is_dist_avail_and_initialized)
from typing import Optional, List

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

# the defenition of the Crowd Counting model
class Model_builder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = self.cfg.MODEL.DECODER_kwargs["num_classes"] # default:2 (person/background)
        self.num_anchor_points = cfg.MODEL.ROW * cfg.MODEL.LINE  # default:4
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder() 
        self.dense_aux_en = bool(getattr(self.cfg.MODEL, 'DENSE_AUX_EN', False))
        self.dense_aux_level = int(getattr(self.cfg.MODEL, 'DENSE_AUX_LEVEL', 3))
        self.dense_sigma = float(getattr(self.cfg.MODEL, 'DENSE_SIGMA', 2.0))
        if self.dense_aux_en:
            self.dense_head = nn.Sequential(
                nn.Conv2d(self.encoder.get_outplanes()[self.dense_aux_level - 1], 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )

    def _build_encoder(self, ):
        #########################################################################################
        # input: image, output: [feat1(H/2,W/2), feat2(H/4,W/4), feat3(H/8,W/8), feat4(H/16,W/16)]
        #########################################################################################
        if self.cfg.MODEL.ENCODER in ['vgg16', 'vgg16_bn']:
            from .Encoder import Base_VGG as build_encoder
        elif self.cfg.MODEL.ENCODER in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            from .Encoder import Base_ResNet as build_encoder
        self.cfg.MODEL.ENCODER_kwargs['name'] = self.cfg.MODEL.ENCODER
        # Direction D: DCNv2
        self.cfg.MODEL.ENCODER_kwargs['dcnv2_enabled'] = self.cfg.MODEL.get('DCNV2_ENABLED', False)
        self.cfg.MODEL.ENCODER_kwargs['dcnv2_layers'] = self.cfg.MODEL.get('DCNV2_LAYERS', 0)
        encoder = build_encoder(**self.cfg.MODEL.ENCODER_kwargs)
        return encoder
    
    def _build_decoder(self, ): 
        if self.cfg.MODEL.DECODER == 'basic':
            from .Decoder import Basic_Decoder_Model as build_decoder 
        elif self.cfg.MODEL.DECODER == 'IFI':
            from .Decoder import IFI_Decoder_Model as build_decoder        
        self.cfg.MODEL.DECODER_kwargs['in_planes'] = self.encoder.get_outplanes()
        self.cfg.MODEL.DECODER_kwargs['line'] = self.cfg.MODEL.LINE
        self.cfg.MODEL.DECODER_kwargs['row'] = self.cfg.MODEL.ROW
        self.cfg.MODEL.DECODER_kwargs['num_anchor_points'] = self.num_anchor_points
        self.cfg.MODEL.DECODER_kwargs['sync_bn'] = False
        self.cfg.MODEL.DECODER_kwargs['AUX_EN'] = self.cfg.MODEL.AUX_EN
        self.cfg.MODEL.DECODER_kwargs['AUX_NUMBER'] = self.cfg.MODEL.AUX_NUMBER
        self.cfg.MODEL.DECODER_kwargs['AUX_RANGE'] = self.cfg.MODEL.AUX_RANGE
        self.cfg.MODEL.DECODER_kwargs['AUX_kwargs'] = self.cfg.MODEL.AUX_kwargs
        decoder = build_decoder(**self.cfg.MODEL.DECODER_kwargs)
        return decoder

    def forward(self, samples: NestedTensor):
        features = self.encoder(samples)
        out = self.decoder(samples, features)       
        if self.dense_aux_en:
            dense_feat = features[self.dense_aux_level - 1]
            out['dense_map'] = self.dense_head(dense_feat)
            out['dense_level'] = self.dense_aux_level
        sample_tensor = samples.tensors if hasattr(samples, 'tensors') else samples
        out['img_shape'] = sample_tensor.shape[-2:]
        return out   # {'pred_logits', 'pred_points', 'offset'}

class SetCriterion_Crowd(nn.Module):
    # Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved. 
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, aux_kwargs,
                 edge_band=0, edge_weight=0.0, dense_aux_en=False, dense_sigma=2.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            edge_band: edge ignore band width (px), 0 = disabled
            edge_weight: loss weight for anchors inside the edge band (0~1)
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.edge_band = edge_band
        self.edge_weight = edge_weight
        self.dense_aux_en = dense_aux_en
        self.dense_sigma = dense_sigma
        self.current_edge_mask = None  # cached in forward(), consumed by loss_*
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        if 'loss_aux' in self.weight_dict:
            self.aux_mode = False
        else:
            self.aux_mode = True
            self.aux_number = aux_kwargs['AUX_NUMBER']
            self.aux_range = aux_kwargs['AUX_RANGE']
            self.aux_kwargs = aux_kwargs['AUX_kwargs']

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # size=[batch*patch, num_queries, 2] [p0, p1]

        # make the gt array, only match idx set 1
        idx = self._get_src_permutation_idx(indices) # batch_idx, src_idx
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # always 1 # size=[num_of_gt_point]
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o  # size=[batch*patch, num_queries] 0/1
        # Per-anchor CE (reduction='none') to support edge masking
        loss_per_anchor = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes,
            weight=self.empty_weight, reduction='none')  # [B, N]
        if self.edge_band > 0 and self.current_edge_mask is not None:
            loss_per_anchor = loss_per_anchor * self.current_edge_mask
        loss_ce = loss_per_anchor.sum() / num_points
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points):
        '''
        only compare to matched pairs
        '''
        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')
        loss_per_point = loss_bbox.sum(dim=-1)  # sum over x,y → [M]
        if self.edge_band > 0 and self.current_edge_mask is not None:
            batch_idx, src_idx = idx  # reuse Hungarian matching indices
            loss_per_point = loss_per_point * self.current_edge_mask[batch_idx, src_idx]
        losses = {}
        losses['loss_points'] = loss_per_point.sum() / num_points
        return losses

    def loss_auxiliary(self, outputs, targets, show):
        # out: {"pred_logits", "pred_points", "offset"}
        # aux_out: {"pos0":out, "pos1":out, "neg0":out, "neg1":out, ...}
        loss_aux_pos = 0.
        loss_aux_neg = 0.
        loss_aux = 0.
        for n_pos in range(self.aux_number[0]):
            src_outputs = outputs['pos%d'%n_pos]
            # cls loss
            pred_logits = src_outputs['pred_logits'] # size=[1, # of gt anchors, 2] [p0, p1]
            target_classes = torch.ones(pred_logits.shape[:2], dtype=torch.int64, device=pred_logits.device) # [1, # of gt anchors], all sample is the head class
            loss_ce_pos = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
            # loc loss
            pred_points = src_outputs['pred_points'][0]
            target_points = torch.cat([t['point'] for t in targets], dim=0)
            target_points = target_points.repeat(1, int(pred_points.shape[0]/target_points.shape[0]))
            target_points = target_points.reshape(-1, 2)
            loss_loc_pos = F.mse_loss(pred_points, target_points, reduction='none')
            loss_loc_pos = loss_loc_pos.sum() / pred_points.shape[0]
            loss_aux_pos += loss_ce_pos + self.aux_kwargs['pos_loc'] * loss_loc_pos
        loss_aux_pos /= (self.aux_number[0] + 1e-9)

        for n_neg in range(self.aux_number[1]):
            src_outputs = outputs['neg%d'%n_neg]
            # cls loss
            pred_logits = src_outputs['pred_logits'] # size=[1, # of gt anchors, 2] [p0, p1]
            target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.int64, device=pred_logits.device) # [1, # of gt anchors], all sample is the head class
            loss_ce_neg = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
            # loc loss
            pred_points = src_outputs['offset'][0]
            target_points = torch.zeros(pred_points.shape, dtype=torch.float, device=pred_logits.device)
            loss_loc_neg = F.mse_loss(pred_points, target_points, reduction='none')
            loss_loc_neg = loss_loc_neg.sum() / pred_points.shape[0]
            loss_aux_neg += loss_ce_neg + self.aux_kwargs['neg_loc'] * loss_loc_neg
        loss_aux_neg /= (self.aux_number[1] + 1e-9)
        
        if show:
            if self.aux_number[0] > 0:
                print("Auxiliary Training: [Pos] loss_cls:", loss_ce_pos, " loss_loc:", loss_loc_pos, " loss:", loss_aux_pos)
            if self.aux_number[1] > 0:
                print("Auxiliary Training: [Neg] loss_cls:", loss_ce_neg, " loss_loc:", loss_loc_neg, " loss:", loss_aux_neg)
        loss_aux = self.aux_kwargs['pos_coef']*loss_aux_pos + self.aux_kwargs['neg_coef']*loss_aux_neg
        losses = {'loss_aux': loss_aux}
        return losses

    def loss_dense(self, outputs, targets):
        if not self.dense_aux_en or 'dense_map' not in outputs or 'img_shape' not in outputs:
            return {'loss_dense': outputs['pred_logits'].sum() * 0.0}

        pred = outputs['dense_map']
        img_h, img_w = outputs['img_shape']
        losses = []
        for i, tgt in enumerate(targets):
            gt_points = tgt['point']
            h, w = pred.shape[-2:]
            target_map = torch.zeros((1, 1, h, w), device=pred.device)
            if gt_points.numel() > 0:
                yy = torch.arange(h, device=pred.device, dtype=torch.float32).view(h, 1)
                xx = torch.arange(w, device=pred.device, dtype=torch.float32).view(1, w)
                sigma = max(self.dense_sigma, 1e-6)
                for pt in gt_points:
                    px = pt[0] / max(float(img_w - 1), 1.0) * (w - 1)
                    py = pt[1] / max(float(img_h - 1), 1.0) * (h - 1)
                    gauss = torch.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2.0 * sigma * sigma))
                    target_map[0, 0] = torch.maximum(target_map[0, 0], gauss)
            losses.append(F.mse_loss(torch.sigmoid(pred[i:i+1]), target_map))
        return {'loss_dense': torch.stack(losses).mean()}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _compute_edge_mask(self, outputs, crop_size):
        """Compute per-anchor loss weights based on distance to crop boundary.

        Anchors whose grid-center falls within edge_band of any crop border
        receive a reduced weight (edge_weight … 1.0 linear ramp).

        Returns:
            mask: [B, N] float tensor of per-anchor weights.
        """
        # Anchor grid centers in image coordinates = pred_points - offset
        anchor_xy = outputs['pred_points'] - outputs['offset']  # [B, N, 2]
        x, y = anchor_xy[..., 0], anchor_xy[..., 1]
        crop_h, crop_w = int(crop_size[0]), int(crop_size[1])  # force Python int

        dist_left = x
        dist_right = crop_w - x
        dist_top = y
        dist_bottom = crop_h - y
        min_dist = torch.min(
            torch.stack([dist_left, dist_right, dist_top, dist_bottom], dim=-1),
            dim=-1)[0]
        min_dist = torch.clamp(min_dist, min=0.0)  # guard against out-of-bounds coords

        # Linear ramp: edge_weight at boundary → 1.0 at edge_band
        band = float(self.edge_band)
        mask = torch.where(
            min_dist < band,
            self.edge_weight + (1.0 - self.edge_weight) * (min_dist / band),
            torch.ones_like(min_dist)
        )
        return mask  # [B, N]

    def forward(self, outputs, targets, show=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points'], 'offset': outputs['offset']}
        if 'dense_map' in outputs:
            output1['dense_map'] = outputs['dense_map']
        if 'img_shape' in outputs:
            output1['img_shape'] = outputs['img_shape']
        indices1 = self.matcher(output1, targets) # return (idx_of_pred, idx_of_gt). # pairs of indices # indices[batch] = (point_coords, gt_idx)

        # Compute per-anchor edge mask (cached for loss_labels / loss_points)
        if self.edge_band > 0:
            crop_size = targets[0].get('crop_size', (256, 256))
            self.current_edge_mask = self._compute_edge_mask(output1, crop_size)
        else:
            self.current_edge_mask = None

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.weight_dict.keys():
            if loss == 'loss_ce':
                losses.update(self.loss_labels(output1, targets, indices1, num_boxes))
            elif loss == 'loss_points':
                losses.update(self.loss_points(output1, targets, indices1, num_boxes))
            elif loss == 'loss_aux':
                out_auxs = output1['aux']
                losses.update(self.loss_auxiliary(out_auxs, targets, show))
            elif loss == 'loss_dense':
                losses.update(self.loss_dense(output1, targets))
            else:
                raise KeyError('do you really want to compute {} loss?'.format(loss))
        print(losses)
        return losses
