# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .transformer import build_glimpse_transformer
import copy
import torchvision


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, glimpse_transformer=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.use_rego = not ( (glimpse_transformer is None) )
        if self.use_rego:
            self.rego_scales = [3.0, 2.0, 1.0] 

            self.dropout = nn.Dropout(p=0.01)
            self.roi_query_dim = 256
            self.feat_gp = 4
            self.roi_feat_dim = self.roi_query_dim #* self.feat_gp

            self.ctx_ch = 64 
            ctx_inconvs = []
            ctx_outconvs = []
            ctx_gns = []
            for i in range(num_feature_levels):
                for gi in range(3):
                    ctx_inconvs.append(nn.Conv2d(hidden_dim, self.ctx_ch, kernel_size=3, stride=1, padding=(3+gi*4), dilation=(3+gi*4), groups=8))
                    ctx_outconvs.append(nn.Conv2d(self.ctx_ch, hidden_dim, kernel_size=1, stride=1, padding=0))
                ctx_gns.append(nn.GroupNorm(32, hidden_dim))

            self.ctx_inconvs = nn.ModuleList(ctx_inconvs)
            self.ctx_outconvs = nn.ModuleList(ctx_outconvs)
            self.ctx_gns = nn.ModuleList(ctx_gns)
            for mm in self.ctx_inconvs.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.xavier_normal_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)
            for mm in self.ctx_outconvs.modules():
                if isinstance(mm, nn.Conv2d):
                    nn.init.normal_(mm.weight, mean=0., std=1e-3)
                    nn.init.constant_(mm.bias, 0.0)

            self.roi_ext = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat2', 'feat3', 'feat4'], 7, 2) 
            num_pred = glimpse_transformer.decoder.num_layers 
            for gi in range(len(self.rego_scales)):
                rcnn_net = nn.Sequential( *[nn.Conv2d(hidden_dim, self.roi_feat_dim, kernel_size=7, stride=1, padding=0, groups=self.feat_gp),   #
                                            nn.Flatten(1), nn.LayerNorm(self.roi_feat_dim), nn.ReLU(),  
                                            nn.Linear(self.roi_feat_dim, self.roi_query_dim), nn.LayerNorm(self.roi_query_dim) ])
                setattr(self, 'rcnn_net_%d'%gi, rcnn_net)
                if gi == 0:
                    setattr(self, 'glimpse_transformer_%d'%gi, glimpse_transformer)
                else:
                    setattr(self, 'glimpse_transformer_%d'%gi, copy.deepcopy(glimpse_transformer))

                rego_hs_fuser = nn.Linear((gi+2) * hidden_dim, hidden_dim, bias=False)
                setattr(self, 'rego_hs_fuser_%d'%gi, _get_clones(rego_hs_fuser, num_pred))
                setattr(self, 'layer_norms_%d'%gi, nn.ModuleList([nn.LayerNorm(hidden_dim) for i in range(num_pred)]))

                rego_class_embed = nn.Linear(hidden_dim, num_classes) 
                rego_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                rego_class_embed.bias.data = torch.ones(num_classes) * bias_value
                nn.init.constant_(rego_bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(rego_bbox_embed.layers[-1].bias.data, 0)
                setattr(self, 'rego_class_embed_%d'%gi, nn.ModuleList([rego_class_embed for _ in range(num_pred)]))
                setattr(self, 'rego_bbox_embed_%d'%gi, nn.ModuleList([rego_bbox_embed for _ in range(num_pred)]))

                for m_str in ['rcnn_net', 'layer_norms']: 
                    m = getattr(self, m_str + '_%d'%gi)
                    for mm in m.modules():
                        if isinstance(mm, nn.Conv2d):
                            nn.init.xavier_normal_(mm.weight)
                            nn.init.constant_(mm.bias, 0.0)
                        elif isinstance(mm, nn.Linear):
                            nn.init.xavier_normal_(mm.weight)
                            nn.init.constant_(mm.bias, 0.0)
                        elif isinstance(mm, nn.LayerNorm):
                            nn.init.constant_(mm.weight, 1.0)
                            nn.init.constant_(mm.bias, 0.0)

                m = getattr(self, 'rego_hs_fuser_%d'%gi)
                for mm in m.modules():
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_normal_(mm.weight)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        if self.use_rego:
            im_shapes = samples.im_shapes
            imh,imw = samples.tensors.shape[-2:]
            batch_num = srcs[-1].size(0)

            feat_dict = {}
            for i in range(len(srcs)):
                feat_dict['feat%d'%i] = srcs[i]
    
            # context features
            lvl_feats = {}
            for i in range(len(srcs)):
                feat = feat_dict['feat%d'%i]
                for gi in range(3):
                    in_feat = feat.detach()
                    local_ctx_feat = self.ctx_inconvs[i*3+gi](F.relu(in_feat)) # N C' H W 
                    local_ctx_feat = F.relu(local_ctx_feat) 
                    ctx_feat = self.ctx_outconvs[i*3+gi](local_ctx_feat) # N C H W
                    feat = feat + 0.1 * self.dropout(ctx_feat ) 
                feat = self.ctx_gns[i](feat)
                lvl_feats.update({'feat%d'%(i+1): feat})

            # rcnn-based glimpse 
            with torch.no_grad():
                im_shape_tensor = torch.ones( (batch_num, 1, 4), device=srcs[-1].device)
                for bi in range(batch_num):
                    im_shape_tensor[bi, 0, 0::2] = im_shapes[bi][1]
                    im_shape_tensor[bi, 0, 1::2] = im_shapes[bi][0]

            hidden_dim = hs.size(3)
            prev_hs = hs[-1]
            prev_pred_hs = hs[-1]
            prev_coord = outputs_coord[-1].detach()
            for gi in range(len(self.rego_scales)):
                with torch.no_grad():
                    scalar_tensor = torch.ones( (batch_num, 1, 4), device=srcs[-1].device)
                    scalar_tensor[:,:,2:] = self.rego_scales[gi] 

                pred_bboxes = (prev_coord * scalar_tensor).clamp(max=1.0)
                pred_bboxes = pred_bboxes * im_shape_tensor
                pred_bboxes = box_ops.box_cxcywh_to_xyxy(pred_bboxes) # N x NP x 4
                pred_bboxes = [pred_bboxes[i] for i in range(batch_num)]

                ext_roi_feat = self.roi_ext(lvl_feats, pred_bboxes, [(imh, imw)])# (Nx3NP) x C x L x L 
                ext_roi_feat = getattr(self, 'rcnn_net_%d'%gi)(ext_roi_feat) 
                rego_in = ext_roi_feat.view(batch_num, -1, self.roi_query_dim) 

                rego_hs = getattr(self, 'glimpse_transformer_%d'%gi)(rego_in, prev_hs)[0] # NL(6) x N x Q x d 
                #rego_hs = getattr(self, 'glimpse_transformer_%d'%gi)(prev_hs, rego_in)[0] # NL(6) x N x Q x d 

                rego_output_classes = []
                rego_output_coords = []
                reference_reg = inverse_sigmoid(prev_coord)
                hs_fusers = getattr(self, 'rego_hs_fuser_%d'%gi)
                l_norms = getattr(self, 'layer_norms_%d'%gi)
                class_embeds = getattr(self, 'rego_class_embed_%d'%gi)
                bbox_embeds = getattr(self, 'rego_bbox_embed_%d'%gi)
                for lvl in range(rego_hs.shape[0]):
                    prev_h = prev_pred_hs.detach() 
                    fuse_h = torch.cat((prev_h, rego_hs[lvl]), 2) 
                    fuse_h = hs_fusers[lvl](fuse_h) 
                    fuse_h = l_norms[lvl](fuse_h)

                    output_class = class_embeds[lvl](fuse_h)
                    reference_reg = reference_reg + bbox_embeds[lvl](fuse_h) 
                    output_coord = reference_reg.sigmoid()

                    rego_output_classes.append(output_class)
                    rego_output_coords.append(output_coord)

                rego_output_classes = torch.stack(rego_output_classes)
                rego_output_coords = torch.stack(rego_output_coords)

                rego_outs = {'pred_logits_rego_%d'%gi: rego_output_classes[-1], 'pred_boxes_rego_%d'%gi: rego_output_coords[-1]} 
                out.update(rego_outs)
                if self.aux_loss:
                    box_aux = self._set_rego_aux_loss(rego_output_classes, rego_output_coords, prefix='_rego_%d'%gi)
                    out['aux_outputs'] += box_aux

                prev_hs = rego_hs[-1]
                prev_pred_hs = torch.cat( (prev_pred_hs, rego_hs[-1]), 2)
                prev_coord = rego_output_coords[-1].detach()
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @torch.jit.unused
    def _set_rego_aux_loss(self, outputs_class, outputs_coord, prefix=''):
        return [{'pred_logits'+prefix: a, 'pred_boxes'+prefix: b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, use_rego=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.use_rego = use_rego
        self.num_rego = 3 

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, extra_indices=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        if self.use_rego and extra_indices:
            for gi in range(self.num_rego):
                rego_src_logits = outputs['pred_logits_rego_%d'%gi]

                indices = extra_indices['rego_%d_ind'%gi]
                idx = self._get_src_permutation_idx(indices)
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                            dtype=torch.int64, device=src_logits.device)
                target_classes[idx] = target_classes_o

                target_classes_onehot = torch.zeros([rego_src_logits.shape[0], rego_src_logits.shape[1], rego_src_logits.shape[2] + 1],
                                                    dtype=rego_src_logits.dtype, layout=rego_src_logits.layout, device=rego_src_logits.device)
                target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

                target_classes_onehot = target_classes_onehot[:,:,:-1]
                loss_rego_ce = sigmoid_focal_loss(rego_src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
                losses['loss_ce_rego_%d'%gi] = loss_rego_ce

                if log:
                    losses['class_error_rego_%d'%gi] = 100 - accuracy(rego_src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, extra_indices=None):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, extra_indices=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        if self.use_rego and extra_indices:
            for gi in range(self.num_rego):
                indices = extra_indices['rego_%d_ind'%gi]
                idx = self._get_src_permutation_idx(indices)
                rego_src_boxes = outputs['pred_boxes_rego_%d'%gi][idx]
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

                loss_rego_bbox = F.l1_loss(rego_src_boxes, target_boxes, reduction='none')
                losses['loss_bbox_rego_%d'%gi] = loss_rego_bbox.sum() / num_boxes
                loss_rego_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(rego_src_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
                losses['loss_giou_rego_%d'%gi] = loss_rego_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, extra_indices=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        if self.use_rego:
            extra_indices = {}
            for gi in range(self.num_rego):
                extra_outs = {'pred_logits': outputs['pred_logits_rego_%d'%gi], 'pred_boxes':outputs['pred_boxes_rego_%d'%gi]}
                indices = self.matcher(extra_outs, targets) 
                extra_indices.update({'rego_%d_ind'%gi: indices})
        else:
            extra_indices = None

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, extra_indices=extra_indices, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_keys = list(aux_outputs.keys())
                if 'rego' in aux_keys[0]:
                    for gi in range(self.num_rego):
                        if 'pred_logits_rego_%d'%gi in aux_outputs.keys():
                            ext_name = '_rego_%d'%gi
                            ext_flag = True
                            break
                else:
                    ext_flag = False
                    ext_name = ''

                if ext_flag:
                    aux_outputs = {'pred_logits': aux_outputs['pred_logits' + ext_name], 
                                'pred_boxes': aux_outputs['pred_boxes' + ext_name]}
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if self.use_rego:
                        kwargs['extra_indices'] = None
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 120, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    glimpse_transformer = build_glimpse_transformer(args) if args.use_rego else None
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        glimpse_transformer=glimpse_transformer
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    orig_loss_scale = 0.8 if args.use_rego else 1.0
    weight_dict = {'loss_ce': args.cls_loss_coef * orig_loss_scale, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes'] #, 'cardinality']
    if args.masks:
        losses += ["masks"]

    if args.use_rego:
        for gi in range(3): 
            rego_weight_dict = {'loss_ce_rego_%d'%gi: args.cls_loss_coef, 'loss_bbox_rego_%d'%gi: args.bbox_loss_coef,  
                                'loss_giou_rego_%d'%gi: args.giou_loss_coef}
            weight_dict.update(rego_weight_dict)

            aux_weight_dict = {}
            for i in range(args.dec_layers - 1 + gi, (args.dec_layers - 1) + gi + 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in rego_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, use_rego=args.use_rego)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
