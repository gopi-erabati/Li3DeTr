import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.cnn import Linear
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from .li3detr_head import Li3DeTrHead
from ...core.bbox.util import normalize_bbox


@HEADS.register_module()
class Li3DeTrKittiHead(Li3DeTrHead):
    """
        This is the head for Li3DeTr model (KITTI Dataset)
    """

    def __int__(self,
                num_classes,
                sync_cls_avg_factor,
                num_query,
                with_box_refine=True,
                as_two_stage=False,
                transformer=None,
                positional_embedding='fourier',
                bbox_coder=None,
                num_cls_fcs=2,
                num_reg_fcs=2,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                loss_bbox=dict(type='L1Loss', loss_weight=0.25),
                train_cfg=dict(
                    assigner=dict(
                        type='HungarianAssignerMsfDetr3D',
                        cls_cost=dict(type='FocalLossCost', weight=2.0),
                        reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                    )),
                code_weights=None,
                init_cfg=None,
                pretrained=None,
                **kwargs
                ):
        super(Li3DeTrKittiHead, self).__int__(num_classes,
                                              sync_cls_avg_factor,
                                              num_query,
                                              with_box_refine=with_box_refine,
                                              as_two_stage=as_two_stage,
                                              transformer=transformer,
                                              positional_embedding=positional_embedding,
                                              bbox_coder=bbox_coder,
                                              num_cls_fcs=num_cls_fcs,
                                              num_reg_fcs=num_reg_fcs,
                                              loss_cls=loss_cls,
                                              loss_bbox=loss_bbox,
                                              train_cfg=train_cfg,
                                              code_weights=code_weights,
                                              init_cfg=init_cfg,
                                              pretrained=pretrained,
                                              **kwargs)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        center_branch = []
        for _ in range(self.num_reg_fcs):
            center_branch.append(Linear(self.embed_dims, self.embed_dims))
            center_branch.append(nn.ReLU())
        center_branch.append(Linear(self.embed_dims, 3))
        center_branch = nn.Sequential(*center_branch)

        size_branch = []
        for _ in range(self.num_reg_fcs):
            size_branch.append(Linear(self.embed_dims, self.embed_dims))
            size_branch.append(nn.ReLU())
        size_branch.append(Linear(self.embed_dims, 3))
        size_branch = nn.Sequential(*size_branch)

        angle_branch = []
        for _ in range(self.num_reg_fcs):
            angle_branch.append(Linear(self.embed_dims, self.embed_dims))
            angle_branch.append(nn.ReLU())
        angle_branch.append(Linear(self.embed_dims, 2))
        angle_branch = nn.Sequential(*angle_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.center_branches = _get_clones(center_branch, num_pred)
            self.size_branches = _get_clones(size_branch, num_pred)
            self.angle_branches = _get_clones(angle_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.center_branches = nn.ModuleList(
                [center_branch for _ in range(num_pred)])
            self.size_branches = nn.ModuleList(
                [size_branch for _ in range(num_pred)])
            self.angle_branches = nn.ModuleList(
                [angle_branch for _ in range(num_pred)])

    def init_weights(self):
        super(BaseDenseHead, self).init_weights()

    def forward(self, point_feats, img_metas):
        """
        Forward Function
        Args:
            point_feats (list[Tensor]): list of point feats of shape
                (B, C, H, W)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other detailssee
                :class `mmdet3d.datasets.pipelines.Collect`.
        Returns:
            A tuple containing two tensors:
                sem_cls_logits (n_lay, B, n_q, #cls)
                bbox_pred (n_lay, B, n_q, 10)
        """

        # This is for Encoder
        batch_size = point_feats[0].size(0)
        point_bev_h, point_bev_w = self.bev_shape
        point_masks = point_feats[0].new_zeros((batch_size, point_bev_h,
                                                point_bev_w))

        mlvl_masks_point = []
        mlvl_pos_enc_point = []
        for point_feat in point_feats:
            # point
            mlvl_masks_point.append(
                F.interpolate(point_masks[None], size=point_feat.shape[
                                                      -2:]).to(
                    torch.bool).squeeze(0)
            )
            mlvl_pos_enc_point.append(self.positional_encoding(
                mlvl_masks_point[-1]))

        query = self.query_embedding.weight  # (n_q, d)

        # This is to include Encoder
        hs, init_reference, inter_reference = self.transformer(
                                                               point_feats,
                                                               mlvl_masks_point,
                                                               mlvl_pos_enc_point,
                                                               query,
                                                               reg_branches=self.center_branches if self.with_box_refine else None,
                                                               img_metas=img_metas)

        # (#layers, n_q, B, d)
        # (B, n_q, 3)
        # (#layers, B, n_q, 3)
        hs = hs.permute(0, 2, 1, 3)  # (#layers, B, n_q, d)

        # outputs for each decoder layer
        all_sem_cls_logits = []
        all_bbox_pred = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_reference[lvl - 1]

            reference = inverse_sigmoid(reference)

            output_class = self.cls_branches[lvl](hs[lvl])  # (B, n_q, #cls)
            bbox_center = self.center_branches[lvl](hs[lvl])  # (B, n_q, 3)
            bbox_size = self.size_branches[lvl](hs[lvl])  # (B, n_q, 3)
            bbox_angle = self.angle_branches[lvl](hs[lvl])  # (B, n_q, 2)

            # add reference to center
            bbox_center += reference
            bbox_center = bbox_center.sigmoid()

            pc_range_ = bbox_center.new_tensor([[self.pc_range[3] -
                                                 self.pc_range[0],
                                                 self.pc_range[4] -
                                                 self.pc_range[1],
                                                 self.pc_range[5] -
                                                 self.pc_range[2]]])
            pc_start_ = bbox_center.new_tensor(
                [[self.pc_range[0], self.pc_range[1],
                  self.pc_range[2]]])
            bbox_center = (bbox_center * pc_range_) + pc_start_

            tmp = torch.cat([bbox_center[..., 0:2], bbox_size[..., 0:2],
                             bbox_center[..., 2:3], bbox_size[..., 2:3],
                             bbox_angle], dim=-1)  # (B, n_q, 8)

            bbox_pred = tmp
            all_sem_cls_logits.append(output_class)
            all_bbox_pred.append(bbox_pred)

        # stack all for all layers
        all_sem_cls_logits = torch.stack(all_sem_cls_logits)
        # (#lay, B, n_q, #cls)
        all_bbox_pred = torch.stack(all_bbox_pred)
        # (#lay, B, n_q, 8)

        return (all_sem_cls_logits, all_bbox_pred)
        # (n_lay, B, n_q, #cls)
        # (n_lay, B, n_q, 8)

    def loss_single(self,
                    sem_cls_logits,
                    bbox_pred,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """Loss function for each decoder layer
            Args:
                sem_cls_logits (Tensor): class logits (B, n_q, #cls)
                bbox_pred (Tensor): Bboxes  (B, n_q, 10)
                gt_bboxes_list (list[Tensor]): Ground truth bboxes for each
                    image with shape (num_gts, 7) in [cx, cy, cz, l, w, h,
                    theta] format. LiDARInstance3DBoxes
                gt_labels_list (list[Tensor]): Ground truth class indices
                    for each image with shape (num_gts, ).
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                (tuple): loss_cls, loss_bbox
        """

        num_imgs = sem_cls_logits.size(0)

        # prepare scores and bboxes list for all images to get targets
        sem_cls_logits_list = [sem_cls_logits[i] for i in range(num_imgs)]
        # [(n_q, #cls+1), ... #images]
        bbox_pred_list = [bbox_pred[i] for i in range(num_imgs)]
        # [(n_q, 8), ... #images]

        cls_reg_targets = self.get_targets(sem_cls_logits_list,
                                           bbox_pred_list,
                                           gt_bboxes_list,
                                           gt_labels_list,
                                           gt_bboxes_ignore_list)

        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)  # (bs*num_q, )
        label_weights = torch.cat(label_weights_list, 0)  # (bs*num_q, )
        bbox_targets = torch.cat(bbox_targets_list, 0)  # (bs * num_q, 8)
        bbox_weights = torch.cat(bbox_weights_list, 0)  # (bs * num_q, 8)

        # classification loss
        pred_logits = sem_cls_logits.reshape(-1, self.cls_out_channels)
        # (bs * num_q, #cls)
        # construct weighted avg_factor
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                pred_logits.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            pred_logits, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_pred.reshape(-1, bbox_pred.size(-1))
        # (bs * num_q, 8)
        normalized_bbox_targets = normalize_bbox(bbox_targets, None)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :8], normalized_bbox_targets[isnotnan, :8],
            bbox_weights[isnotnan, :8], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    def _get_target_single(self,
                           sem_cls_logits,
                           bbox_pred,
                           gt_bboxes, gt_labels,
                           gt_bboxes_ignore=None):
        """Compute regression and classification targets for one
            image.
        Args:
            sem_cls_logits (Tensor): Box score logits for each
                batch element (n_q, #cls)
            bbox_pred (Tensor): Bbox predictions for each
                batch element (n_q, 10)
            gt_bboxes (Tensor): Ground truth bboxes for one batch with
                shape (num_gts, 7) in [cx, cy, cz, l, w, h, theta] format.
            gt_labels (Tensor): Ground truth class indices for one batch
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - dir_targets (Tensor): Direction targets for each image.
                - dir_weights (Tensor): Direction weights for each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)

        assign_result = self.assigner.assign(bbox_pred,
                                             sem_cls_logits.reshape(-1,
                                                                    self.cls_out_channels),
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds  # (num_pos, ) number of
        # predicted bounding boxes with matched ground truth box, the index
        # of those matched predicted bounding boxes
        neg_inds = sampling_result.neg_inds  # (num_neg, ) indices of
        # negative predicted bounding boxes

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),  # (num_q, ) filled w/ cls
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # here only the matched positive boxes are assigned with labels of
        # matched boxes from ground truth
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # Here, we assign predicted boxes to gt boxes and therefore we want
        # labels and bbox_targets both in predicted box shape but with gt
        # labels and boxes in it!!!
        bbox_targets = torch.zeros_like(bbox_pred)[..., :7]
        # because bbox_pred is 8 values but targets is 7 values
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights,
                bbox_targets, bbox_weights,
                pos_inds, neg_inds)
        # labels (num_q, )
        # label_weights (num_q, )
        # bbox_targets (num_q, 7)
        # bbox_weights (num_q, 7)
        # pos_inds (num_q, )
        # neg_inds (num_q, )

    @force_fp32(apply_to=('all_sem_cls_logits', 'all_bbox_pred',))
    def get_bboxes(self,
                   all_sem_cls_logits,
                   all_bbox_pred,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_sem_cls_logits (Tensor): class logits from each layer of
                decoder Shape (n_lay, B, n_q, #cls)
            all_bbox_pred (Tensor): Box predictions from each layer of decoder
                of shape (n_lay, B, n_q, 10)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is an (n, 7) tensor, where the
                7 columns are bounding box positions
                (cx, cy, cz, l, w, h, theta). The second item is a (n,
                ) tensor where each item is predicted score between 0 and 1.
                The third item is a (n,) tensor where each item is the
                predicted class label of the corresponding box.
        """

        sem_cls_logits = all_sem_cls_logits[-1]
        bbox_pred = all_bbox_pred[-1]

        preds_dicts = self.bbox_coder.decode(sem_cls_logits, bbox_pred)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, 7)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list