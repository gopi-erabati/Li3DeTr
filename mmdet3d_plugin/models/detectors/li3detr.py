import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS, build_head
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models.builder import (build_voxel_encoder,
                                    build_middle_encoder, build_backbone,
                                    build_neck)
from mmdet3d.ops import Voxelization


@DETECTORS.register_module()
class Li3DeTr(Base3DDetector):
    """
    Li3DeTr as described in the IEEE/CVF WACV 2023 Paper
    """
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Li3DeTr, self).__init__(init_cfg)

        # POINTS FEATURES : Points Voxel Layer, Points Voxel Encoder,
        # Points Voxel Scatter, Pts backbone (SECOND), Pts neck (FPN)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
            self.pts_voxel_layer_cfg = pts_voxel_layer
        if pts_voxel_encoder:
            self.pts_voxel_encoder = build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)

        # build head
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @force_fp32(apply_to=('points'))
    def forward(self, points, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(points, **kwargs)
        else:
            return self.forward_test(points, **kwargs)

    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      gt_bboxes_3d_ignore=None):
        """
        Args:
            points (list[Tensor]): Points of each sample of shape (N, d)
            gt_bboxes_3d (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes of shape (num_box, 7)
            gt_labels_3d (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes labels
                of shape (num_box, )
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other detailssee
                :class `mmdet3d.datasets.pipelines.Collect`.
            gt_bboxes_3d_ignore (None | list[Tensor]): Specify which
                bounding boxes can be ignored when computing the loss.
        Returns:
            dict [str, Tensor]: A dictionary of loss components
        """
        # Extract Point Features
        point_feats = self.extract_feat(points, img_metas)
        # list[(B, 256, H, W), ...]
        losses = self.bbox_head.forward_train(point_feats,
                                              gt_bboxes_3d, gt_labels_3d,
                                              gt_bboxes_3d_ignore, img_metas)
        return losses

    def extract_feat(self, points, img_metas=None):
        """Extract Point Features
        Args:
            points (List[Tensor]): Points for each sample
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other detailssee
                :class `mmdet3d.datasets.pipelines.Collect`.

        Returns:
            List[Tensor]: list of point feats
                as BEV in different strides (x4, x8, x16, x32) of 1024 size of
                shape (B, C, H, W).
        """

        # Point Features
        point_feats = self.extract_point_features(points)  # list[Tensor]

        return point_feats
        # list[(B, 256, H, W), ...]

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """

        # Hard Voxelize
        if self.pts_voxel_layer_cfg['max_num_points'] != -1:
            voxels, coors, num_points = [], [], []
            for res in points:
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
            voxels = torch.cat(voxels, dim=0)  # (B*num_voxels, num_points, dim)
            num_points = torch.cat(num_points, dim=0)  # (B*num_voxels, )
            coors_batch = []
            for i, coor in enumerate(coors):  # (num_voxels, 3)
                coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)  # (n, 4)
                coors_batch.append(coor_pad)
            coors_batch = torch.cat(coors_batch, dim=0)  # (B*num_vox, 4)
            return voxels, num_points, coors_batch
        elif self.pts_voxel_layer_cfg['max_num_points'] == -1:
            # Dynamic Voxelization w/ center aware
            coors = []
            # dynamic voxelization only provide a coors mapping
            for res in points:
                res_coors = self.pts_voxel_layer(res)  # (N, 3)
                coors.append(res_coors)
            points = torch.cat(points, dim=0)  # (B*N, 4)
            coors_batch = []
            for i, coor in enumerate(coors):
                coor_pad = F.pad(coor, (1, 0), mode='constant',
                                 value=i)  # (N,1+3)
                coors_batch.append(coor_pad)
            coors_batch = torch.cat(coors_batch, dim=0)  # (B*N, 1+3)
            return points, coors_batch  # (B*N, 4), (B*N, 1+3)

    def extract_point_features(self, points):
        """ Extract features of Points using encoder, middle encoder,
        backbone and neck.
        Here points is list[Tensor] of batch """

        # Hard Voxelize
        if self.pts_voxel_layer_cfg['max_num_points'] != -1:
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            if self.pts_neck is not None:
                x = self.pts_neck(x)
            return x
            # [(B, 256, H256, W256), (B, 256, H128, W128), (B, 256, H64, W64),
            # (B, 256, H32, W32)]
        elif self.pts_voxel_layer_cfg['max_num_points'] == -1:
            # Dynamic Voxelization w/ center aware
            voxels, coors = self.voxelize(points)  # (B*N, 4), (B*N, 1+3)
            voxel_features, feature_coors = self.pts_voxel_encoder(voxels,
                                                                   coors)
            # (M, 128), (M, 1+3)
            batch_size = feature_coors[-1, 0].item() + 1
            x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
            x = self.pts_backbone(x)
            if self.pts_neck is not None:
                x = self.pts_neck(x)
            return x

    def forward_test(self,
                     points,
                     img_metas,
                     **kwargs):
        """
        Args:
            points (list[Tensor]): Points of each sample of shape (N, d)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        bbox_list = self.simple_test(points[0], img_metas[0], **kwargs)
        return bbox_list

    def simple_test(self, points, img_metas, rescale=False):
        """ Test function without test-time augmentation.

        Args:
            points (list[Tensor]): Points of each sample of shape (N, d)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.

        Returns:
            list[dict]: Predicted 3d boxes. Each list consists of a dict
            with keys: boxes_3d, scores_3d, labels_3d.
        """
        point_feats = self.extract_feat(points, img_metas)
        bbox_list = self.bbox_head.simple_test_bboxes(point_feats,
                                                      img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test(self, img, proj_img, proj_idxs, img_idxs, img_metas,
                 rescale=False):
        pass
