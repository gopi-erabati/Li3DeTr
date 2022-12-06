from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from .li3detr import Li3DeTr


@DETECTORS.register_module()
class Li3DeTrKitti(Li3DeTr):
    """
        Li3DeTr as described in the IEEE/CVF WACV 2023 Paper for KITTI dataset
    """
    def __int__(self,
                pts_voxel_layer=None,
                pts_voxel_encoder=None,
                pts_middle_encoder=None,
                pts_backbone=None,
                pts_neck=None,
                bbox_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None,
                init_cfg=None
                ):
        super(Li3DeTrKitti, self).__int__(pts_voxel_layer=pts_voxel_layer,
                 pts_voxel_encoder=pts_voxel_encoder,
                 pts_middle_encoder=pts_middle_encoder,
                 pts_backbone=pts_backbone,
                 pts_neck=pts_neck,
                 bbox_head=bbox_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg)

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

        return bbox_results

    def aug_test(self, img, proj_img, proj_idxs, img_idxs, img_metas,
                 rescale=False):
        pass
