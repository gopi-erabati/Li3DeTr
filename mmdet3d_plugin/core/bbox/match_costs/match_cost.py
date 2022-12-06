import torch
from torch.nn.functional import smooth_l1_loss

from mmdet.core.bbox.match_costs.builder import MATCH_COST


@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class BBox3DSmoothL1Cost:
    """
    BBox L1 cost for 3D bounding boxes

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox3d_pred, gt3d_bboxes):
        """

        Args:
            bbox3d_pred (Tensor): Predicted 3D bounding boxes (cx, cy, cz,
            l, w, h, theta). Shape [num_query, 7]
            gt3d_bboxes (Tensor): Ground truth 3D bounding boxes (cx, cy, cz,
            l, w, h, theta). Shape [num_gt, 7]

        Returns:
            torch.Tensor: bbox_cost value with weight.
            Shape [num_query, num_gt]
        """

        bbox3d_pred_expand = bbox3d_pred.unsqueeze(1).repeat(1,
                                                             gt3d_bboxes.shape[
                                                                 0], 1)
        gt3d_bboxes_expand = gt3d_bboxes.unsqueeze(0).repeat(
            bbox3d_pred.shape[0], 1, 1)

        bbox3d_cost = smooth_l1_loss(bbox3d_pred_expand, gt3d_bboxes_expand,
                                     reduction='none').sum(-1)

        return bbox3d_cost * self.weight