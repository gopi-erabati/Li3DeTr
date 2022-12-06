import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# For Encoder and Decoder in Transformer to get global relationship in Encoder
@TRANSFORMER.register_module()
class Li3DeTrTransformerEncDec(BaseModule):
    """Implements the MSF3DDETR transformer (Encoder + Decoder).
    Args:
        num_feature_levels_point (int): number of feat mpas of BEV LiDAR feats
        encoder (dict): Encoder config dict
        decoder (dict): Decoder config dict
    """

    def __init__(self,
                 num_feature_levels_point=4,
                 encoder=None,
                 decoder=None,
                 **kwargs):
        super(Li3DeTrTransformerEncDec, self).__init__(**kwargs)
        self.encoder_point = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels_point = num_feature_levels_point
        self.init_layers()

    def init_layers(self):
        self.level_embeds_point = nn.Parameter(torch.Tensor(
            self.num_feature_levels_point, self.embed_dims))
        # self.query_pos = nn.Linear(3, self.embed_dims)
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or \
                    isinstance(m, Li3DeTrCrossAttention):
                m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds_point)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self,
                mlvl_feats_point,
                mlvl_masks_point,
                mlvl_pos_enc_point,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for Li3DeTr Transformer.
        Args:
            mlvl_feats_point (list[Tensor]): list of point feats of shape
                (B, C, H, W)
            mlvl_masks_point (list[Tensor]): Multi-level masks for point feats
                of shape (B, C, H, W)
            mlvl_pos_enc_point (list[Tensor]): Multi-level position encoding
                of shape (B, C, H, W)
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, d].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, num_query, bs, embed_dims), else has \
                      shape (1, num_query, bs, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 3).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, 3)
        """
        assert query_embed is not None

        # Encoder
        # for image and Point features
        feat_flatten_point = []
        mask_flatten_point = []
        lvl_pos_embed_flatten_point = []
        spatial_shapes_point = []
        for lvl, (feat_point, mask_point, pos_embed_point) in enumerate(
                zip(mlvl_feats_point, mlvl_masks_point, mlvl_pos_enc_point)):
            # point (BEV)
            bs_point, c_point, h_point, w_point = feat_point.shape
            spatial_shape_point = (h_point, w_point)
            spatial_shapes_point.append(spatial_shape_point)
            feat_point = feat_point.flatten(2).transpose(1, 2)  # (bs, h*w, c)
            mask_point = mask_point.flatten(1)  # (bs, h*w)
            pos_embed_point = pos_embed_point.flatten(2).transpose(1,
                                                                   2)  # (bs, h*w, c)
            lvl_pos_embed_point = pos_embed_point + self.level_embeds_point[
                lvl].view(1, 1, -1)
            lvl_pos_embed_flatten_point.append(lvl_pos_embed_point)
            # [(bs, h*w, c)...]
            feat_flatten_point.append(feat_point)  # [(bs, h*w, c) ...]
            mask_flatten_point.append(mask_point)  # [(bs, h*w)...]

        # point (BEV) encoder
        feat_flatten_point = torch.cat(feat_flatten_point,
                                       1)  # (bs, lvl*h*w, c)
        mask_flatten_point = torch.cat(mask_flatten_point, 1)
        lvl_pos_embed_flatten_point = torch.cat(lvl_pos_embed_flatten_point, 1)
        spatial_shapes_point_tensor = torch.as_tensor(
            spatial_shapes_point, dtype=torch.long,
            device=feat_flatten_point.device)
        level_start_index_point = torch.cat(
            (spatial_shapes_point_tensor.new_zeros(
                (1,)),
             spatial_shapes_point_tensor.prod(1).cumsum(0)[:-1]))  # (lvl, )
        valid_ratios_point = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks_point],
            1)  # (bs, lvls, 2)

        reference_points_point = \
            self.get_reference_points(spatial_shapes_point_tensor,
                                      valid_ratios_point,
                                      device=feat_point.device)  # (bs, lvl*h*w, 2)

        feat_flatten_point = feat_flatten_point.permute(1, 0,
                                                        2)  # (lvl*H*W, bs,
        # embed_dims)
        lvl_pos_embed_flatten_point = lvl_pos_embed_flatten_point.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory_point = self.encoder_point(
            query=feat_flatten_point,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten_point,
            query_key_padding_mask=mask_flatten_point,
            spatial_shapes=spatial_shapes_point_tensor,
            reference_points=reference_points_point,
            level_start_index=level_start_index_point,
            valid_ratios=valid_ratios_point,
            **kwargs)
        # (lvl*h*w, bs, c)
        # bring back the shape of feature maps
        memory_point_list = memory_point.split([H_ * W_ for H_, W_ in
                                                spatial_shapes_point],
                                               dim=0)
        # this creates list [(h*w, bs, c), (h*w, bs, c), ... for lvls]
        memory_point_fmap_list = []
        for level, (H_, W_) in enumerate(spatial_shapes_point):
            memory_point_fmap = memory_point_list[level].permute(
                1, 2, 0).reshape(bs_point, c_point, H_, W_)
            memory_point_fmap_list.append(memory_point_fmap)
            # this contains list [(bs, c, h, w), ... for levels]

        # from the Encoder two feature maps are:
        # for images memory_img_views_fmap_list_updated
        # [(bs, n_img, c, h, w), ... levels]
        # for point memory_point_fmap_list
        # [(bs, c, h, w), ... for levels]

        # Decoder

        bs = memory_point_fmap_list[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)  # (N, B, d)
        query_pos = query_pos.permute(1, 0, 2)  # (N, B, d)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory_point_fmap_list,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)
        # (#layers, n_q, B, d)
        # (#layers, B, n_q, 3)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out
        # (#layers, n_q, B, d)
        # (B, n_q, 3)
        # (#layers, B, n_q, 3)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Li3DeTrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in Li3DeTr transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Li3DeTrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for Li3DeTr Transformer.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points has shape (bs, num_query, 3) .
            reg_branches: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            (tuple):
                if return_intermediate is True
                    intermediate_query of shape (#layers, n_q, B, d)
                    intermediate_ref_points of shape (#layers, B, n_q, 3)
                else
                    output of shape (n_q, B, d)
                    ref_points of shape (B, n_q, 3)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)  # (B, n_q, d)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)  # (B, n_q, 3)

                assert reference_points.shape[-1] == 3

                # for center_branch
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] \
                                                + inverse_sigmoid(
                    reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 2:3] + \
                                                 inverse_sigmoid(
                                                     reference_points[...,
                                                     2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)  # (n_q, B, d)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
            # (#layers, n_q, B, d)
            # (#layers, B, n_q, 3)

        return output, reference_points
        # (n_q, B, d)
        # (B, n_q, 3)


@ATTENTION.register_module()
class Li3DeTrCrossAttention(BaseModule):
    """An attention module used in Detr3d.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels_point (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        pc_range (list): point cloud range in X, Y and Z
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        voxel_size (list): The voxel sizes in X, Y and Z
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels_point=4,
                 num_points=5,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False,
                 voxel_size=None,
                 ):
        super(Li3DeTrCrossAttention, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.voxel_size = voxel_size

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels_point = num_levels_point
        self.num_heads = num_heads
        self.num_points = num_points

        self.attention_weights_points = nn.Linear(embed_dims,
                                                  num_levels_point * num_points)
        self.output_proj_points = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights_points, val=0., bias=0.)
        xavier_init(self.output_proj_points, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                **kwargs):
        """Forward Function of Li3DeTrCrossAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (list[Tensor]): The RGB and LiDAR feats, of shape
                (B, N, C, H, W) and (B, C, H, W)
            value (list[Tensor]): The RGB and LiDAR feats, of shape
                (B, N, C, H, W) and (B, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 3).
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)  # (B, n_q, d)

        # Image Features Cross Attn
        bs, num_query, _ = query.size()

        # Point Features Cross Attn
        point_feats = value  # (n_q, B, d)

        reference_points_3d, output_points, mask_points = feature_sampling_points(
            point_feats,
            reference_points,
            self.pc_range,
            self.voxel_size)
        # (B, C, n_q, 1, 4)
        # (B, 1, n_q, 1, 1)
        output_points = torch.nan_to_num(output_points)
        mask_points = torch.nan_to_num(mask_points)

        attention_weights_points = self.attention_weights_points(
            query).view(bs, 1, num_query, self.num_points,
                        self.num_levels_point)
        # (B, 1, n_q, 1, 4)
        attention_weights_points = attention_weights_points.sigmoid() * mask_points
        # (B, 1, n_q, 1, 4)
        output_points = output_points * attention_weights_points
        # (B, C, n_q, 1, 4)
        output_points = output_points.sum(-1).sum(-1)  # (B, C, n_q)
        output_points = output_points.permute(2, 0, 1)  # (n_q, B, C)

        output_fused = self.output_proj_points(output_points)  # (n_q, B, C)

        pos_feat = self.position_encoder(inverse_sigmoid(
            reference_points_3d)).permute(1, 0, 2)
        # (n_q, B, embed_dims)

        return self.dropout(output_fused) + inp_residual + pos_feat
        # (n_q, B, embed_dims)


def feature_sampling_points(mlvl_feats, reference_points, pc_range,
                            voxel_size):
    """ Feature Sampling for LiDAR BEV Features
        Args:
            mlvl_feats (list[Tensor]): Multi level LiDAR BEV feats of shape
                (B, C, H, W)
            reference_points (Tensor): Normalized reference points of shape
                (B, n_q, 3)
            pc_range (list): Point cloud range in X, Y and Z axis
            voxel_size (list): Voxel size in X, Y and Z axis.
        Returns:
            (tuple):
                sampled_feats (Tensor): sampled feats of shape (B, C, n_q, 1, 4)
                mask (Tensor): Mask for camera views of shape (B, 1, n_q, 1, 1)
        """
    reference_points = reference_points.clone()  # (B, n_q, 3)
    reference_points_3d = reference_points.clone()

    B, num_query = reference_points.size()[:2]

    reference_points[..., 0:1] = reference_points[..., 0:1] * (
            pc_range[3] - pc_range[0])  # 0 to 102.4
    reference_points[..., 1:2] = reference_points[..., 1:2] * (
            pc_range[4] - pc_range[1])
    reference_points[..., 2:3] = reference_points[..., 2:3] * (
            pc_range[5] - pc_range[2])

    reference_points[..., 0:1] = reference_points[..., 0:1] / voxel_size[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] / voxel_size[1]
    # 0 to 512

    reference_points_bev = reference_points[..., 0:2]  # (B, n_q, 2) 0 to 512.

    if voxel_size[0] == 0.2:
        point_feat_hw = 512
    elif voxel_size[0] == 0.1:
        point_feat_hw = 1024
    reference_points_bev /= point_feat_hw  # (B, n_q, 2)  divide by points feature
    # height, width which is 512 x 512

    reference_points_bev = (reference_points_bev - 0.5) * 2  # (B, n_q, 2)
    mask = ((reference_points_bev[..., 0:1] > -1.0)
            & (reference_points_bev[..., 0:1] < 1.0)
            & (reference_points_bev[..., 1:2] > -1.0)
            & (reference_points_bev[..., 1:2] < 1.0))  # (B, n_q, 1)

    mask = mask.view(B, 1, num_query, 1, 1)
    mask = torch.nan_to_num(mask)  # (B, 1, n_q, 1, 1)

    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, C, H, W = feat.size()
        reference_points_bev_lvl = reference_points_bev.view(B, num_query,
                                                             1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_bev_lvl)
        # (B, C, n_q, 1)
        # input (BN, C, Hin, Win) ; grid (BN, Hout, Wout, 2) then output has
        # the shape (BN, C, Hout, Wout). So here the output has the shape of
        # (BN, C, n_q, 1)  [x-coord - W ; y_coord - H]
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)  # (B, C, n_q, 1, 4)
    sampled_feats = sampled_feats.view(B, C, num_query, 1, len(mlvl_feats))
    # (B, C, n_q, 1, 4)
    return reference_points_3d, sampled_feats, mask
    # (B, C, n_q, 1, 4)
    # (B, 1, n_q, 1, 1)


@ATTENTION.register_module()
class Li3DeTrCrossAttentionKitti(BaseModule):
    """An attention module used in Detr3d.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels_point (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels_point=4,
                 num_points=5,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False,
                 voxel_size=None,
                 ):
        super(Li3DeTrCrossAttentionKitti, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.voxel_size = voxel_size

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels_point = num_levels_point
        self.num_heads = num_heads
        self.num_points = num_points

        # self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.attention_weights_points = nn.Linear(embed_dims,
                                                  num_levels_point * num_points)
        self.output_proj_points = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights_points, val=0., bias=0.)
        xavier_init(self.output_proj_points, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                **kwargs):
        """Forward Function of Li3DeTrCrossAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (list[Tensor]): The RGB and LiDAR feats, of shape
                (B, N, C, H, W) and (B, C, H, W)
            value (list[Tensor]): The RGB and LiDAR feats, of shape
                (B, N, C, H, W) and (B, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 3).
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)  # (B, n_q, d)

        # Image Features Cross Attn
        bs, num_query, _ = query.size()

        # Point Features Cross Attn
        point_feats = value  # (n_q, B, d)

        reference_points_3d, output_points, mask_points = \
            feature_sampling_points_kitti(
            point_feats,
            reference_points,
            self.pc_range,
            self.voxel_size)
        # (B, C, n_q, 1, 4)
        # (B, 1, n_q, 1, 1)
        output_points = torch.nan_to_num(output_points)
        mask_points = torch.nan_to_num(mask_points)

        attention_weights_points = self.attention_weights_points(
            query).view(bs, 1, num_query, self.num_points,
                        self.num_levels_point)
        # (B, 1, n_q, 1, 4)
        attention_weights_points = attention_weights_points.sigmoid() * mask_points
        # (B, 1, n_q, 1, 4)
        output_points = output_points * attention_weights_points
        # (B, C, n_q, 1, 4)
        output_points = output_points.sum(-1).sum(-1)  # (B, C, n_q)
        output_points = output_points.permute(2, 0, 1)  # (n_q, B, C)

        output_fused = self.output_proj_points(output_points)  # (n_q, B, C)

        pos_feat = self.position_encoder(inverse_sigmoid(
            reference_points_3d)).permute(1, 0, 2)
        # (n_q, B, embed_dims)

        return self.dropout(output_fused) + inp_residual + pos_feat
        # (n_q, B, embed_dims)


def feature_sampling_points_kitti(mlvl_feats, reference_points, pc_range,
                            voxel_size):
    """ Feature Sampling for LiDAR BEV Features
    Args:
        mlvl_feats (list[Tensor]): Multi level LiDAR BEV feats of shape
            (B, C, H, W)
        reference_points (Tensor): Normalized reference points of shape
            (B, n_q, 3)
        pc_range (list): Point cloud range in X, Y and Z axis
        voxel_size (list): Voxel size in X, Y and Z axis.
    Returns:
        (tuple):
            sampled_feats (Tensor): sampled feats of shape (B, C, n_q, 1, 4)
            mask (Tensor): Mask for camera views of shape (B, 1, n_q, 1, 1)
    """
    reference_points = reference_points.clone()  # (B, n_q, 3)
    reference_points_3d = reference_points.clone()

    B, num_query = reference_points.size()[:2]

    reference_points[..., 0:1] = reference_points[..., 0:1] * (
            pc_range[3] - pc_range[0])  # 0 to 102.4
    reference_points[..., 1:2] = reference_points[..., 1:2] * (
            pc_range[4] - pc_range[1])
    reference_points[..., 2:3] = reference_points[..., 2:3] * (
            pc_range[5] - pc_range[2])

    reference_points[..., 0:1] = reference_points[..., 0:1] / voxel_size[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] / voxel_size[1]
    # 0 to 512

    reference_points_bev = reference_points[..., 0:2]  # (B, n_q, 2) 0 to 512.

    if voxel_size[0] == 0.16:  # kitti pillar
        point_feat_hw = reference_points.new_tensor([432, 496])
    elif voxel_size[0] == 0.05:  # kitti voxel
        point_feat_hw = reference_points.new_tensor([1408, 1600])
    elif voxel_size[0] == 0.08:  # waymo voxel
        point_feat_hw = reference_points.new_tensor([1920, 1280])
    reference_points_bev /= point_feat_hw  # (B, n_q, 2)  divide by points feature
    # height, width which is 512 x 512

    reference_points_bev = (reference_points_bev - 0.5) * 2  # (B, n_q, 2)
    mask = ((reference_points_bev[..., 0:1] > -1.0)
            & (reference_points_bev[..., 0:1] < 1.0)
            & (reference_points_bev[..., 1:2] > -1.0)
            & (reference_points_bev[..., 1:2] < 1.0))  # (B, n_q, 1)

    mask = mask.view(B, 1, num_query, 1, 1)
    mask = torch.nan_to_num(mask)  # (B, 1, n_q, 1, 1)

    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, C, H, W = feat.size()
        reference_points_bev_lvl = reference_points_bev.view(B, num_query,
                                                             1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_bev_lvl)
        # (B, C, n_q, 1)
        # input (BN, C, Hin, Win) ; grid (BN, Hout, Wout, 2) then output has
        # the shape (BN, C, Hout, Wout). So here the output has the shape of
        # (BN, C, n_q, 1)  [x-coord - W ; y_coord - H]
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)  # (B, C, n_q, 1, 4)
    sampled_feats = sampled_feats.view(B, C, num_query, 1, len(mlvl_feats))
    # (B, C, n_q, 1, 4)
    return reference_points_3d, sampled_feats, mask
    # (B, C, n_q, 1, 4)
    # (B, 1, n_q, 1, 1)


@TRANSFORMER.register_module()
class Li3DeTrTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=1,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Li3DeTrTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        # self.query_pos = nn.Linear(3, self.embed_dims)
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m,
                                                                          Li3DeTrCrossAttention):
                m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                point_feats,
                query_embed,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            reference_points (Tensor): (B, n_q, 3)
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert query_embed is not None
        bs = point_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        # query_pos = query_pos.permute(0, 2, 1)  # (B, n_q, d)
        # query = torch.zeros_like(query_pos)  # (B, n_q, d)
        # query = query.unsqueeze(0).expand(bs, -1, -1)  # (B, n_q, d)

        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        # reference_points = reference_points_org
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)  # (N, B, d)
        query_pos = query_pos.permute(1, 0, 2)  # (N, B, d)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=point_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)
        # (#layers, n_q, B, d)
        # (#layers, B, n_q, 3)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out
        # (#layers, n_q, B, d)
        # (B, n_q, 3)
        # (#layers, B, n_q, 3)