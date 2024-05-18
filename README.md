# Li3DeTr: A LiDAR based 3D Detection Transformer

This is the official PyTorch implementation of the paper **[Li3DeTr: A LiDAR based 3D Detection Transformer](https://openaccess.thecvf.com/content/WACV2023/papers/Erabati_Li3DeTr_A_LiDAR_Based_3D_Detection_Transformer_WACV_2023_paper.pdf)**, by Gopi Krishna Erabati and Helder Araujo at *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023*.

Our implementation is based on MMDetection3D.

## Abstract
Inspired by recent advances in vision transformers for object detection, we propose Li3DeTr, an end-to-end LiDAR based 3D Detection Transformer for autonomous driving, that inputs LiDAR point clouds and regresses 3D bounding boxes. The LiDAR local and global features are encoded using sparse convolution and multi-scale deformable attention respectively. In the decoder head, firstly, in the novel Li3DeTr cross-attention block, we link the LiDAR global features to 3D predictions leveraging the sparse set of object queries learnt from the data. Secondly, the object query interactions are formulated using multi-head self-attention. Finally, the decoder layer is repeated L dec number of times to refine the object queries. Inspired by DETR, we employ set-to-set loss to train the Li3DeTr network. Without bells and whistles, the Li3DeTr network achieves 61.3% mAP and 67.6% NDS surpassing the state-of-the-art methods with non-maximum suppression (NMS) on the nuScenes dataset and it also achieves competitive performance on the KITTI dataset. We also employ knowledge distillation (KD) using a teacher and student model that slightly improves the performance of our network.

![Li3DeTr](https://user-images.githubusercontent.com/22390149/198293675-5cc61685-d50f-434c-8c85-a1cb94f4da6b.png)

## Results

### Predictions on nuScenes dataset

![li3detr_pred](https://user-images.githubusercontent.com/22390149/198314047-d9e8c3b1-1eb4-4a65-a73e-71eda4101156.gif)

### nuScenes Dataset

| LiDAR Backbone | mAP | NDS | Weights |
| :---------: | :----: |:----: | :------: |
| VoxelNet | 61.3 | 67.6 | [Model](https://drive.google.com/file/d/1C_aBs9uyghA26T3nFoLQMg-4L97FA-Vo/view?usp=sharing) |
| PointPillars | 53.8 | 63.0 | [Model](https://drive.google.com/file/d/1IQ8tko4LY-kgLDHb7OKdJxjdPP4IRTTG/view?usp=sharing) |

### KITTI Dataset (AP<sub>3D</sub>)

| LiDAR Backbone | Easy | Mod. | Hard | Weights |
| :---------: | :----: | :----: |:----: | :------: |
| VoxelNet | 87.6 | 76.8 | 73.9 | [Model](https://drive.google.com/file/d/1PsKu-yOY0EJJSLzgNB0IJCoPLt-jm3A_/view?usp=sharing) |

## Usage

### Prerequisite

The code is tested on the following configuration:
- python==3.6
- cuda==11.1
- PyTorch==1.8.1
- [mmcv](https://github.com/open-mmlab/mmcv)==1.4.2
- [mmdet](https://github.com/open-mmlab/mmdetection)==2.20.0
- [mmseg](https://github.com/open-mmlab/mmsegmentation)==0.20.2
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)==0.18.0

### Data
Follow [MMDetection3D](https://mmdetection3d.readthedocs.io/en/latest/data_preparation.html) to prepare the nuScenes dataset and symlink the data directory to `data/` folder of this repository.

### Clone the repository
```
git clone https://github.com/gopi231091/Li3DeTr.git
cd Li3DeTr
```

### Training

1. Download the [backbone pretrained weights]() to `ckpts/`
2. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
3. To train the MSF3DDETR with ResNet101 and VoxelNet backbones on 2 GPUs, please run

`tools/dist_train.sh configs/li3detr_voxel_adam_nus-3d.py 2 --work-dir {WORK_DIR}`

### Testing
1. Downlaod the weights of the models accordingly.
2. Add the present working directory to PYTHONPATH `export PYTHONPATH=$(pwd):$PYTHONPATH`
3. To evaluate the model using 2 GPUs, please run

`tools/dist_test.sh configs/li3detr_voxel_adam_nus-3d.py /path/to/ckpt 2 --eval=bbox`

## Acknowlegement
We sincerely thank the contributors for their open-source code: [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

## Reference
Feel free to cite our article if you find our method useful.
```
@InProceedings{Erabati_2023_WACV,
    author    = {Erabati, Gopi Krishna and Araujo, Helder},
    title     = {Li3DeTr: A LiDAR Based 3D Detection Transformer},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {4250-4259}
}
```


