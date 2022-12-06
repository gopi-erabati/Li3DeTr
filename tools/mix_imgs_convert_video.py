import cv2
import numpy as np
import os
from tqdm import tqdm
import natsort

# data_root = '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results/msf3ddetr_v4/images/'
data_root = '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results/Li3DeTr' \
            '/LiDAR3DDETREncDec_voxel-nus_q900_trail1/images/'
img_array = []

img_names = os.listdir(data_root)
img_names = natsort.natsorted(img_names)

for imgname in tqdm(img_names):
    img_name = data_root + imgname
    img = cv2.imread(img_name)

    # put text
    # text
    textgt = 'Green: Ground-truth'
    textpred = 'Blue: Predicted'

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    orggt = (690, 50)
    orgpred = (690, 75)
    # fontScale
    fontScale = 1
    # Red color in BGR
    colorgt = (0, 255, 0)
    colorpred = (255, 0, 0)
    # Line thickness of 1
    thickness = 1

    # Using cv2.putText() method
    img = cv2.putText(img, textgt, orggt, font, fontScale,
                      colorgt, thickness, cv2.LINE_4, False)
    img = cv2.putText(img, textpred, orgpred, font, fontScale,
                      colorpred, thickness, cv2.LINE_4, False)

    img_array.append(img)
    # break

# cv2.imshow('img', img)
# cv2.waitKey()

out = cv2.VideoWriter('/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results'
                      '/Li3DeTr/LiDAR3DDETREncDec_voxel-nus_q900_trail1'
                      '/predictions.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 7, (1024, 1024))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
