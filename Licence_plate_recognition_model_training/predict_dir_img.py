
import time

import cv2
import numpy as np
import os
from tqdm import tqdm

from yolo import YOLOX_infer

if __name__ == "__main__":
    yolox_weight = '../../weights/best_epoch_weights.pth'
    classes_path = './model_data/self_classes.txt'
    input_shape = [640, 640]
    phi = 's'
    confidence = 0.5
    nms_iou = 0.3
    cuda = 'cuda:0'

    # 目标检测模型初始化
    yoloX = YOLOX_infer(model_path=yolox_weight, classes_path=classes_path, input_shape=input_shape,
                        phi=phi, confidence=confidence, nms_iou=nms_iou, cuda=cuda)

    dir_origin_path = r"../../img"
    dir_save_path = "img/img_out/"

    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = cv2.imread(image_path)
            lst_result, r_image = yoloX.infer(image)
            cv2.imwrite(os.path.join(dir_save_path, img_name), r_image)

