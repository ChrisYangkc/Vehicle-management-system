
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLOX_infer

if __name__ == "__main__":
    yolox_weight = r"D:/Bishe_Program/Licence_plate_recognition_model_training/weigth/best_epoch_weights.pth"
    classes_path = r"D:/Bishe_Program/Licence_plate_recognition_model_training/model_data/self_classes.txt"
    input_shape = [640, 640]
    phi = 's'
    confidence = 0.5
    nms_iou = 0.3
    cuda = 'cuda:0'

    # 目标检测模型初始化
    yoloX = YOLOX_infer(model_path=yolox_weight, classes_path=classes_path, input_shape=input_shape,
                        phi=phi, confidence=confidence, nms_iou=nms_iou, cuda=cuda)

    video_path = r'../../img/test_video.mp4'
    capture = cv2.VideoCapture(video_path)

    fps = 0.0
    while True:

        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            # continue
            break
        t1 = time.time()
        # 进行检测
        lst_result, r_image = yoloX.infer(frame)
        # RGBtoBGR满足opencv显示格式
        # frame = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        r_image = cv2.putText(r_image, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", r_image)
        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break

    print("Video Detection Done!")
    capture.release()
    cv2.destroyAllWindows()

