
import os
import time

import cv2
import random
import numpy as np
import torch
import torch.nn as nn

from .dataset.data_predict import ImagePreprocessor
from .nets.yolo import YoloX
from .utils.utils import get_classes, show_config
from .utils.utils_bbox import decode_outputs, non_max_suppression


class YOLOX_infer(object):

    def __init__(self, model_path, classes_path, input_shape, phi, confidence, nms_iou, cuda):

        self.model_path = model_path
        self.classes_path = classes_path
        # 输入图片的大小，必须为32的倍数。 格式为： [h, w]
        self.input_shape = input_shape
        # nano、tiny、s、m、l、x
        self.phi = phi
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.cuda = cuda
        # 数据预处理
        self.normal = 255.0
        self.rgb_means = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 获取类别和类别数量
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # 画框设置不同的颜色
        self.colors = self.generate_color_list(self.num_classes)

        # 模型初始化
        self.net = YoloX(self.num_classes, self.phi)
        device = torch.device(self.cuda)
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net.to(self.cuda)
        # print(self.net)
        # 加载数据预处理
        self.preprocessor = ImagePreprocessor(self.input_shape, self.normal, self.rgb_means, self.std, self.cuda)

        # 打印参数
        show_config(model_path=self.model_path, classes_path=self.classes_path, input_shape=self.input_shape,
                    phi=self.phi, confidence=self.confidence, nms_iou=self.nms_iou, cuda=self.cuda)

    def infer(self, image):
        # 图片的高和宽
        image_shape = image.shape[0:2]
        # 数据预处理 b, c, h, w
        if 'cpu' in self.cuda:
            images = self.preprocessor.preprocess_cpu(image)
        else:
            images = self.preprocessor.preprocess_gpu(image)

        with torch.no_grad():
            # 预测推理
            outputs = self.net(images)

            # 输出解码
            outputs = decode_outputs(outputs, self.input_shape)
            # 非极大抑制
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return None

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        lst_result = []
        for i, c in list(enumerate(top_label)):
            label = self.class_names[int(c)]
            box = top_boxes[i]
            score = round(top_conf[i], 2)
            # 注意顺序
            y1, x1, y2, x2 = box
            y1 = max(0, np.floor(y1).astype('int32'))
            x1 = max(0, np.floor(x1).astype('int32'))
            y2 = min(image_shape[1], np.floor(y2).astype('int32'))
            x2 = min(image_shape[0], np.floor(x2).astype('int32'))
            lst_result.append([label, score, [x1, y1, x2, y2]])
        img_res = self.vis_box(image, lst_result)
        return lst_result, img_res

    def vis_box(self, img, lst_result):
        # 绘制矩形框和类别信息
        for data in lst_result:
            color = self.colors[0]  # 获取对应类别的颜色
            thickness = 2  # 矩形框线条粗细
            x1, y1, x2, y2 = data[2]  # 矩形框左上角和右下角坐标
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # 在矩形框上添加类别和置信度信息
            label = f"{data[0]}: {data[1]:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 3
            size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
            pt1 = (x1, y1 - size[1] - 20)
            pt2 = (x1 + size[0] + 10, y1)
            cv2.rectangle(img, pt1, pt2, color, cv2.FILLED)
            cv2.putText(img, label, (x1, y1 - baseline), font, font_scale, (0, 0, 0), thickness)
        return img

    def generate_color_list(self, num_classes, seed=0):
        # 设置随机数种子，保证每次运行程序得到的随机序列是固定的
        random.seed(seed)
        # 生成对应数量的 RGB 颜色值，取值范围为 0-255
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                  range(num_classes)]
        return colors

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        # 图片的高和宽
        image_shape = image.shape[0:2]
        # 数据预处理 b, c, h, w
        images = self.preprocessor.preprocess_gpu(image)

        with torch.no_grad():

            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)

            # 非极大抑制
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return
