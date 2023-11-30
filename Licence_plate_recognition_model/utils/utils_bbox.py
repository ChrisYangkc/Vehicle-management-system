import numpy as np
import torch
from torchvision.ops import nms, boxes


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    # if letterbox_image:
    # -----------------------------------------------------------------#
    #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    #   new_shape指的是宽高缩放情况
    # -----------------------------------------------------------------#
    new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def decode_outputs(outputs, input_shape):
    grids = []
    strides = []
    # 取出每一层的大小[[80, 80], [40, 40]...]
    hw = [x.shape[-2:] for x in outputs]
    # ---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   flatten:
    #       batch_size, 4 + 1 + num_classes, 80, 80 --> batch_size, 4 + 1 + num_classes, 6400
    #       batch_size, 5 + num_classes, 40, 40
    #       batch_size, 5 + num_classes, 20, 20
    #   torch.cat:
    #       batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   permute(0, 2, 1):
    #       堆叠后为batch_size, 8400, 5 + num_classes
    # ---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    # sigmoid 将是否包含物体，以及每个特征点属于每一个种类的概率进行sigmoid
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        # 根据特征层的高宽生成网格点
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        # ---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        # ---------------------------#
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h)) # 使用下采样倍率填充 8、 16、 32
    # ---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    # ---------------------------#
    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())
    # ------------------------#
    #   根据网格点进行解码
    #   乘 strides: 将从下采样8、16、32 倍率下得到的xywh，恢复到以原图为基准的xywh
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # -----------------#
    #   归一化
    # -----------------#
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
    return outputs


def non_max_suppression(prediction, num_classes, input_shape, image_shape, conf_thres=0.5,
                        nms_thres=0.4):
    # ----------------------------------------------------------#
    #   将预测结果的格式(中心点、h、w格式)转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.clone()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    # ----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    # ----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [8400, 1]    种类置信度
        #   class_pred  [8400, 1]    种类
        # ----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选conf_mask (8400,)
        # ----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [8400, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        scores = detections[:, 4] * detections[:, 5]
        # 进行非极大抑制
        nms_out_index = boxes.batched_nms(detections[:, :4], scores, detections[:, 6], nms_thres)

        output[i] = detections[nms_out_index]

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    return output
