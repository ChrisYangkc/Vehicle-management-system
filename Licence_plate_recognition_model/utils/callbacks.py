import os

import torch
import matplotlib

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import decode_outputs, non_max_suppression
from .utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        # 初始化函数，设置日志目录、损失列表、验证损失列表
        self.log_dir = log_dir  # 日志文件存储目录
        self.losses = []  # 记录训练损失的列表
        self.val_loss = []  # 记录验证损失的列表

        os.makedirs(self.log_dir)  # 创建日志目录
        self.writer = SummaryWriter(self.log_dir)  # 创建TensorBoard写入器
        try:
            # 尝试在TensorBoard中添加模型图
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            # 如果添加模型图失败，则忽略
            pass

    def append_loss(self, epoch, loss, val_loss):
        # 向日志中添加损失和验证损失，并绘制损失曲线
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)  # 添加训练损失
        self.val_loss.append(val_loss)  # 添加验证损失

        # 将损失写入文本文件
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        # 将损失写入TensorBoard
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()  # 绘制损失曲线

    def loss_plot(self):
        # 绘制训练和验证损失曲线
        iters = range(len(self.losses))

        plt.figure()
        # 绘制训练损失和验证损失
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            # 如果损失列表长度足够，尝试绘制平滑的损失曲线
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        except:
            # 如果绘制平滑曲线失败，则忽略
            pass

        plt.grid(True)  # 添加网格
        plt.xlabel('Epoch')  # 设置x轴标签
        plt.ylabel('Loss')  # 设置y轴标签
        plt.legend(loc="upper right")  # 添加图例

        # 保存损失曲线图像
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()  # 清除当前活动轴
        plt.close("all")  # 关闭所有图像窗口



class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True,
                 MINOVERLAP=0.5, eval_flag=True, period=1):
        # 初始化函数，设置模型评估时使用的参数
        super(EvalCallback, self).__init__()  # 调用父类的构造函数

        self.net = net  # 网络模型
        self.input_shape = input_shape  # 输入数据的形状
        self.class_names = class_names  # 类别名称
        self.num_classes = num_classes  # 类别总数
        self.val_lines = val_lines  # 验证集的数据行
        self.log_dir = log_dir  # 日志目录
        self.cuda = cuda  # 是否使用CUDA（用于指定是否在GPU上运行）
        self.map_out_path = map_out_path  # 用于存储中间输出的目录路径，默认为".temp_map_out"
        self.max_boxes = max_boxes  # 每张图片处理的最大检测框数
        self.confidence = confidence  # 置信度阈值
        self.nms_iou = nms_iou  # 非最大抑制（Non-Maximum Suppression, NMS）的IOU阈值
        self.letterbox_image = letterbox_image  # 是否对图像进行letterbox处理，使得图像不失真地缩放
        self.MINOVERLAP = MINOVERLAP  # 计算mAP时使用的最小重叠阈值
        self.eval_flag = eval_flag  # 是否进行评估的标志
        self.period = period  # 评估周期，每多少个epoch进行一次评估

        self.maps = [0]  # 存储每个epoch的mAP值，初始为0
        self.epoches = [0]  # 存储每次评估对应的epoch数，初始为0
        if self.eval_flag:
            # 如果设置为进行评估，将初始mAP值写入日志文件
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")


    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, conf_thres=self.confidence, nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]

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

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                # ------------------------------#
                #   读取图像并转换成RGB图像
                # ------------------------------#
                image = Image.open(line[0])
                # ------------------------------#
                #   获得预测框
                # ------------------------------#
                gt_boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                # ------------------------------#
                #   获得预测txt
                # ------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)

                # ------------------------------#
                #   获得真实框txt
                # ------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
