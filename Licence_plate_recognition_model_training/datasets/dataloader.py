import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input
class Album:
    def __init__(self, inpt_size):
        # 初始化函数，设置图像的高度和宽度
        self.height = inpt_size[0]
        self.width = inpt_size[1]

        # 设置图像预处理操作：缩放、填充和调整大小
        self.trans_resize = A.Compose([
            # 按照最长边进行缩放
            A.LongestMaxSize(max_size=max(self.width, self.height)),
            # 添加边缘填充
            A.PadIfNeeded(min_height=self.height, min_width=self.width, border_mode=cv2.BORDER_CONSTANT),
            # 缩放到指定大小
            A.Resize(height=self.height, width=self.width),
            # 传入的box是voc格式，当box面积小于4时，抛弃此box
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=4))

        # 设置图像增强操作：翻转、缩放、平移、旋转、亮度和对比度调整、RGB通道偏移
        self.transform = A.Compose([
            # 左右翻转
            A.HorizontalFlip(p=0.5),
            # 随机缩放、平移和旋转
            A.ShiftScaleRotate(p=0.5),
            # 随机亮度和对比度调整
            A.RandomBrightnessContrast(p=0.3),
            # 随机 RGB 通道偏移
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            # 传入的box是voc格式，当box面积小于4时，抛弃此box
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=4))


class YoloDataset(Dataset):
    def __init__(self, annotation_path, input_shape, transform):
        # 初始化函数，读取数据集信息
        super(YoloDataset, self).__init__()
        # 读取数据集对应的txt
        with open(annotation_path, encoding='utf-8') as f:
            self.annotation_lines = f.readlines()
        self.input_shape = input_shape
        self.transform = transform

        # 设置归一化参数
        self.normal = 255.0
        self.rgb_means = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 利用Album进行数据增强
        self.Album = Album(self.input_shape)

    def __len__(self):
        # 返回数据集的大小
        return len(self.annotation_lines)

    def __getitem__(self, index):
        # 获取单个数据项
        line = self.annotation_lines[index].split()
        # 对图像进行预处理
        img = cv2.imread(line[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = img.shape

        # 解析bounding boxes
        bboxes = [list(map(int, box.split(','))) for box in line[1:]]
        # 根据image大小修改box
        bboxes = self.clip_box_to_image(bboxes, image_height, image_width)
        # 无形变resize
        transformed = self.Album.trans_resize(image=img, bboxes=bboxes)
        img = transformed['image']
        bboxes = transformed['bboxes']
        # 是否进行transform变换
        if self.transform:
            transformed = self.Album.transform(image=img, bboxes=bboxes)
            img = transformed['image']
            bboxes = transformed['bboxes']

        # 归一化
        img = np.array(img / self.normal, dtype=np.float32)
        # 减去均值后除以标准差
        img = (img - self.rgb_means) / self.std

        # 调整图像维度顺序
        image = np.transpose(img, (2, 0, 1))
        # 将bounding boxes转换为numpy数组
        box = np.array(bboxes, dtype=np.float32)
        # 将bounding boxes的格式转换为中心点坐标加宽高
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def clip_box_to_image(self, bboxes, image_height, image_width):
        # 将bounding boxes限制在图像内部
        np.random.shuffle(bboxes)
        lst_box = []
        for box in bboxes:
            x1, y1, x2, y2, cls = box
            # 将 x1 和 x2 约束在 [0, image_width) 的范围内
            x1 = max(0, min(x1, image_width - 1))
            x2 = max(0, min(x2, image_width - 1))
            # 将 y1 和 y2 约束在 [0, image_height) 的范围内
            y1 = max(0, min(y1, image_height - 1))
            y2 = max(0, min(y2, image_height - 1))
            # 如果经过修正后的 x1 或 y1 大于等于 x2 或 y2，则将其设置为 0
            if x1 >= x2 or y1 >= y2:
                x1, y1, x2, y2 = 0, 0, 0, 0
            lst_box.append((x1, y1, x2, y2, cls))
        return lst_box

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    # 批量处理函数，整合一批数据
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    # 将图像和bounding boxes转换为torch张量
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes
