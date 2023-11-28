import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input


class Album:
    def __init__(self, inpt_size):
        self.height = inpt_size[0]
        self.width = inpt_size[1]

        self.trans_resize = A.Compose([
            # 按照最长边进行缩放
            A.LongestMaxSize(max_size=max(self.width, self.height)),
            # 添加边缘填充
            A.PadIfNeeded(min_height=self.height, min_width=self.width, border_mode=cv2.BORDER_CONSTANT),
            # 缩放到指定大小
            A.Resize(height=self.height, width=self.width),
            # 传入的box是voc格式，当box面积小于4时，抛弃此box
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=4))

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
        super(YoloDataset, self).__init__()
        # 读取数据集对应的txt
        with open(annotation_path, encoding='utf-8') as f:
            self.annotation_lines = f.readlines()
        self.input_shape = input_shape
        self.transform = transform

        self.normal = 255.0
        self.rgb_means = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 利用Album进行数据增强
        self.Album = Album(self.input_shape)

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        line = self.annotation_lines[index].split()
        # 对图像进行预处理
        img = cv2.imread(line[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = img.shape

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
        # 均值
        img = (img - self.rgb_means) / self.std

        image = np.transpose(img, (2, 0, 1))
        box = np.array(bboxes, dtype=np.float32)
        # 改为 cx cy w h
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def clip_box_to_image(self, bboxes, image_height, image_width):
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
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes
