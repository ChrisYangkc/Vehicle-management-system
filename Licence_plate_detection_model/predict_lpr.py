# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''
from torch.utils.data import DataLoader


from PIL import Image, ImageDraw, ImageFont

# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os

from models.LPRNet import CHARS, LPRNet
from datasets.dataloader import LPRDataset


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default=r".\traindata\train\blue", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return torch.stack(imgs, 0), torch.from_numpy(labels), lengths

def test():
    args = get_parser()

    weights = r'.\runs\best.pth'
    cuda = "cuda:0"
    lprnet = LPRNet(lpr_max_len=args.lpr_max_len, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device(cuda)

    print("Successful to build network!")
    # 模型初始化
    lprnet.load_state_dict(torch.load(weights))
    lprnet = lprnet.eval()
    lprnet.to(device)

    with torch.no_grad():
        for root, dirs, files in os.walk(args.test_img_dirs):
            for file in files:
                print('标签：', file.split('.')[0])
                if 'jpg' not in file:
                    continue
                filename = os.path.join(root, file)

                img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
                img = cv2.resize(img, args.img_size)
                img = img.astype('float32')
                img -= 127.5
                img *= 0.0078125
                img = np.transpose(img, (2, 0, 1))

                img = torch.from_numpy(img)
                img = img.unsqueeze(0)
                img = Variable(img.cuda())
                prebs = lprnet(img)
                prebs = prebs.cpu().detach().numpy()

                for preb in prebs:
                    # preb 每张图片 [68, 18]
                    preb_label = []
                    for j in range(preb.shape[1]):  # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
                        preb_label.append(np.argmax(preb[:, j], axis=0))

                    no_repeat_blank_label = []
                    pre_c = preb_label[0]
                    if pre_c != len(CHARS) - 1:  # 记录重复字符
                        no_repeat_blank_label.append(CHARS[pre_c])
                    for c in preb_label:  # 去除重复字符和空白字符'-'
                        if pre_c == c or c == len(CHARS) - 1:
                            if c == len(CHARS) - 1:
                                pre_c = c
                            continue
                        no_repeat_blank_label.append(CHARS[c])
                        pre_c = c
                print('预测：', ''.join(no_repeat_blank_label))



if __name__ == "__main__":
    test()
