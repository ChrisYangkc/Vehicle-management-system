import torch
import cv2
import numpy as np


class ImagePreprocessor:
    """预测时，预处理输入图片的类，包含在 GPU 和 CPU 上进行预处理的方法"""

    def __init__(self, input_size, normal, rgb_means, std, device='cuda:0'):
        """
        初始化方法

        Args:
            input_size: 神经网络期望的输入图像尺寸，格式为 (高度, 宽度)
            normal: 归一化所需的参数，实际上就是 255.0
            rgb_means: RGB 三通道的均值，格式为 (红色通道均值, 绿色通道均值, 蓝色通道均值)
            std: 标准差，即 1 / 255.0
            device: 默认为cuda:0
        """
        self.input_size = input_size
        self.normal = normal
        self.rgb_means = rgb_means
        self.std = std
        self.device = torch.device(device)
        # 转换到GPU上，并转相应的维度
        self.normal_gpu = torch.from_numpy(normal * np.ones([1, 3, 1, 1], dtype=np.float32)).to(self.device)
        self.rgb_means_gpu = torch.from_numpy(np.array(rgb_means, dtype=np.float32).reshape(1, 3, 1, 1)).to(self.device)
        self.std_gpu = torch.from_numpy(np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)).to(self.device)

    def preprocess_gpu(self, img):
        """
        在 GPU 上预处理输入的图片
        速度：2ms完成预处理
        Args:
            img: 待预处理的原始图片

        Returns:
            处理后的图片，已经放在了CUDA上了
        """
        # 创建一个全零张量作为填充后的图像
        padded_img = torch.zeros((self.input_size[0], self.input_size[1], 3), dtype=torch.float32, device=self.device)

        # 计算图片的缩放比例
        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        # 使用 OpenCV 对图片进行缩放
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r))).astype(np.float32)
        # 将 NumPy 数组转换成张量，并将其移动到 GPU 上
        resized_img = torch.from_numpy(resized_img).to(self.device)

        # 计算填充后的图像在新尺寸中的位置
        h, w = resized_img.shape[:2]
        hr, wr = (self.input_size[0] - h) // 2, (self.input_size[1] - w) // 2
        # 将缩放后的图像放置在填充后的图像中心
        padded_img[hr: hr + h, wr: wr + w] = resized_img

        # 将 BGR 通道顺序转换为 RGB 顺序
        padded_img = padded_img.flip(dims=[2])

        # 将维度从 (H, W, C) 转换成 (C, H, W)，并添加一个额外的维度表示 batch size
        padded_img = padded_img.permute(2, 0, 1).unsqueeze(0)

        # 归一化和减去均值
        padded_img = (padded_img / self.normal_gpu - self.rgb_means_gpu) / self.std_gpu

        return padded_img

    def preprocess_cpu(self, img):
        """
        在 CPU 上预处理输入的图片
        速度：15ms
        Args:
            img: 待预处理的原始图片

        Returns:
            处理后的图片，已经放在了CUDA上了
        """
        # 创建一个全零的 NumPy 数组作为填充后的图像
        padded_img = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.float32)

        # 计算图片的缩放比例
        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        # 使用 OpenCV 对图片进行缩放
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r))).astype(np.float32)

        # 计算填充后的图像在新尺寸中的位置
        h, w = resized_img.shape[:2]
        hr, wr = (self.input_size[0] - h) // 2, (self.input_size[1] - w) // 2

        # 将缩放后的图像放置在填充后的图像中心
        padded_img[hr: hr + h, wr: wr + w] = resized_img

        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

        # 归一化和减去均值
        padded_img = (padded_img / self.normal - self.rgb_means) / self.std

        # 将维度从 (H, W, C) 转换成 (C, H, W)，并添加一个额外的维度表示 batch size
        padded_img = np.transpose(padded_img, (2, 0, 1))
        padded_img = np.expand_dims(padded_img, axis=0)
        # 转为连续内存
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        # 转换为张量
        images = torch.from_numpy(padded_img)
        images = images.to(self.device)

        return images


if __name__ == '__main__':
    input_size = (640, 640)
    img = cv2.imread(
        r'E:\lg\BaiduSyncdisk\project\AI_project_process\yolox_pytorch\img\img_out\002344348659-90_84-429&369_530&406-525&405_425&398_428&364_528&371-0_0_17_26_30_24_8-105-11.png')
    cv2.imshow('img', img)

