import cv2
import os
import numpy as np

# 定义输入和输出文件夹
input_folder = 'camera'
output_folder = 'processed_images'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def process_image(image_path, output_path):
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 边缘检测 - 使用Canny和Sobel算子
    edges_canny = cv2.Canny(binary, 100, 200)
    sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = cv2.magnitude(sobelx, sobely)
    edges = cv2.bitwise_or(edges_canny, np.uint8(edges_sobel))

    # 图像增强 - 直方图均衡化和高斯模糊
    equalized = cv2.equalizeHist(np.uint8(edges))
    enhanced = cv2.GaussianBlur(equalized, (5, 5), 0)

    # 背景消除（示例中暂不实现具体的背景消除方法）
    # 通常背景消除需要特定的算法或条件，这里留作未实现
    # background_removed = ...

    # 保存处理后的图像
    cv2.imwrite(output_path, enhanced)

# 遍历文件夹中的所有图像
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') and ('camera_0' in filename or 'camera_1' in filename):
        input_path = os.path.join(input_folder, filename)
        # 输出文件名加上 "afterp" 标签
        output_path = os.path.join(output_folder, filename.replace('.jpg', '_afterp.jpg'))
        process_image(input_path, output_path)
        print(f"Processed and saved: {output_path}")
