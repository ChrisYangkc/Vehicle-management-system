import tensorflow as tf
import cv2
import numpy as np
import os

def load_and_preprocess_image(path, target_size=(128, 48)):
    # 读取图像
    image = cv2.imread(path)
    # 裁剪和缩放
    image = cv2.resize(image, target_size)
    # 转换为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用高斯滤波
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # 边缘检测
    image = cv2.Canny(image, 100, 200)
    # 二值化处理
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    # 形态学处理
    kernel = np.ones((3,3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    # 转换为网络输入所需的形状
    image = np.expand_dims(image, axis=-1)
    return image

# 创建字符到索引的映射
# 假设这是您需要识别的中文字符集
chinese_chars = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼'

max_length = 7  # 车牌最大长度
num_characters = 75  # 字符集大小，包括中文、英文字符和数字

# 更新字符映射以包括填充字符
char_to_idx = {char: i for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' + chinese_chars + '-' + '学挂澳使领港临')}
num_characters = len(char_to_idx)

# 使用更新后的映射来转换标签
def label_to_array(label):
    return [char_to_idx[char] for char in label]

# 修改 load_data 函数来处理标签
def load_data(file_path):
    images = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ')
            image_path = os.path.join('E:/licence/CBLPRD-330k_v1', parts[0])
            label = parts[1]  # 确保这是一个字符串
            image = load_and_preprocess_image(image_path)
            images.append(image)
            labels.append(label)
    return np.array(images), labels  # 仅对images使用np.array

def standardize_label(label):
    # 如果label是数字列表，转换为字符列表
    if isinstance(label, list) and all(isinstance(item, int) for item in label):
        label = [str(item) for item in label]

    # 如果label不是字符串，转换为字符串
    if not isinstance(label, str):
        label = ''.join(label)

    # 标准化长度
    if len(label) > max_length:
        label = label[:max_length]  # 截断超出长度的部分
    else:
        label += '-' * (max_length - len(label))  # 使用'-'填充剩余部分

    # 分别对每个字符进行独热编码
    one_hot_label = np.zeros((max_length, num_characters))
    for i, char in enumerate(label):
        index = char_to_idx[char]
        one_hot_label[i, index] = 1
    return one_hot_label

# 在转换标签为独热编码之前，确保所有标签都是字符串
train_images, train_labels = load_data('E:/licence/CBLPRD-330k_v1/train.txt')
train_labels = np.array([standardize_label(label) for label in train_labels])

val_images, val_labels = load_data('E:/licence/CBLPRD-330k_v1/val.txt')
val_labels = np.array([standardize_label(label) for label in val_labels])

model = tf.keras.models.Sequential([
    # 第一层卷积层
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # 第二层卷积层
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 将卷积层的输出展平
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(max_length * num_characters, activation='softmax'),
    tf.keras.layers.Reshape((max_length, num_characters))  # 添加Reshape层
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=10)

# 评估模型
val_loss, val_accuracy = model.evaluate(val_images, val_labels)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_accuracy)
