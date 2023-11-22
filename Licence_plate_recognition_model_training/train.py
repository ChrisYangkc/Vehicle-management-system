import os
import numpy as np
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Reshape
from tensorflow.keras.utils import to_categorical

# 图像预处理参数
IMG_SIZE = (32, 32)

# 省份和字母数字映射字典
province_dict = {
    0: "皖", 1: "沪", 2: "津", 3: "渝", 4: "冀", 5: "晋", 6: "蒙",
    7: "辽", 8: "吉", 9: "黑", 10: "苏", 11: "浙", 12: "京", 13: "闽",
    14: "赣", 15: "鲁", 16: "豫", 17: "鄂", 18: "湘", 19: "粤", 20: "桂",
    21: "琼", 22: "川", 23: "贵", 24: "云", 25: "藏", 26: "陕", 27: "甘",
    28: "青", 29: "宁", 30: "新"
}

ads_dict = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "J", 9: "K", 10: "L", 11: "M", 12: "N", 13: "P", 14: "Q", 15: "R",
    16: "S", 17: "T", 18: "U", 19: "V", 20: "W", 21: "X", 22: "Y", 23: "Z",
    24: "0", 25: "1", 26: "2", 27: "3", 28: "4", 29: "5", 30: "6", 31: "7",
    32: "8", 33: "9"
}

# 图像预处理函数
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def parse_filename(filename):
    parts = filename.split('-')
    # 提取倾斜角度、边界框和车牌号码
    angles = parts[1].split('_')
    bbox_coords = parts[2].split('_')
    plate_coords = parts[3].split('_')
    license_plate_numbers = [int(n) for n in parts[4].split('_')]

    # 将提取的数据转换为需要的格式
    bbox = [int(x) for x in bbox_coords[0].split('&')] + [int(x) for x in bbox_coords[1].split('&')]
    plate_points = [tuple(map(int, coord.split('&'))) for coord in plate_coords]

    return angles, bbox, plate_points, license_plate_numbers

def crop_and_resize(image, size=IMG_SIZE):
    resized_image = cv2.resize(image, size)
    return resized_image

def complete_preprocess(image_path):
    image = load_image(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    processed_image = crop_and_resize(gray_image)
    return processed_image

# 标签编码函数
def encode_label(label):
    encoded = []
    # 省份
    encoded.append(label[0])
    # 字母和数字
    for num in label[1:]:
        encoded.append(num)
    return encoded

# 数据集加载和处理
dataset_dir = "E:/licence/ccpd_base"  # 替换为实际路径
images = []
labels = []

for filename in os.listdir(dataset_dir):
    filepath = os.path.join(dataset_dir, filename)
    image = complete_preprocess(filepath)
    angles, bbox, plate_points, license_plate_numbers = parse_filename(filename)
    encoded_label = encode_label(license_plate_numbers)
    
    images.append(image)
    labels.append(encoded_label)

images = np.array(images).reshape(-1, 32, 32, 1)  # 重塑为适合CNN的格式
labels = np.array(labels)

# 独热编码
num_classes = 34
encoded_labels = [to_categorical(label, num_classes=num_classes) for label in labels]
encoded_labels = np.array(encoded_labels)

# 随机选取三张图片作为测试样本
test_sample_size = 3
test_indices = random.sample(range(len(images)), test_sample_size)

test_images = np.array([images[i] for i in test_indices]).reshape(-1, 32, 32, 1)
test_labels = np.array([encoded_labels[i] for i in test_indices])

# 移除选中的测试样本，剩下的作为训练集
train_images = np.delete(images, test_indices, axis=0)
train_labels = np.delete(encoded_labels, test_indices, axis=0)

# CNN模型构建
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(32, 32, 1)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes * 7),  # 假设车牌有7个字符
    Reshape((7, num_classes)),
    Activation('softmax')
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, batch_size=32, epochs=100, validation_split=0.1)

# 测试模型并输出结果
test_predictions = model.predict(test_images)

for i in range(test_sample_size):
    print("真实标签:", np.argmax(test_labels[i], axis=1))
    print("预测标签:", np.argmax(test_predictions[i], axis=1))
