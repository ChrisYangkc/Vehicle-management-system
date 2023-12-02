import os
import random
import shutil
import xml.etree.ElementTree as ET


########################################################################
# 只需修改voc_ann_path、voc_img_path、yolo_out_path三个路径，自动生成为yolo
# 训练格式的数据集
########################################################################
# VOC数据集路径
voc_ann_path = r"D:\lg\BaiduSyncdisk\project\person_code\project_self\chepai_OCR\data\traindata\VOC_LP_det\Annotations"
voc_img_path = r"D:\lg\BaiduSyncdisk\project\person_code\project_self\chepai_OCR\data\traindata\VOC_LP_det\JPEGImages"
# YOLO数据集路径
yolo_out_path = r"D:\lg\BaiduSyncdisk\project\person_code\project_self\chepai_OCR\data\traindata\yolo"

train_img_path = os.path.join(yolo_out_path, 'train', 'images', 'train')
train_labels_path = train_img_path.replace('images', 'labels')
test_img_path = os.path.join(yolo_out_path, 'test', 'images', 'test')
test_labels_path = test_img_path.replace('images', 'labels')
val_img_path = os.path.join(yolo_out_path, 'val', 'images', 'val')
val_labels_path = val_img_path.replace('images', 'labels')

# VOC类别名称和对应的编号
classes = {"blue": 0, "green": 1}  # 根据实际情况修改

# 随机划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 获取所有文件名
all_files = os.listdir(voc_ann_path)
# 随机打乱文件顺序
random.shuffle(all_files)

# 计算划分的索引位置
train_index = int(len(all_files) * train_ratio)
val_index = train_index + int(len(all_files) * val_ratio)

# 创建保存txt文件的文件夹
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# 创建保存图片的文件夹
os.makedirs(train_img_path, exist_ok=True)
os.makedirs(test_img_path, exist_ok=True)
os.makedirs(val_img_path, exist_ok=True)

# 遍历VOC数据集文件夹
for i, filename in enumerate(all_files):
    # 解析XML文件
    tree = ET.parse(os.path.join(voc_ann_path, filename))
    root = tree.getroot()
    # 获取图片尺寸
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    # 创建YOLO标注文件
    yolo_filename = filename.replace(".xml", ".txt")
    if i < train_index:
        yolo_file = open(os.path.join(train_labels_path, yolo_filename), "w")
        img_dest_path = train_img_path
    elif i < val_index:
        yolo_file = open(os.path.join(val_labels_path, yolo_filename), "w")
        img_dest_path = val_img_path
    else:
        yolo_file = open(os.path.join(test_labels_path, yolo_filename), "w")
        img_dest_path = test_img_path

    # 遍历XML文件中的所有目标
    for obj in root.findall("object"):
        # 获取目标类别名称和边界框坐标
        name = obj.find("name").text
        xmin = int(obj.find("bndbox").find("xmin").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        # 计算边界框中心点坐标和宽高
        x = (xmin + xmax) / 2 / width
        y = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        # 将目标写入YOLO标注文件
        class_id = classes[name]
        yolo_file.write(f"{class_id} {x} {y} {w} {h}\n")

    yolo_file.close()
    # 复制图片到对应的划分文件夹
    img_filename = filename.replace(".xml", ".jpg")
    img_src_path = os.path.join(voc_img_path, img_filename)
    shutil.copy(img_src_path, img_dest_path)
