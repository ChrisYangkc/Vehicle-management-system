import os
import xml.etree.ElementTree as ET

from utils.utils import get_classes


def convert_annotation(xml_path, classes):
    # 打开指定路径的 XML 文件并解析其内容
    in_file = open(xml_path, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    # 定义一个空列表用于存储所有检测到的边界框信息
    lst_result = []
    # 遍历 XML 文件中的每个对象
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        # 获取对象的类别标签
        cls = obj.find('name').text
        # 如果对象的类别不在指定的类别列表中或者对象被标记为 "difficult"，则跳过该对象
        if cls not in classes or int(difficult) == 1:
            continue
        # 获取对象的类别 ID
        cls_id = classes.index(cls)
        # 获取对象的边界框坐标信息
        xmlbox = obj.find('bndbox')
        b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text))]
        # 将当前对象的边界框信息、类别 ID 和类别标签添加到结果列表中
        lst_result.append([b, cls_id, cls])
    # 返回所有检测到的边界框信息
    return lst_result


def write_txt(img_path, ann_path, classes_path, txt_path):
    classes, _ = get_classes(classes_path)
    dic_class = {}
    file_txt = open(txt_path, 'w')
    for root, dirs, files in os.walk(ann_path):
        for file in files:
            if not file.endswith('.xml'):
                continue
            print(file)
            filename, ext = os.path.splitext(file)
            xml_path = os.path.join(root, file)
            jpg_path = os.path.join(img_path, filename + '.jpg')
            if not os.path.exists(jpg_path):
                print('不存在图片：', jpg_path)
                continue
            lst_result = convert_annotation(xml_path, classes)
            for result in lst_result:
                # xmin, ymin, xmax, ymax = result[0]
                # 每类的索引
                cls_id = result[1]
                # label名
                cls = result[2]
                file_txt.write(jpg_path + " " + ",".join(str(x) for x in result[0]) + ',' + str(cls_id))
                file_txt.write('\n')

                if cls not in dic_class.keys():
                    dic_class[cls] = 1
                else:
                    dic_class[cls] += 1

    print('每类的数据量为：', dic_class)


if __name__ == "__main__":
    # 训练集
    img_path_train = r"D:/Bishe_Program/Licence_plate_recognition_model_training/traindata/train/JPEGImages" 
    ann_path_train = r"D:/Bishe_Program/Licence_plate_recognition_model_training/traindata/train/Annotations"
    train_txt_path = r"D:/Bishe_Program/Licence_plate_recognition_model_training/VOCdevkit/train.txt"
    # 测试集
    img_path_val = r"D:/Bishe_Program/Licence_plate_recognition_model_training/traindata/test/JPEGImages" 
    ann_path_val = r"D:/Bishe_Program/Licence_plate_recognition_model_training/traindata/test/Annotations"
    val_txt_path = r"D:/Bishe_Program/Licence_plate_recognition_model_training/VOCdevkit/val.txt"
    # 类别文件
    classes_path = r"D:/Bishe_Program/Licence_plate_recognition_model_training/model_data/self_classes.txt"
    # 制作train txt
    write_txt(img_path_train, ann_path_train, classes_path, train_txt_path)

    # 制作val txt
    write_txt(img_path_val, ann_path_val, classes_path, val_txt_path)
