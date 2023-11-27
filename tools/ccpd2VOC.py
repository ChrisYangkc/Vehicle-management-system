import shutil
import cv2
import os

from lxml import etree

# 定义labelimg_Annotations_xml类，用于创建XML文件结构
class labelimg_Annotations_xml:
    # 构造函数，初始化XML根节点
    def __init__(self, folder_name, filename, path, database="Unknown"):
        self.root = etree.Element("annotation")  # 创建annotation标签
        child1 = etree.SubElement(self.root, "folder")  # 创建folder子标签
        child1.text = folder_name  # 设置folder标签的文本
        child2 = etree.SubElement(self.root, "filename")  # 创建filename子标签
        child2.text = filename  # 设置filename标签的文本
        # 注释掉path标签的创建，因为在此例中未使用
        child4 = etree.SubElement(self.root, "source")  # 创建source子标签
        child5 = etree.SubElement(child4, "database")  # 创建database子标签
        child5.text = database  # 设置database标签的文本

    # 设置图像的尺寸信息
    def set_size(self, width, height, channel):
        size = etree.SubElement(self.root, "size")  # 创建size标签
        widthn = etree.SubElement(size, "width")  # 创建width子标签
        widthn.text = str(width)  # 设置width的文本
        heightn = etree.SubElement(size, "height")  # 创建height子标签
        heightn.text = str(height)  # 设置height的文本
        channeln = etree.SubElement(size, "channel")  # 创建channel子标签
        channeln.text = str(channel)  # 设置channel的文本

    # 设置是否分割的标记
    def set_segmented(self, seg_data=0):
        segmented = etree.SubElement(self.root, "segmented")  # 创建segmented标签
        segmented.text = str(seg_data)  # 设置segmented的文本

    # 设置目标物体的详细信息
    def set_object(self, label, x_min, y_min, x_max, y_max,
                   pose='Unspecified', truncated=0, difficult=0):
        object = etree.SubElement(self.root, "object")  # 创建object标签
        namen = etree.SubElement(object, "name")  # 创建name子标签
        namen.text = label  # 设置name的文本
        posen = etree.SubElement(object, "pose")  # 创建pose子标签
        posen.text = pose  # 设置pose的文本
        truncatedn = etree.SubElement(object, "truncated")  # 创建truncated子标签
        truncatedn.text = str(truncated)  # 设置truncated的文本
        difficultn = etree.SubElement(object, "difficult")  # 创建difficult子标签
        difficultn.text = str(difficult)  # 设置difficult的文本
        bndbox = etree.SubElement(object, "bndbox")  # 创建bndbox标签
        xminn = etree.SubElement(bndbox, "xmin")  # 创建xmin子标签
        xminn.text = str(x_min)  # 设置xmin的文本
        yminn = etree.SubElement(bndbox, "ymin")  # 创建ymin子标签
        yminn.text = str(y_min)  # 设置ymin的文本
        xmaxn = etree.SubElement(bndbox, "xmax")  # 创建xmax子标签
        xmaxn.text = str(x_max)  # 设置xmax的文本
        ymaxn = etree.SubElement(bndbox, "ymax")  # 创建ymax子标签
        ymaxn.text = str(y_max)  # 设置ymax的文本

     # 保存XML文件
    def savefile(self, filename):
        tree = etree.ElementTree(self.root)  # 创建ElementTree对象
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')  # 将XML写入文件

def translate(path, save_path):
    for filename in os.listdir(path):  # 遍历指定路径下的所有文件
        print(filename)  # 打印当前处理的文件名

        # 使用字符串分割方法来解析文件名。文件名被假设为包含多个用'-'分隔的部分。
        list1 = filename.split("-", 3)  # 将文件名按'-'分割，最多分割成4部分
        subname = list1[2]  # 获取分割后的第三部分，预期是包含坐标信息的部分

        list2 = filename.split(".", 1)  # 将文件名按'.'分割，分成两部分
        subname1 = list2[1]  # 获取文件的扩展名
        if subname1 == 'txt':  # 如果扩展名是txt，则跳过当前循环
            continue

        lt, rb = subname.split("_", 1)  # 从文件名的一部分中进一步分割以获取坐标，预期格式为"lx&ly_rx&ry"
        lx, ly = lt.split("&", 1)  # 分割左上角坐标，格式为"lx&ly"
        rx, ry = rb.split("&", 1)  # 分割右下角坐标，格式为"rx&ry"
        print(lx, ly, rx, ry)  # 打印坐标信息

        results_xml = [['green', lx, ly, rx, ry]]  # 创建一个包含标签和坐标信息的列表，用于后续生成XML

        img = cv2.imread(os.path.join(path, filename))  # 使用OpenCV读取图片文件
        if img is None:  # 如果img为空，说明图片文件无效或不存在，则跳过当前循环
            continue


        height, width, channel = img.shape  # 获取图片的尺寸和通道信息

        save_xml_name = filename.replace('jpg', 'xml')  # 更改文件扩展名为xml

        anno = labelimg_Annotations_xml('folder_name', filename + '.jpg', 'path')  # 创建XML标注对象
        anno.set_size(width, height, channel)  # 设置图片尺寸
        anno.set_segmented()  # 设置segmented标记
        for data in results_xml:
            label, x_min, y_min, x_max, y_max = data
            anno.set_object(label, x_min, y_min, x_max, y_max)  # 添加目标物体信息
        anno.savefile(os.path.join(save_path, save_xml_name))  # 保存XML文件


if __name__ == '__main__':
    # 指定图片和XML文件的存储路径
    img_path = r"D:/Bishe_Program/Licence_plate_recognition_model_training/traindata/train/JPEGImages"  # 设置源图片的路径。这里使用的是相对路径，指向当前脚本所在目录的上级目录下的ccpd_data文件夹。
    save_path = r"D:/Bishe_Program/Licence_plate_recognition_model_training/traindata/train/Annotations"  # 设置XML文件的保存路径。这是一个绝对路径，指向一个具体的目录。
    translate(img_path, save_path)  # 调用translate函数，传入之前定义的源图片路径和XML保存路径。