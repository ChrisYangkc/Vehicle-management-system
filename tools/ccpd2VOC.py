import shutil
import cv2
import os

from lxml import etree


class labelimg_Annotations_xml:
    def __init__(self, folder_name, filename, path, database="Unknown"):
        self.root = etree.Element("annotation")
        child1 = etree.SubElement(self.root, "folder")
        child1.text = folder_name
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
        # child3 = etree.SubElement(self.root, "path")
        # child3.text = path
        child4 = etree.SubElement(self.root, "source")
        child5 = etree.SubElement(child4, "database")
        child5.text = database

    def set_size(self, width, height, channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(width)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "channel")
        channeln.text = str(channel)

    def set_segmented(self, seg_data=0):
        segmented = etree.SubElement(self.root, "segmented")
        segmented.text = str(seg_data)

    def set_object(self, label, x_min, y_min, x_max, y_max,
                   pose='Unspecified', truncated=0, difficult=0):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        posen = etree.SubElement(object, "pose")
        posen.text = pose
        truncatedn = etree.SubElement(object, "truncated")
        truncatedn.text = str(truncated)
        difficultn = etree.SubElement(object, "difficult")
        difficultn.text = str(difficult)
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(x_min)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(y_min)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(x_max)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(y_max)

    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')


def translate(path, save_path):
    for filename in os.listdir(path):
        print(filename)

        list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        list2 = filename.split(".", 1)
        subname1 = list2[1]
        if subname1 == 'txt':
            continue
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        print(lx, ly, rx, ry)
        results_xml = [['green', lx, ly, rx, ry]]
        img = cv2.imread(os.path.join(path, filename))
        if img is None:  # 自动删除失效图片（下载过程有的图片会存在无法读取的情况）
            # os.remove(os.path.join(path, filename))
            continue

        height, width, channel = img.shape

        save_xml_name = filename.replace('jpg', 'xml')

        anno = labelimg_Annotations_xml('folder_name', filename + '.jpg', 'path')
        anno.set_size(width, height, channel)
        anno.set_segmented()
        for data in results_xml:
            label, x_min, y_min, x_max, y_max = data
            anno.set_object(label, x_min, y_min, x_max, y_max)
        anno.savefile(os.path.join(save_path, save_xml_name))


if __name__ == '__main__':
    # det图片存储地址
    img_path = r"../ccpd_data"
    # det xml存储地址
    save_path = r"D:\lg\BaiduSyncdisk\project\person_code\project_self\chepai_OCR\11"
    translate(img_path, save_path)
