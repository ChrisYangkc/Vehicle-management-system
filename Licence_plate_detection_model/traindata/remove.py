import os

# 指定.TXT文件的路径
txt_file_path = r"D:/Bishe_Program/Licence_plate_detection_model/traindata/data.txt"
# 指定包含图片的文件夹路径
images_folder_path = r"D:/Bishe_Program/Licence_plate_detection_model/traindata/blue"

# 设置要删除的标记
delete_tags = ["双层黄牌", "新能源大型车", "拖拉机绿牌"]

# 要删除的图片列表
images_to_delete = []

# 读取.TXT文件
with open(txt_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 遍历每一行来找到带有特定标记的图片
# 遍历每一行
for line in lines:
    # 移除每行的前后空格，并以空格分割，分割后的结果存入parts列表
    parts = line.strip().split(' ')

    # 检查parts列表的长度是否至少为3
    # 这是为了确保列表包含文件路径、车牌号和车牌类型
    if len(parts) >= 3:
        # 获取车牌号部分（假设车牌号总是在第二个位置，即索引为1）
        plate_number = parts[1]

        # 检查车牌类型是否在待删除标签列表中
        if any(tag in parts for tag in delete_tags):
            # 构造文件名，假设所有图片都是.jpg格式
            image_name = plate_number + '.jpg'

            # 将待删除的图片文件名添加到images_to_delete列表中
            images_to_delete.append(image_name)


# 删除标记的图片
for image_name in images_to_delete:
    image_path = os.path.join(images_folder_path, image_name)
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted '{image_path}'")
    else:
        print(f"File '{image_path}' not found.")

