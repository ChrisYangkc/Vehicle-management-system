import os

# 指定.TXT文件的路径
txt_file_path = r"D:/Bishe_Program/Picture_name/data.txt"
# 指定包含图片的文件夹路径
images_folder_path = r"D:/Bishe_Program/Picture_name/picture"

# 读取.TXT文件
with open(txt_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 遍历每一行
for line in lines:
    # 分割每行内容，获取图片名称和新名字
    parts = line.strip().split(' ')
    if len(parts) >= 3:
        # 提取文件名（去除路径部分）
        image_file_name = os.path.basename(parts[0])
        new_name = parts[1] + '.jpg'  # 新名字，加上.jpg后缀
        # 构建完整的图片文件路径
        image_path = os.path.join(images_folder_path, image_file_name)
        new_image_path = os.path.join(images_folder_path, new_name)

        # 如果文件存在，则重命名
        if os.path.exists(image_path):
            os.rename(image_path, new_image_path)
            print(f"Renamed '{image_path}' to '{new_image_path}'")
        else:
            print(f"File '{image_path}' not found.")
