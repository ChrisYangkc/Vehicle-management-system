import os
import random
import shutil

def move_random_images(source_folder, target_folder, percentage=0.05):
    # 获取源文件夹中所有文件的列表
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # 筛选出图片文件（根据需要调整扩展名）
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # 计算要移动的图片数量
    num_images_to_move = int(len(images) * percentage)
    
    # 随机选择图片
    selected_images = random.sample(images, num_images_to_move)
    
    # 移动选中的图片
    for image in selected_images:
        shutil.move(os.path.join(source_folder, image), os.path.join(target_folder, image))

# 示例使用
source_folder = r"D:/Bishe_Program/Licence_plate_detection_model/traindata/test" # 源文件夹路径
target_folder = r"D:/Bishe_Program/Licence_plate_detection_model/traindata/val"  # 目标文件夹路径

move_random_images(source_folder, target_folder)
