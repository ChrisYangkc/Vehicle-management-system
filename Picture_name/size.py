from PIL import Image
import os

def resize_images_in_folder(folder_path, size=(94, 24)):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(folder_path, file_name)
            with Image.open(file_path) as img:
                img = img.resize(size, Image.ANTIALIAS)
                img.save(file_path)
                print(f"Resized '{file_name}' to {size}")

# 指定图片所在的文件夹路径
folder_path = r"E:/毕设论文/毕设材料/训练数据集/CBLPRD-330k_v1/CBLPRD-330k"
resize_images_in_folder(folder_path)
