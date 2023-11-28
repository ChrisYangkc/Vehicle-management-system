import os

def rename_files_in_folder(folder_path):
    """
    Rename all .jpg files in the specified folder by appending '-01' to their filename.

    :param folder_path: Path to the folder containing .jpg files.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # Construct the new filename
            new_filename = filename[:-4] + "-02.jpg"
            # Construct the full old and new file paths
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed {filename} to {new_filename}")

# 示例用法，将以下路径替换为您的文件夹路径
folder_path = r"C:/Users/yangk/Desktop/Picture_tag/green"
rename_files_in_folder(folder_path)
