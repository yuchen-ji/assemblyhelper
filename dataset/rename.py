import os
import shutil

def rename_and_move_images(input_folder, output_folder):
    # 获取输入文件夹下的所有文件夹
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    for folder in subfolders:
        # 获取文件夹名称，用作新文件名

        # 遍历文件夹中的所有图片文件
        for file in os.scandir(folder):
            if file.is_file() and file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 构建新的文件路径
                new_file_path = os.path.join(output_folder, os.path.basename(folder) + '.png')

                # 复制并重命名文件
                shutil.copy(file.path, new_file_path)

if __name__ == "__main__":
    # 输入文件夹路径，包含小文件夹和图片文件
    input_folder_path = "dataset/sam_output"

    # 输出文件夹路径，用于存放重命名后的文件
    output_folder_path = "dataset/tmp"

    # 创建输出文件夹
    os.makedirs(output_folder_path, exist_ok=True)

    # 执行重命名和移动操作
    rename_and_move_images(input_folder_path, output_folder_path)