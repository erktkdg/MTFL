import os
import shutil

def find_missing_files(folder_a, folder_b):
    mp4_files_a = set()
    txt_files_b = set()

    # 获取文件夹 A 中的 .mp4 文件名
    for root, _, files in os.walk(folder_a):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files_a.add(os.path.splitext(file)[0])

    # 获取文件夹 B 中的 .txt 文件名
    for root, _, files in os.walk(folder_b):
        for file in files:
            if file.endswith(".txt"):
                txt_files_b.add(os.path.splitext(file)[0])

    # 查找文件夹 A 中有但文件夹 B 中没有的 .mp4 文件
    missing_files = mp4_files_a - txt_files_b

    return missing_files

def copy_files(missing_files, source_folder, destination_folder):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 复制文件到目标文件夹
    for file_name in missing_files:
        source_file = os.path.join(source_folder, file_name + ".mp4")
        destination_file = os.path.join(destination_folder, file_name + ".mp4")
        shutil.copy(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")

if __name__ == "__main__":
    folder_a = "/media/DataDrive2/dataset/XD/2372-2804"  # 指定文件夹 A 的路径
    folder_b = "/media/DataDrive2/features/XD_parts/XD_VST_pure/L64/2372-2804"  # 指定文件夹 B 的路径
    destination_folder = "/media/DataDrive2/dataset/XD/unprocessed/2372-2804"  # 指定目标文件夹的路径

    missing_mp4_files = find_missing_files(folder_a, folder_b)

    print("Files in folder A but not in folder B:")
    for mp4_file in missing_mp4_files:
        print(mp4_file + ".mp4")

    missing_mp4_files = find_missing_files(folder_a, folder_b)

    print("Files in folder A but not in folder B:")
    for mp4_file in missing_mp4_files:
        print(mp4_file + ".mp4")

    copy_files(missing_mp4_files, folder_a, destination_folder)
