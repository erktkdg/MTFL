import os

def process_files(folder_path):
    file_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                new_file_name = file.replace(".txt", ".mp4")

                label = "Normal" if "label_A" in file else "Abnormal"
                file_list.append([new_file_name, label])

    with open("/media/DataDrive/yiling/annotation/XD_train_annotation.txt", "w") as f:
        for item in file_list:
            f.write(f"{item[0]}\t{item[1]}\n")

if __name__ == "__main__":
    # 调用函数并传入文件夹路径
    folder_path = "/media/DataDrive2/features/XD_VST_pure/training/L8"
    process_files(folder_path)
