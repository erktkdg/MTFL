import os

if __name__ == "__main__":
    # 读取第一个txt文件的第一列文件名到一个集合中
    file_set1 = set()
    with open("/media/DataDrive/yiling/annotation/XD_test_annotation.txt", "r") as file1:
        for line in file1:
            filename = line.strip().split()[0]  # 假设每行的数据以空格分隔
            filename = os.path.basename(filename)
            file_set1.add(filename)

    # 读取第二个txt文件的第一列文件名到一个集合中
    file_set2 = set()
    with open("/media/DataDrive/yiling/annotation/XD_train_annotation.txt", "r") as file2:
        for line in file2:
            filename = line.strip().split()[0]  # 假设每行的数据以空格分隔
            filename = os.path.basename(filename)
            file_set2.add(filename)

    # 检查两个集合的交集
    common_files = file_set1.intersection(file_set2)
    cnt = len(common_files)
    # 如果交集不为空，则表示两个文件中有相同的文件名
    if common_files:
        print("两个文件中有相同的文件: ", cnt)
    else:
        print("两个文件中没有相同的文件名.")
