import os

def add_dir(txt_path, output_path, parent_dir):
    with open(txt_path, "r") as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        # 拆分每行的数据
        columns = line.strip().split("\t")  # 假设列之间用制表符分隔，你可以根据实际情况修改分隔符
        if columns:  # 检查是否有数据
            # 更新第一列
            columns[0] = os.path.join(parent_dir, columns[0])
            # 重新组合每行数据
            updated_line = "\t".join(columns) + "\n"  # 同样，使用相同的分隔符重新组合数据
            updated_lines.append(updated_line)

    with open(output_path, "w") as file:
        file.writelines(updated_lines)

if __name__ == "__main__":
    txt_path = "/media/DataDrive/yiling/annotation/XD_train_annotation_file.txt"
    output_path = "/media/DataDrive/yiling/annotation/XD_train_annotation.txt"

    add_dir(txt_path, output_path, 'train')

