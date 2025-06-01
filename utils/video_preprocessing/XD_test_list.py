import os

def get_filenames_from_txt(txt_path):
    filenames = set()
    with open(txt_path, "r") as f:
        for line in f:
            file, ext = os.path.splitext(line.strip())
            filenames.add(file)
    return filenames

def find_difference(txt1_path, txt2_path):
    filenames1 = get_filenames_from_txt(txt1_path)
    filenames2 = get_filenames_from_txt(txt2_path)
    difference1 = filenames1.difference(filenames2)
    difference2 = filenames2.difference(filenames1)
    return difference1, difference2

def write_to_txt(filename, difference):
    with open(filename, "w") as f:
        for item in difference:
            f.write("%s\n" % item)

if __name__ == "__main__":
    txt1_path = "/media/DataDrive/yiling/annotation/XD_test_annotation.txt"
    txt2_path = "/media/DataDrive/yiling/annotation/XD_feature_list.txt"
    difference1, difference2 = find_difference(txt1_path, txt2_path)

    output_txt1 = "/media/DataDrive/yiling/annotation/difference1.txt"
    output_txt2 = "/media/DataDrive/yiling/annotation/difference2.txt"

    write_to_txt(output_txt1, difference1)
    write_to_txt(output_txt2, difference2)

    print("txt1中不同的部分已存储到:", output_txt1)
    print("txt2中不同的部分已存储到:", output_txt2)
