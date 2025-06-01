import os
import subprocess


def process_txt_file(txt_file_path, root_dir, output_txt_path):
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()

    updated_lines = []
    label = 'Abnormal'
    for line in lines:
        items = line.split()
        file_name = items[0]
        file_path = find_file(root_dir, file_name)
        if file_path:
            frame_count = count_frames(file_path)
            updated_lines.append(f"{os.path.basename(file_path)} {label} {frame_count} {' '.join(items[1:])}\n")

    with open(output_txt_path, "w") as f:
        f.writelines(updated_lines)


def find_file(root_dir, file_name):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file_name in file:
                return os.path.join(root, file)
    return None


def count_frames(video_file_path):
    # 使用FFmpeg获取视频帧数
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of",
               "default=nokey=1:noprint_wrappers=1", video_file_path]
    output = subprocess.check_output(command)
    return int(output)


if __name__ == "__main__":
    txt_file_path = "/media/DataDrive/yiling/annotation/XD_annotations.txt"
    output_txt_path = "/media/DataDrive/yiling/annotation/XD_test_annotation.txt"
    root_dir = "/media/DataDrive2/dataset/XD"

    process_txt_file(txt_file_path, root_dir, output_txt_path)
