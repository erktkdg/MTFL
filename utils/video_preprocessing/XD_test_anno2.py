import os

def count_frames(video_path):
    # 使用FFmpeg获取视频帧数
    command = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 {video_path}"
    frame_count = os.popen(command).read().strip()
    return frame_count

def update_txt_file(txt_file_path, normal_test, root_dir):
    with open(txt_file_path, "r") as f:
        lines = f.readlines()

    list_files = []
    for line in lines:
        file_name = line.split()[0]
        list_files.append(file_name)

    label = 'Normal'
    with open(normal_test, "a") as f:
        for root, _, files in os.walk(root_dir):
            for filename in files:
                video_path = os.path.join(root, filename)
                if filename.endswith(".mp4") and filename not in list_files:
                    frame_count = count_frames(video_path)
                    f.write(f"{filename} {label} {frame_count}\n")

if __name__ == "__main__":
    root_dir = "/media/DataDrive2/dataset/XD/test_videos_2"
    txt_file_path = "/media/DataDrive/yiling/annotation/XD_test_annotation.txt"
    normal_test = "/media/DataDrive/yiling/annotation/XD_test_normal.txt"

    update_txt_file(txt_file_path, normal_test, root_dir)
