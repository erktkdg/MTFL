"""
Unify FPS and Resolution Script

This script processes video files in the input directory, unifying their frame rate
and resolution to match the target specifications (target_fps and target_res).
It then saves the processed videos to the output directory.

Usage:
    python video_format_unifier.py --video_dir <input_directory> --out_dir <output_directory>

Parameters:
    --video_dir: Path to the input directory containing the video files.
    --out_dir: Path to the output directory where processed videos will be saved.

Note:
    Please run this in your local environment and then transfer the processed videos to the server for
    further tasks. The server environment may have plugin issues due to the opencv-python.

UCF format
target_res = (320, 240)
target_fps = 30.0
"""

import cv2
import os
import sys
import shutil
import argparse

# UCF format
target_res = (320, 240)
target_fps = 30.0

# XD format
# target_res = (640, 336)
# target_fps = 24.0


def get_args():
    parser = argparse.ArgumentParser(description="Unify FPS and Resolution Parser")
    # io
    parser.add_argument('--video_dir', type=str, default="/home/yiling/workspace/demo/test_videos/Anonymized_010",
                        help="path to videos")
    parser.add_argument('--out_dir', type=str, default="/home/yiling/workspace/demo/test_videos/Anonymized_010_320x240",
                        help="path to videos")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    video_to_resize = []
    video_fail = []
    counter = 0 # Counter for processing videos

    # Loop through files in the input video directory
    for file in os.listdir(args.video_dir):
        counter += 1
        if not file.endswith('.mp4'):
            continue
        if not os.path.exists(f"{args.out_dir}"):
            os.makedirs(f"{args.out_dir}")

        print(f"Processing video {counter}")

        input_path = os.path.join(args.video_dir, file)
        output_path = os.path.join(args.out_dir, file)
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Check if the video matches the target specifications
        if fps==target_fps and width==target_res[0] and height==target_res[1]:
            print(f"Skip {input_path}")
            shutil.copyfile(input_path, output_path)
            cap.release()
            continue

        video_to_resize.append(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, target_res)

        if not cap.isOpened():
            print(f"Error opening video: {input_path}")
            video_fail.append(f"{input_path}")
            sys.exit(1)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                b = cv2.resize(frame, target_res, interpolation=cv2.INTER_AREA)
                out.write(b)
            else:
                break
        cap.release()
        out.release()

