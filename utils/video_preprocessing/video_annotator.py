"""
Annotation Parser Script

This script is used for annotating video files with anomaly information, such as the number of frames containing anomalies
and the start-end frame couples where anomalies occur. It provides a graphical interface for replaying videos, pausing,
marking frames with anomalies, and undoing markings.

Usage:
    python video_annotator.py --root_dir <root_directory> --video_subdir <video_subdirectory> --annotation_file <output_annotation_file> --anomaly_type <anomaly_class_name>

Parameters:
    --root_dir: Path to the root directory containing the video files.
    --video_subdir: Subdirectory within the root directory where video files are located.
    --annotation_file: Path to the output annotation file where annotated information will be saved.
    --anomaly_type: The class name or type of anomalies to be annotated.

Note:
    Please run this in your local environment and then transfer the annotation file to the server for
    further tasks. The server environment may have plugin issues due to the opencv-python.
"""
import cv2
import sys
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Annotation Parser")
    # io
    parser.add_argument('--root_dir', type=str, default="H:\\Projects\\VAD\\Test",
                        help="path to root directory")
    parser.add_argument('--video_subdir', type=str, default="Anomaly_videos",
                        help="path to video subfolder")
    parser.add_argument('--annotation_file', type=str, default="H:\\Projects\\VAD\\Test\\Anomaly_videos.txt",
                        help="the output annotation file")
    parser.add_argument('--anomaly_type', type=str, default='Abnormal', help="The class name of videos.")

    return parser.parse_args()


def get_files(folder_path):
    # Store a list of video files
    video_files = []

    # Traverse the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv')):
                video_files.append(os.path.join(root, file))

    return video_files


if __name__ == "__main__":
    args = get_args()
    # Loop over all files in directory
    video_dir = os.path.join(args.root_dir, args.video_subdir)
    files = get_files(video_dir)

    for filename in files:
        if not filename.endswith(".mp4"):
            continue

        cap = cv2.VideoCapture(filename)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{filename}: {num_frames}")

        if not cap.isOpened():
            print("Error opening video")
            sys.exit(1)

        mem = []
        actions = []

        counter = 0
        max_frame = 0
        marked_frame = 0

        playing = True
        marking = False

        while cap.isOpened():
            if playing:
                if counter == max_frame:
                    ret, frame = cap.read()
                    if ret:
                        if len(mem) == 200:
                            mem.pop(0)
                        mem.append(frame)
                    else:
                        break
                    counter += 1
                    max_frame += 1

                else:
                    frame = mem[counter - max_frame - 1]
                    counter += 1
            else:
                frame = mem[counter - max_frame - 1]
                cv2.putText(frame, f"{len(mem) + counter - max_frame}", (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 255, 255))

            if marking:
                cv2.circle(frame, (int(frame.shape[1]/2), 50), 10, (0, 0, 255), -1)

            cv2.imshow("Replay", frame)
            key = cv2.waitKey(60)
            if key & 0xFF == ord(' '):  # Press space to pause
                playing = not playing
            elif key & 0xFF == ord('a') and not playing:  # a to reverse 10 frames
                if (max_frame - counter) < len(mem) - 11:
                    counter -= 10
            elif key & 0xFF == ord('d') and not playing:  # d to skip 10 frames
                if (max_frame - counter) > 10:
                    counter += 10
            elif key & 0xFF == ord(',') and not playing:  # , to reverse 1 frame
                if (max_frame - counter) < len(mem) - 1:
                    counter -= 1
            elif key & 0xFF == ord('.') and not playing:  # . to skip 1 frame
                if (max_frame - counter) > 0:
                    counter += 1
            elif key & 0xFF == ord('m'):  # m to mark this frame
                if marking:
                    actions.append((marked_frame, counter))
                else:
                    marked_frame = counter
                marking = not marking
                print(f'marking {counter}, waiting for the end frame: {marking}')
            elif key & 0xFF == ord('z'):  # z to undo last marking
                print(f'cancel last marking')
                if not marking:
                    actions.pop(-1)
                marking = not marking
            elif key & 0xFF == ord('s'):  # s to skip this video
                break
            elif key & 0xFF == ord('q'):  # q to quit
                sys.exit(1)

        # the annotation format:
        # relative path to video dir    anomaly type      num_frames     start-end-couples
        num_action = len(actions)
        dir_and_file = os.path.split(filename)
        relative_path = os.path.relpath(filename, args.root_dir)
        line = f"{relative_path} {args.anomaly_type} {num_frames}"
        for i in range(num_action):
            line += f" {actions[i][0]} {actions[i][1]}"
        line += '\n'

        with open(args.annotation_file, "a") as f:
            f.write(line)

        cap.release()
        cv2.destroyWindow("Replay")

    cv2.destroyAllWindows()
