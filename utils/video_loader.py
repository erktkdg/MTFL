import logging
import os
import sys

import numpy as np
from torch.utils import data
from torchvision.datasets.video_utils import VideoClips


class VideoIter(data.Dataset):
    def __init__(self,
                 clip_length,
                 frame_stride,
                 proc_video=None,
                 dataset_path=None,
                 video_transform=None,
                 use_splits=False,
                 return_label=False,):
        super().__init__()
        # video clip properties
        self.frames_stride = frame_stride
        self.total_clip_length_in_frames = clip_length * frame_stride
        self.video_transform = video_transform

        # IO
        self.dataset_path = dataset_path
        self.video_list = self._get_video_list(dataset_path=self.dataset_path, proc_video=proc_video, use_splits=use_splits)
        self.return_label = return_label

        # data loading
        self.video_clips = VideoClips(video_paths=self.video_list,
                                      clip_length_in_frames=self.total_clip_length_in_frames,
                                      frames_between_clips=int(self.total_clip_length_in_frames), # between start frame
                                      )

    @property
    def video_count(self):
        return len(self.video_list)

    def getitem_from_raw_video(self, idx):
        """
        Load and preprocess a video clip from the dataset.

        Args:
            idx (int): Index of the video clip to load.

        Returns:
            tuple: A tuple containing the loaded and preprocessed video clip, clip index, directory, and filename.
        """
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        in_clip_frames = list(range(0, self.total_clip_length_in_frames, self.frames_stride))
        video = video[in_clip_frames]
        if self.video_transform is not None:
            video = self.video_transform(video)

        dir, file = video_path.split(os.sep)[-2:]
        file = file.split('.')[0]

        if self.return_label:
            label = 0 if "Normal" in video_path else 1
            return video, label, clip_idx, dir, file

        return video, clip_idx, dir, file

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, index):
        succ = False
        while not succ:
            try:
                batch = self.getitem_from_raw_video(index)
                succ = True
            except Exception as e:
                index = np.random.choice(range(0, self.__len__()))
                trace_back = sys.exc_info()[2]
                line = trace_back.tb_lineno
                logging.warning(f"VideoIter:: ERROR (line number {line}) !! (Force using another index:\n{index})\n{e}")

        return batch


    def _get_video_list(self, dataset_path, proc_video, use_splits):
        """
        Get a list of video files from the dataset path, considering the specified processing options.

        Args:
            dataset_path (str): Path to the dataset directory.
            proc_video (list or None): List of processed video filenames or None.
            use_splits (bool): Whether to use splits for data selection.

        Returns:
            list: List of video file paths.
        """
        assert os.path.exists(dataset_path), "VideoIter:: failed to locate: `{}'".format(dataset_path)
        with open("/media/DataDrive/yiling/annotation/recognition/splits/VAD/splits_video_list.txt", "r") as file:
            splits_files = [line.strip() for line in file.readlines()]
        vid_list = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                file_path = os.path.join(path, name)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                relative_path = os.path.join(parent_dir, name)
                if use_splits and relative_path not in splits_files:
                    continue
                if ('mp4' not in name and 'avi' not in name) or relative_path.split(".")[0] in proc_video:
                    continue
                vid_list.append(os.path.join(path, name))

        logging.info(f"Found {len(vid_list)} unprocessed video files in {dataset_path}")

        return vid_list

