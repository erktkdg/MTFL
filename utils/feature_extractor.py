"""Reference with Ivo's implementation"""
import argparse
import logging
import os
from os import path, mkdir
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from video_loader import VideoIter
from utils import register_logger, get_torch_device
import transforms_video
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# Video Swin Transformer related repository
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import warnings

warnings.filterwarnings("ignore", message="The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")
warnings.filterwarnings('ignore', message='No handlers found: "aten::pad". Skipped.')


def get_args():
    parser = argparse.ArgumentParser(description="VST Feature Extractor Parser")
    # I/O
    parser.add_argument('--dataset_path', default='test_videos',
                        help="path to dataset")
    parser.add_argument('--save_dir', type=str, default="features",
                        help="set output root for the features.")
    # extraction params
    parser.add_argument('--model_type', default='swinB',
                        type=str,
                        help="type of feature extractor")
    parser.add_argument('--pretrained_3d',
                        default='/media/DataDrive/yiling/models/VST_finetune/hflip_speed_120_2d/best_top1_acc_epoch_15.pth',
                        type=str,
                        help="load default 3D pretrained feature extractor model.")
    parser.add_argument('--clip_length', type=int, default=8,
                        help="define the length of each input sample.")
    parser.add_argument('--frame_interval', type=int, default=1,
                        help="define the sampling interval between frames.")
    parser.add_argument('--use_splits', type=bool, default=False,
                        help="use full anomalous data or splits, only applicable of Split Dataset of UCF-CRIME and VAD")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    # running cfg
    parser.add_argument('--num_workers', type=int, default=0,
                        help="define the number of workers used for loading the videos")
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--log_every', type=int, default=10,
                        help="log the writing of clips every n steps.")
    parser.add_argument('--log_file', type=str,
                        help="set logging file.")
    parser.add_argument('--gpu', type=int, default=0, help="gpu id")

    return parser.parse_args()


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def to_segments(data, num=32):
    """
	These code is taken from:
	https://github.com/rajanjitenpatel/C3D_feature_extraction/blob/b5894fa06d43aa62b3b64e85b07feb0853e7011a/extract_C3D_feature.py#L805
	:param data: list of features of a certain video
	:return: list of 32 segments
	"""
    data = np.array(data)
    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=num + 1)).astype(int)
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)
        if np.linalg.norm == 0:
            logging.error("Feature norm is 0")
            exit()
        if len(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features


class FeaturesWriter:
    def __init__(self, num_videos, chunk_size=16):
        """
        Initialize a FeaturesWriter instance.

        Args:
            num_videos (int): Total number of videos to process.
            chunk_size (int, optional): Chunk size for writing features, and not used. Defaults to 16.
        """
        self.path = None
        self.dir = None
        self.data = None
        self.chunk_size = chunk_size
        self.num_videos = num_videos
        self.dump_count = 0

    def _init_video(self, video_name, dir):
        self.path = path.join(dir, f"{video_name}.txt")
        self.dir = dir
        self.data = dict()

    def has_video(self):
        return self.data is not None

    def dump(self):
        logging.info(f'{self.dump_count} / {self.num_videos}:	Dumping {self.path}')
        self.dump_count += 1
        if not path.exists(self.dir):
            os.mkdir(self.dir)
        features = to_segments([self.data[key] for key in sorted(self.data)])
        with open(self.path, 'w') as fp:
            for d in features:
                d = [str(x) for x in d]
                fp.write(' '.join(d) + '\n')

    def _is_new_video(self, video_name, dir):
        new_path = path.join(dir, f"{video_name}.txt")
        if self.path != new_path and self.path is not None:
            return True

        return False

    def store(self, feature, idx):
        self.data[idx] = list(feature)

    def write(self, feature, video_name, idx, dir):
        if not self.has_video():
            self._init_video(video_name, dir)

        if self._is_new_video(video_name, dir):
            self.dump()
            self._init_video(video_name, dir)

        self.store(feature, idx)


def get_features_loader(dataset_path, clip_length, frame_interval, batch_size, num_workers, save_dir, use_splits):
    """
    Get the data loader for extracting video features.

    Args:
        dataset_path (str): Path to the videos.
        clip_length (int): Length of each input sample.
        frame_interval (int): Sampling interval between frames.
        batch_size (int): Batch size.
        num_workers (int): Number of workers used for loading videos.
        save_dir (str): Directory to save features.
        use_splits (bool): Whether to use full anomalous data or splits.

    Returns:
        data_loader (VideoIter): Video data loader.
        data_iter (DataLoader): Torch data loader for video features extraction.
    """
    mean = [0.400, 0.388, 0.372]  # VAD mean and std in RGB
    std = [0.247, 0.245, 0.243]
    size = 224
    resize = size, size
    crop = size

    res = transforms.Compose([
        transforms_video.ToTensorVideo(),
        transforms_video.ResizeVideo(resize),
        transforms_video.CenterCropVideo(crop),
        transforms_video.NormalizeVideo(mean=mean, std=std)
    ])

    if os.path.exists(save_dir):
        proc_v = []
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, save_dir)
                proc_v.append(relative_path)
        proc_v = [v.split(".")[0] for v in proc_v]
        if len(proc_v) > 0:
            logging.info(
                f"[Data] Already {len(proc_v)} files have been processed"
            )

    data_loader = VideoIter(
        dataset_path=dataset_path,
        proc_video=proc_v,
        clip_length=clip_length,
        frame_stride=frame_interval,
        video_transform=res,
        use_splits=use_splits,
        return_label=False,
    )

    data_iter = torch.utils.data.DataLoader(
        data_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader, data_iter


def load_VST(checkpoint, device):
    """load pretrained VST"""
    config = 'utils/swin_config/recognition/swin/swin_base_patch244_window877_kinetics400_22k_VAD.py'
    cfg = Config.fromfile(config)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')

    return model.to(device)


def main():
    args = get_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(args.gpu)
    device = get_torch_device()
    register_logger(log_file=args.log_file)

    if args.seed is not None:
        set_random_seed(args.seed)

    cudnn.benchmark = True

    feature_path = os.path.join(args.save_dir, 'L'+str(args.clip_length))

    if not path.exists(feature_path):
        mkdir(feature_path)

    data_loader, data_iter = get_features_loader(args.dataset_path,
                                                 args.clip_length,
                                                 args.frame_interval,
                                                 args.batch_size,
                                                 args.num_workers,
                                                 feature_path,
                                                 args.use_splits, )
    if data_loader.video_count == 0:
        return

    model = load_VST(args.pretrained_3d, device)

    features_writer = FeaturesWriter(num_videos=data_loader.video_count)
    loop_i = 0
    # Perform feature extraction on the dataset
    with torch.no_grad():
        for data, clip_idxs, dirs, vid_names in data_iter: # 1 batch
            outputs = model.extract_feat(data.to(device))
            outputs = outputs.mean(dim=[2, 3, 4])
            outputs = outputs.detach().cpu().numpy()

            for i, (dir, vid_name, clip_idx) in enumerate(zip(dirs, vid_names, clip_idxs)):
                if loop_i == 0:
                    logging.info(
                        f"Video {features_writer.dump_count} / {features_writer.num_videos} : Writing clip {clip_idx} of video {vid_name}")

                loop_i += 1
                loop_i %= args.log_every

                dir = path.join(feature_path, dir)
                features_writer.write(feature=outputs[i],
                                      video_name=vid_name,
                                      idx=clip_idx,
                                      dir=dir, )
    # Dump the remaining features to files
    features_writer.dump()


if __name__ == "__main__":
    main()
