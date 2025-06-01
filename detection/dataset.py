import torch.utils.data as data
import os
import torch
torch.set_default_tensor_type('torch.FloatTensor')


def read_features(feature_path):
    """
    Read features from a text file and convert them into a torch tensor.

    Args:
        feature_path (str): Path to the text file containing features.

    Returns:
        features (torch.Tensor): A tensor containing the features. Shape is T x C.
    """
    with open(feature_path, 'r') as file:
        lines = file.readlines()
    features = []
    for line in lines:
        feature = [float(value) for value in line.strip().split()]
        features.append(feature)
    features = torch.tensor(features).float() # T x C
    return features


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        """
        Custom dataset class for loading features and labels.

        Args:
            args: Argument object containing paths and options.
            is_normal (bool): Whether the dataset represents normal samples.
            transform: Data transformation to be applied.
            test_mode (bool): Whether the dataset is for testing.

        Attributes:
            is_normal (bool): Whether the dataset represents normal samples.
            transform: Data transformation to be applied.
            test_mode (bool): Whether the dataset is for testing.
            list (list): List of feature paths and labels information.
        """
        self.is_normal = is_normal
        self.transform = transform
        self.test_mode = test_mode

        if self.test_mode:
            annotation_path = args.test_anno
        else:
            annotation_path = args.train_anno

        self.list = self._get_features_list(args.lf_dir, args.mf_dir, args.sf_dir, annotation_path)

    def __getitem__(self, index):
        label = self.get_label()
        if self.test_mode:
            lf_path, mf_path, sf_path, num_frames, start_end_couples, file = self.list[index]
            l_features = read_features(lf_path)
            m_features = read_features(mf_path)
            s_features = read_features(sf_path)
            return l_features, m_features, s_features, label, start_end_couples, num_frames, file
        else:
            lf_path, mf_path, sf_path = self.list[index]
            l_features = read_features(lf_path)
            m_features = read_features(mf_path)
            s_features = read_features(sf_path)
            return l_features, m_features, s_features, label

    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def _get_features_list(self, lf_dir, mf_dir, sf_dir, annotation_path):
        """
        Generate a list of features and labels information from annotations.

        Args:
            lf_dir (str): Path to long-frame-length features directory.
            mf_dir (str): Path to medium-frame-length features directory.
            sf_dir (str): Path to short-frame-length features directory.
            annotation_path (str): Path to annotation file.

        Returns:
            list: A list of tuples containing features and labels information.
        """
        assert os.path.exists(lf_dir)
        assert os.path.exists(mf_dir)
        assert os.path.exists(sf_dir)
        features_list = []
        with open(annotation_path) as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                items = line.split()
                #file = items[0].split(".")[0] for XD
                file, ext = os.path.splitext(items[0])
                file = file.replace("/", os.sep)
                lf_path = os.path.join(lf_dir, file + '.txt')
                mf_path = os.path.join(mf_dir, file + '.txt')
                sf_path = os.path.join(sf_dir, file + '.txt')
                cls_name = items[1]
                if self.test_mode:
                    start_end_couples = [int(x) for x in items[3:]]
                    num_frames = int(items[2])
                    features_list.append((lf_path, mf_path, sf_path, num_frames, start_end_couples, file))
                elif ("Normal" == cls_name) == self.is_normal:
                    features_list.append((lf_path, mf_path, sf_path))

        return features_list

