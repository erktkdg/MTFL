import torch.utils.data as data
import os
import torch
torch.set_default_tensor_type('torch.FloatTensor')

class_to_int = {
    'Normal': 0,
    'Abuse': 1,
    'Arrest': 2,
    'Arson': 3,
    'Assault': 4,
    'Burglary': 5,
    'Explosion': 6,
    'Fighting': 7,
    'Robbery': 8,
    'Shooting': 9,
    'Shoplifting': 10,
    'Stealing': 11,
    'Vandalism': 12,
    'RoadAccidents_EMVvsEMV': 13,
    'RoadAccidents_EMVvsVRU': 14,
    'RoadAccidents_VRUvsVRU': 15,
    'DangerousThrowing': 16,
    'Littering': 17
}


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
        if self.test_mode:
            lf_path, mf_path, sf_path, label, file = self.list[index]
            l_features = read_features(lf_path)
            m_features = read_features(mf_path)
            s_features = read_features(sf_path)
            label = torch.tensor(label)
            return s_features, m_features, l_features, label, file
        else:
            lf_path, mf_path, sf_path, label = self.list[index]
            l_features = read_features(lf_path)
            m_features = read_features(mf_path)
            s_features = read_features(sf_path)
            label = torch.tensor(label)
            return s_features, m_features, l_features, label

    def __len__(self):
        return len(self.list)

    def _get_features_list(self, lf_dir, mf_dir, sf_dir, annotation_path):
        """
        Construct a feature list from the given directories and annotation file.

        Args:
            lf_dir (str): Directory path containing long-frame-length feature files.
            mf_dir (str): Directory path containing medium-frame-length feature files.
            sf_dir (str): Directory path containing short-frame-length feature files.
            annotation_path (str): Path to a text file containing annotation information.

        Returns:
            list: A list of tuples, each containing (lf_path, mf_path, sf_path, cls) or (lf_path, mf_path, sf_path, cls, file).

        Raises:
            AssertionError: If the input directories do not exist.

        Note:
            - If test_mode is True, each tuple contains (lf_path, mf_path, sf_path, cls, file), where file is the file name.
            - If test_mode is False, each tuple contains (lf_path, mf_path, sf_path, cls), and selection is based on whether it is normal (is_normal).

        """
        assert os.path.exists(lf_dir)
        assert os.path.exists(mf_dir)
        assert os.path.exists(sf_dir)
        features_list = []
        with open(annotation_path) as f:
            lines = f.read().splitlines(keepends=False)
            for line in lines:
                items = line.split()
                file = items[0].split(".")[0]
                file = file.replace("/", os.sep)
                lf_path = os.path.join(lf_dir, file + '.txt')
                mf_path = os.path.join(mf_dir, file + '.txt')
                sf_path = os.path.join(sf_dir, file + '.txt')
                unsupported_class = 18
                if not items[1].isdigit():
                    cls = class_to_int.get(items[1], unsupported_class)
                else:
                    cls = int(items[1])
                if self.test_mode:
                    features_list.append((lf_path, mf_path, sf_path, cls, file))
                elif (cls == class_to_int['Normal']) == self.is_normal:
                    features_list.append((lf_path, mf_path, sf_path, cls))

        return features_list

