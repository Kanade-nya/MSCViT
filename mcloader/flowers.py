import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch

from mcloader.base import BaseDataset
from mcloader.io import pathmgr


class Flowers(BaseDataset):
    def __init__(self, data_path, split, transform):
        super(Flowers, self).__init__(split, transform)
        self.nb_classes = 102
        assert pathmgr.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for Flowers".format(split)
        self.data_path = data_path
        self.labels = loadmat(os.path.join(data_path, 'imagelabels.mat'))['labels'][0] - 1
        all_files = loadmat(os.path.join(data_path, 'setid.mat'))
        if split == 'train':
            self.ids = np.concatenate([all_files['trnid'][0], all_files['valid'][0]])
        else:
            self.ids = all_files['tstid'][0]

    def __len__(self):
        return len(self.ids)

    def __smooth_one_hot_labels(self, labels):
        n_classes, label_smooth = self.nb_classes, 0.0
        err_str = "Invalid input to one_hot_vector()"
        assert labels.ndim == 1 and labels.max() < n_classes, err_str
        shape = (labels.shape[0], n_classes)
        neg_val = label_smooth / n_classes
        pos_val = 1.0 - label_smooth + neg_val
        labels_one_hot = torch.full(shape, neg_val, dtype=torch.float, device=labels.device)
        labels_one_hot.scatter_(1, labels.long().view(-1, 1), pos_val)
        return labels_one_hot

    def _get_data(self, idx):
        label = self.labels[self.ids[idx] - 1]
        fname = 'image_%05d.jpg' % self.ids[idx]
        img = Image.open(os.path.join(self.data_path, 'jpg', fname))
        img = img.convert('RGB')
        # label = torch.Tensor(label)
        # label = self.__smooth_one_hot_labels(label)
        return img, label
