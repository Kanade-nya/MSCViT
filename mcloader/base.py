from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

import torch
import numpy as np


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    Base class for dataset with support for offline distillation.
    """

    def __init__(self, split, transforms):
        self.primary_tfl = transforms
        self.features = None

    @abstractmethod
    def _get_data(self, index):
        """
        Returns the image and its label at index.
        """
        pass

    def __getitem__(self, index):
        img, label = self._get_data(index)
        img = self.primary_tfl(img)


        return img, label
