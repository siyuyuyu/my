#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adapted from https://github.com/txie-93/cgcnn/blob/master/cgcnn/main.py
"""
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from utilities.log import logger


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


class AverageMeter(object):
    """
    Computes and stores average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]

    # dataset.to(device)
    # mean = dataset.y.mean(dim=0, keepdim=True)
    # std = dataset.y.std(dim=0, keepdim=True)
    # dataset.y = (dataset.y - mean) / std
    # mean, std = mean[:, 0].item(), std[:, 0].item()


def get_train_val_test_indices(data_size, train_ratio=None, val_ratio=0.1, test_ratio=0.1):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    !!! The dataset needs to be shuffled before using the function !!!
    Parameters
    ----------
    data_size: float
    train_ratio: float
    val_ratio: float
    test_ratio: float

    Returns
    -------
    """

    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        logger.warning("train_ratio is None, using all training data.")
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(data_size))

    train_size = int(train_ratio * data_size)

    test_size = int(test_ratio * data_size)

    valid_size = int(val_ratio * data_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size) : -test_size])
    test_sampler = SubsetRandomSampler(indices[-test_size:])

    return train_sampler, val_sampler, test_sampler
