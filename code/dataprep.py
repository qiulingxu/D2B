import functools
import numpy as np

import os
import config
from random import shuffle


def imagenet_dataset(mode, batch_size):
    from dataset.imagenet_utils import imagenet
    if mode == "train":
        imagenet_iter = imagenet(batch_size, dataset="train")
    elif mode == "eval":
        imagenet_iter = imagenet(batch_size, dataset="val")
    def next_batch():
        ### return x_batch [BS,224,224,3] RGB, y_batch [BS,]
        return imagenet_iter.get_next_batch()
    config.config["next_batch"] = next_batch
    config.config["data_statistics"] = {}

def cifar10_dataset(mode, batch_size):
    from dataset.cifar10_input import CIFAR10Data
    raw_cifar = CIFAR10Data("cifar10_data")
    if mode == "eval":
        cifar_data = raw_cifar.eval_data
    elif mode == "train":
        cifar_data = raw_cifar.train_data
    def next_batch():
        ### return x_batch [BS,224,224,3] RGB, y_batch [BS,]
        return cifar_data.get_next_batch(
            batch_size=batch_size, multiple_passes=True)
    config.config["next_batch"] = next_batch
    config.config["data_statistics"] = {}

def mnist_dataset(mode, batch_size):
    from mnist import MNIST
    mndata = MNIST(os.path.join(".",'mnist'))
    if mode == "train":
        images, labels = mndata.load_training()
    else:
        images, labels = mndata.load_testing()
    images = np.array(images).reshape([-1,28,28,1])
    labels = np.array(labels)
    def next_batch():
        nonlocal images, labels
        assert images.shape[0] == labels.shape[0]
        while True:
            l = images.shape[0]
            idx = list(range(l))
            shuffle(idx)
            images = images[idx]
            labels = labels[idx]
            for i in range(0, l - batch_size, batch_size):
                yield images[i:i+batch_size], labels[i:i+batch_size]
    next_batch_iter = next_batch()
    def wrap_next_batch():
        return next(next_batch_iter)

    config.config["next_batch"] = wrap_next_batch
    config.config["data_statistics"] = {}

def init_data(mode):
    assert mode in ["train","eval"]
    BATCH_SIZE = config.config["BATCH_SIZE"]
    data_set = config.config["DATA_SET"]

    assert data_set in ["CIFAR10","SVHN","IMAGENET", "MNIST"]

    if data_set == "IMAGENET":
        imagenet_dataset(mode,BATCH_SIZE)
    elif data_set == "CIFAR10":
        cifar10_dataset(mode,BATCH_SIZE)
    elif data_set == "MNIST":
        mnist_dataset(mode,BATCH_SIZE)
