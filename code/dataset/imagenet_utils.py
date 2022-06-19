#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py

import os
import numpy as np
from abc import abstractmethod

import cv2
import tensorflow as tf

import multiprocessing as mp
import socket

from tensorpack import imgaug, dataset, ModelDesc
from tensorpack.dataflow import (
    BatchData, MultiThreadMapData, DataFromList, RepeatedData, LMDBSerializer)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.dataflow import (
    send_dataflow_zmq, MapData, TestDataSpeed, FakeData, dataset,
    AugmentImageComponent, BatchData, PrefetchDataZMQ, MultiProcessMapDataZMQ, LocallyShuffleData)

cpu_max = None

class imagenet:
    def __init__(self, batchsize, dataset= "val"):
        self.batchsize = batchsize
        self.dataset = dataset
        self.init()

    def init(self, ):
        global cpu_max
        if self.dataset == "train":
            cpu_max = 6
            augs = fbresnet_augmentor(True)
            #self.data = get_data("imagenet", self.batchsize, augs)
            self.data = get_val_dataflow(
                "imagenet", self.batchsize, dataname=self.dataset, augmentors=augs)
        elif self.dataset == "val":
            cpu_max = 6
            self.data = get_val_dataflow(
                "imagenet", self.batchsize, dataname=self.dataset)
        else:
            assert False
        self.data.reset_state()
        self.iter = iter(self.data)
        #self.data = tf.transpose(data, [0, 3, 1, 2])

    def get_next_batch(self):
        pack = next(self.iter, None)
        if pack is None:
            self.data.reset_state()
            self.iter = iter(self.data)
            pack = next(self.iter, None)
        
        x_batch, y_batch = pack
        x_batch = np.array(x_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.int32)
        return x_batch, y_batch


def get_data(data, batch, augmentors):
    """
    Sec 3, Remark 4:
    Use a single random shuffling of the training data (per epoch) that is divided amongst all k workers.

    NOTE: Here we do not follow the paper, but it makes little differences.
    """
    ds = dataset.ILSVRC12(data, 'train', shuffle=True)
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    ds = BatchData(ds, batch, remainder=False)
    ds = PrefetchDataZMQ(ds, min(cpu_max, mp.cpu_count()))
    return ds

def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
            imgaug.ColorSpace(cv2.COLOR_BGR2RGB)
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_val_dataflow(
        datadir, batch_size,
        augmentors=None, parallel=None,
        num_splits=None, split_index=None, dataname="val"):
    if augmentors is None:
        augmentors = fbresnet_augmentor(False)
    assert datadir is not None
    assert isinstance(augmentors, list)
    if parallel is None:
        parallel = min(cpu_max, mp.cpu_count())

    fast_cache_file = os.path.join(datadir, "ILSVRC-%s.lmdb"%dataname)
    use_cache =False
    def load_ds( shuffle = False):
        nonlocal use_cache
        if os.path.exists(fast_cache_file):
            ds = LMDBSerializer.load(fast_cache_file, shuffle=False)
            ds = LocallyShuffleData(ds, 1000)
            use_cache = True
        else:
            ds = dataset.ILSVRC12Files(datadir, dataname, shuffle=True)
            print("Alert Not using Fast Cache")
        return ds
    if num_splits is None:
        ds = load_ds(shuffle=False)
    else:
        # shard validation data
        assert split_index < num_splits
        files = load_ds(shuffle= True)
        files.reset_state()
        files = list(files.get_data())
        logger.info("Number of validation data = {}".format(len(files)))
        split_size = len(files) // num_splits
        start, end = split_size * split_index, split_size * (split_index + 1)
        end = min(end, len(files))
        logger.info("Local validation split = {} - {}".format(start, end))
        files = files[start: end]
        ds = DataFromList(files, shuffle=False)
    
    aug = imgaug.AugmentorList(augmentors)
    if use_cache:
        def mapf(dp):
            im, cls = dp
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            #from BGR to RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = aug.augment(im)
            return im, cls
    else:
        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            #from BGR to RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = aug.augment(im)
            return im, cls
    if dataname == "train":
        ds = MultiThreadMapData(ds, parallel, mapf,
                                buffer_size=min(200, ds.size()), strict=True)
             #MultiProcessMapDataZMQ(ds, parallel, mapf,
             #               buffer_size=min(200, ds.size()), strict=True)
    elif dataname == "val":
        ds = MultiThreadMapData(ds, parallel, mapf,
                                buffer_size=min(200, ds.size()), strict=True)
    else:
        assert False
    ds = BatchData(ds, batch_size, remainder=False)
    ds = RepeatedData(ds, num=-1)
    # do not fork() under MPI
    return ds


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))

