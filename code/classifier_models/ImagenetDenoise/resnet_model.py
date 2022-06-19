# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from tensorpack.tfutils.argscope import argscope
from tensorpack.models import (
    Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm, FullyConnected, BNReLU)
from layer_supervisor import get_current_layer_wrapper
from sync_batch import sync_batch_norm
import config
import functools


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)



def resnet_bottleneck(l, ch_out, stride, group=1, res2_bottleneck=64, last=False):
    """
    Args:
        group (int): the number of groups for resnext
        res2_bottleneck (int): the number of channels in res2 bottleneck.
    The default corresponds to ResNeXt 1x64d, i.e. vanilla ResNet.
    """
    global resnet_output
    ch_factor = res2_bottleneck * group // 64
    shortcut = l
    l = Conv2D('conv1', l, ch_out * ch_factor, 1, strides=1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out * ch_factor, 3, strides=stride, activation=BNReLU, split=group)
    """
    ImageNet in 1 Hour, Sec 5.1:
    the stride-2 convolutions are on 3×3 layers instead of on 1×1 layers
    """
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    """
    ImageNet in 1 Hour, Sec 5.1: each residual block's last BN where γ is initialized to be 0
    """
    shortcut = resnet_shortcut(
        shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    if last:
        #
        # Note that the short cut of 2 to 4 blocks is ill-shaped distribution , could discard it for good sake
        if config.config["resnet_record_mode"] in ["None", "Combine"]:
            resnet_output = tf.concat([shortcut, l],  axis=1)
            output = config.config["record_func"](resnet_output)
            ms = output.shape.as_list()[1] //2
            ret = output[:,:ms] + output[:,ms:]
        elif config.config["resnet_record_mode"] in ["Main"]:
            resnet_output = l
            output = config.config["record_func"](resnet_output)
            ret = output + shortcut
        elif config.config["resnet_record_mode"] in ["Shortcut"]:
            resnet_output = shortcut
            output = config.config["record_func"](resnet_output)
            ret = l + output
    else:
        ret = l + shortcut
    
    return tf.nn.relu(ret, name='block_output')


def filter_record_array(x, layer_id):
    _sp = x.shape[3] // 2
    if layer_id == 0:
        return x
    if config.config["resnet_record_mode"] in ["None", "Combine"]:
        return x
    elif config.config["resnet_record_mode"] in ["Main"]:
        return x[:, :, :, _sp:]
    elif config.config["resnet_record_mode"] in ["Shortcut"]:
        return x[:, :, :, :_sp]
    else:
        assert False

def resnet_group(name, l, block_func, features, count, stride):
    
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                current_stride = stride if i == 0 else 1
                l = block_func(l, features, current_stride, last= (i==count-1))
    
    return l

def build_layer_wrapper_embeds(l, training):
    """The input should be assumed NHWC"""
    channel = l.shape.as_list()[3]
    #l = tf.transpose(l, [0, 3, 1, 2])
    """ Doesn't work, this is a numerical instable model, thresh = 1e9
    if training:
        l = tf.where(tf.greater(l, thresh),  tf.ones_like(l)*thresh, l)
        l = tf.where(tf.less(l, -thresh), tf.ones_like(l)*(-thresh), l)"""
    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'), \
            argscope(Conv2D, use_bias=False,
                    kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        #print("###############")
        #print(l.shape.as_list())
        l = sync_batch_norm(l, decay=0.995, epsilon=1e-5, is_training=training, scope ="bn1", offset= False)
        #print(l.shape.as_list())
        l = Conv2D('conv0', l, min(256, channel), 1, strides=1, activation=None)
        #print(l.shape.as_list())
        l = tf.nn.relu(l)
        #l = BatchNorm('bn', l, sync_statistics="nccl")
        l = GlobalAvgPooling('gap', l)
        #print(l.shape.as_list())

        l = sync_batch_norm(l, decay=0.995, epsilon=1e-5,
                            is_training=training, scope="bn2", offset=False)
        #print(l.shape.as_list())
        embeds = l

        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits, embeds

def resnet_backbone(image, num_blocks, group_func, block_func):
    global resnet_output
    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, use_bias=False,
                     kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        lwp = get_current_layer_wrapper()
        def new_record(input, idx):
            #print("idx=",idx)
            _input0 = tf.transpose(input, [0,2,3,1])
            _input = lwp.record_by_id(_input0, idx)
            if _input is _input0:
                return input
            else:
                _input = tf.transpose(_input, [0,3,1,2])
                return _input
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=None)
        # break BNRelu down into parts so I can insert the monitor after BatchNorm
        with tf.variable_scope("conv0"):
            l = BatchNorm('bn', l)
            l = new_record(l, 0)
            l = tf.nn.relu(l)
        #l = BNReLU(l)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')   
        config.config["record_func"] = functools.partial(new_record, idx=1)     
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        #new_record(resnet_output, 1)
        config.config["record_func"] = functools.partial(new_record, idx=2)   
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        #new_record(resnet_output, 2)
        config.config["record_func"] = functools.partial(new_record, idx=3)   
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        #new_record(resnet_output, 3)
        config.config["record_func"] = functools.partial(new_record, idx=4)   
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        #new_record(resnet_output, 4)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        """
        ImageNet in 1 Hour, Sec 5.1:
        The 1000-way fully-connected layer is initialized by
        drawing weights from a zero-mean Gaussian with standard deviation of 0.01
        """
    return logits


def resnet_bottleneck_for_analysis(l, ch_out, stride, group=1, res2_bottleneck=64, last=False):
    """
    Args:
        group (int): the number of groups for resnext
        res2_bottleneck (int): the number of channels in res2 bottleneck.
    The default corresponds to ResNeXt 1x64d, i.e. vanilla ResNet.
    """
    global resnet_output
    ch_factor = res2_bottleneck * group // 64
    shortcut = l
    l = Conv2D('conv1', l, ch_out * ch_factor, 1, strides=1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out * ch_factor, 3,
               strides=stride, activation=BNReLU, split=group)
    """
    ImageNet in 1 Hour, Sec 5.1:
    the stride-2 convolutions are on 3×3 layers instead of on 1×1 layers
    """
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    """
    ImageNet in 1 Hour, Sec 5.1: each residual block's last BN where γ is initialized to be 0
    """
    shortcut = resnet_shortcut(
        shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    if last:
        ret = l + shortcut
        # Note that the short cut of 2 to 4 blocks is ill-shaped distribution , could discard it for good sake
        # after relu, before relu, without shortcut
        config.config["record_func"](tf.nn.relu(ret))
        config.config["record_func"](ret)
        config.config["record_func"](l)
        config.config["record_func"](shortcut)
    else:
        ret = l + shortcut

    return tf.nn.relu(ret, name='block_output')

def resnet_backbone_for_analysis(image, num_blocks, group_func, block_func):
    global resnet_output
    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, use_bias=False,
                     kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        lwp = get_current_layer_wrapper()

        def new_record(input):
            #print("idx=",idx)
            _input0 = tf.transpose(input, [0, 2, 3, 1])
            _input = lwp.record(_input0)
            if _input is _input0:
                return input
            else:
                _input = tf.transpose(_input, [0, 3, 1, 2])
                return _input
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=None)
        # break BNRelu down into parts so I can insert the monitor after BatchNorm
        with tf.variable_scope("conv0"):
            l = BatchNorm('bn', l)
            new_record(l)
            l = tf.nn.relu(l)
        #l = BNReLU(l)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        config.config["record_func"] = new_record
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        #new_record(resnet_output, 4)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        """
        ImageNet in 1 Hour, Sec 5.1:
        The 1000-way fully-connected layer is initialized by
        drawing weights from a zero-mean Gaussian with standard deviation of 0.01
        """
    return logits


def denoising(name, l, embed=True, softmax=True):
    """
    Feature Denoising, Fig 4 & 5.
    """
    with tf.variable_scope(name):
        f = non_local_op(l, embed=embed, softmax=softmax)
        f = Conv2D('conv', f, l.shape[1], 1, strides=1, activation=get_bn(zero_init=True))
        l = l + f
    return l


def non_local_op(l, embed, softmax):
    """
    Feature Denoising, Sec 4.2 & Fig 5.
    Args:
        embed (bool): whether to use embedding on theta & phi
        softmax (bool): whether to use gaussian (softmax) version or the dot-product version.
    """
    n_in, H, W = l.shape.as_list()[1:]
    if embed:
        theta = Conv2D('embedding_theta', l, n_in / 2, 1,
                       strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        phi = Conv2D('embedding_phi', l, n_in / 2, 1,
                     strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        g = l
    else:
        theta, phi, g = l, l, l
    if n_in > H * W or softmax:
        f = tf.einsum('niab,nicd->nabcd', theta, phi)
        if softmax:
            orig_shape = tf.shape(f)
            f = tf.reshape(f, [-1, H * W, H * W])
            f = f / tf.sqrt(tf.cast(theta.shape[1], theta.dtype))
            f = tf.nn.softmax(f)
            f = tf.reshape(f, orig_shape)
        f = tf.einsum('nabcd,nicd->niab', f, g)
    else:
        f = tf.einsum('nihw,njhw->nij', phi, g)
        f = tf.einsum('nij,nihw->njhw', f, theta)
    if not softmax:
        f = f / tf.cast(H * W, f.dtype)
    return tf.reshape(f, tf.shape(l))
