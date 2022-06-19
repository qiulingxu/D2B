import tensorflow as tf
import numpy as np
from PIL import Image
import os

import config

class container:
    def __init__(self):
        pass


def build_on_logits(logits, label=None, onehot_label=None, class_num=None, confidence=1, label_smoothing = 0.0, target_label = None):
    ### The higher loss, the better attack we get

    def prediction_correct(logits, label, topk=1, name='correct_vector'):
        with tf.name_scope('prediction_correct'):
            x = tf.nn.in_top_k(logits, label, topk)
        return tf.cast(x, tf.float32, name=name)

    cont = container()
    INF_LARGE = 1e7
    assert (label is None) ^ (onehot_label is None), "You can only specify one kind of label"
    if class_num is None:
        class_num = config.config["CLASS_NUM"]
    if label is not None:
        cont.label = label
        cont.onehot_label =tf.one_hot(label,class_num)
    elif onehot_label is not None:
        cont.onehot_label = onehot_label
        cont.label = tf.argmax(onehot_label,axis=-1)

    cont.logits = logits
    cont.prediction = tf.argmax(logits, axis=-1)
    cont.possibility = tf.nn.softmax(cont.logits, axis=-1)
    cont.correct_1 = prediction_correct(logits, label, 1, name='correct-top1')

    # for target attack
    if target_label is not None:
        cont.target_accuracy = prediction_correct(logits, target_label, 1)
        cont.target_accuracy_mean = tf.reduce_mean(cont.target_accuracy)
        cont.target_label = target_label
        cont.target_onehot_label = tf.one_hot(target_label,class_num)
        _target_attack_loss_true = tf.reduce_sum(cont.logits * cont.target_onehot_label, axis=-1)
        _target_attack_loss_wrong = tf.reduce_max(cont.logits * (1-cont.target_onehot_label) - cont.target_onehot_label * 1e7, axis=-1)
        cont.target_attack_loss = - tf.nn.relu( _target_attack_loss_wrong - _target_attack_loss_true  + confidence)
        cont.target_attack_loss_sum = tf.reduce_sum(cont.target_attack_loss)

    cont.accuracy_1 =  tf.reduce_mean(cont.correct_1)
    true_logits = tf.reduce_sum(
        cont.logits * cont.onehot_label, axis=-1)

    wrong_logits = tf.reduce_max(
        cont.logits * (1-cont.onehot_label) - cont.onehot_label * 1e7, axis=-1)
    cont.target_loss_1 = - tf.nn.relu(true_logits - wrong_logits + confidence)
    if class_num>50:
        cont.correct_5 = prediction_correct(logits, label, 5, name='correct-top5')
        cont.accuracy_5 = tf.reduce_mean(cont.correct_5)

        wrong_logits_5, _ = tf.nn.top_k(
            cont.logits * (1- cont.onehot_label) - cont.onehot_label * INF_LARGE, k=5, sorted=False)
        
        true_logits_5 = tf.reduce_sum(cont.logits * cont.onehot_label, axis=-1, keep_dims=True)
        cont.target_loss_5 = - tf.reduce_sum(tf.nn.relu(
            true_logits_5 - wrong_logits_5 + confidence), axis=1)
        ## Dynamically binding different top-k stats
        cont.correct = cont.correct_5
        cont.target_loss = cont.target_loss_5
        cont.target_loss_sum = tf.reduce_sum(cont.target_loss_5)
        cont.accuracy = cont.accuracy_5
        

    else:
        cont.accuracy = cont.accuracy_1
        cont.correct = cont.correct_1
        cont.target_loss = cont.target_loss_1
        cont.target_loss_sum = tf.reduce_sum(cont.target_loss)



    cont.entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label, logits=cont.logits)
    cont.smooth_entropy = tf.losses.softmax_cross_entropy(
        cont.onehot_label,
        logits, label_smoothing=label_smoothing)
    cont.smooth_entropy_mean = tf.reduce_mean(cont.smooth_entropy)

    cont.entropy_mean = tf.reduce_mean(cont.entropy)
    cont.entropy_sum = tf.reduce_sum(cont.entropy)

    return cont


def get_scope_var(scope_name):
    var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    #assert (len(var_list) >= 1)
    return var_list

def directory_path(path):
    return os.path.dirname(os.path.abspath(path))

def save_rgb_img(img, path):
    img = img.astype(np.uint8)
    dir_path = directory_path(path)
    os.makedirs(dir_path, exist_ok = True)
    #img=np.reshape(img,[28,28])
    Image.fromarray(img, mode='RGB').save(path)

def save_pair_diff(img1,img2,path, significant=None, dynamic=False):
    
    diff = np.abs(img1-img2)
    if dynamic:
        diff_max = np.amax(diff)+1e-5
        significant = 255.0/diff_max
    if significant is None:
        significant = 1
    diff = np.clip(diff*significant, 0, 255)
    combine_img = np.concatenate(
        (img1, img2, diff))
    save_rgb_img(combine_img, path)


def np_l2_dist(x1=None, x2=None, mask=None, scale=None, diff=None):
    if diff is None:
        l = len(x1.shape)
        samples = min(x1.shape[0], x2.shape[0])
        x1 = x1[:samples]
        x2 = x2[:samples]

        diff = (x1-x2)
    else:
        l = len(diff.shape)

    if scale is not None:
        diff = diff / scale

    diff = np.multiply(diff, diff)
    dist = np.sum(diff, axis=tuple(range(1, l)))
    dist = np.sqrt(dist)
    #print(dist)
    if mask is None:
        dist = np.mean(dist)
    else:
        dist = np.sum(dist*mask)/np.sum(mask)
    return dist


def np_linf_dist(x1=None, x2=None, mask=None, scale=None, diff=None):
    if diff is None:
        l = len(x1.shape)
        samples = min(x1.shape[0], x2.shape[0])
        x1 = x1[:samples]
        x2 = x2[:samples]

        diff = np.abs(x1-x2)
    else:
        l = len(diff.shape)

    if scale is not None:
        diff = diff / scale

    dist = np.max(diff, axis=tuple(range(1, l)))

    #print(dist)
    if mask is None:
        dist = np.mean(dist)
    else:
        dist = np.sum(dist*mask)/np.sum(mask)
    return dist


def np_l1_dist(x1=None, x2=None, mask=None, scale=None, diff=None):
    if diff is None:
        l = len(x1.shape)
        samples = min(x1.shape[0], x2.shape[0])
        x1 = x1[:samples]
        x2 = x2[:samples]

        diff = np.abs((x1-x2))
    else:
        l = len(diff.shape)

    if scale is not None:
        diff = diff / scale

    dist = np.sum(diff, axis=tuple(range(1, l)))

    #print(dist)
    if mask is None:
        dist = np.mean(dist)
    else:
        dist = np.sum(dist*mask)/np.sum(mask)
    return dist


# from https://gist.github.com/plasmaman/5508278
from scipy.stats import exponweib
from scipy.optimize import fmin
def fitweibull(x):
    def optfun(theta):
        return -np.sum(np.log(exponweib.pdf(x, 1, theta[0], scale=theta[1], loc=0)))
    logx = np.log(x)
    shape = 1.2 / np.std(logx)
    scale = np.exp(np.mean(logx) + (0.572 / shape))
    return fmin(optfun, [shape, scale], xtol=0.01, ftol=0.01, disp=0)




