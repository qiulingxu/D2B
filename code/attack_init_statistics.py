import tensorflow as tf
import numpy as np

import multigpu
import classifiers
import dataprep
import config
from record import recorder
from layer_supervisor import layer_1x1cnn_dense_embed
from attacks import create_attack, _wrap_run_atk_with_switch
from tensorpack.tfutils.tower import TowerContext
import attacks
import os
import sys

from tensorflow.python import debug as tf_debug
from utils import save_pair_diff
from utils import np_l2_dist,np_linf_dist
from layer_supervisor import get_current_layer_wrapper, get_layer_wrapper_by_name
import argparse
from tqdm import tqdm

from redirect_output import overide_write

overide_write("try_internal_embed.log")
print("command arguments: ", sys.argv)

parser =  argparse.ArgumentParser(description=  "Internal Embedding Restriction")
parser.add_argument("--model", type=str,
                    choices=["Imagenet_Resnet_Normal_P", "Imagenet_Denoise_Resnet", "Imagenet_Resnet_Adv", "Imagenet_Denoise", "Imagenet_VGG16_AVG", "Imagenet_Resnet_Adv_For_Analysis"], default="Imagenet_Resnet_Normal")
#parser.add_argument("--targetlayer", type=int, help="Which layer to attack. Only in attack mode.", default=1)
parser.add_argument("--images", type=str, help="The folders to save the images.", default="embeds")
parser.add_argument("--gpus", type=int, help="The number of gpus to use.", default=None)
parser.add_argument("--overwriteattack", action="store_true", help="Attack the batch even it is in the database")
parser.add_argument("--embed", help="Attack the embedding dimensionally reduced from internal", action="store_true")
parser.add_argument("--internal", help="Attack the internal", action="store_true")

args = parser.parse_args()

# Denoise will generate forward NaN activation for second block and
# backward NaN gradient for first block on some samples.
# It is problematic to handle that. Thus let us ignore it first.

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
# config = tf.ConfigProto(gpu_options=gpu_options))

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)


### !!!! Find the bug of Imagenet_Denoise_Resnet Model, the accuracy is pretty low, Just Ignore the model for a while
model_name = args.model #"Imagenet_Resnet_Normal"#"Imagenet_Denoise"
attacks.Handler["Self_Feature_Induce_Attack"]["parameter"] = {}
print("Begin to use models for testing attacks: ", model_name)

config.set_data_set("IMAGENET")
task_id = "CollectStat"
attack_internal = True
dynamic_step_size = None
gpu_nums = args.gpus
if task_id == "Attack":
    config.config["MULTIGPU_NUM"] = 1 if gpu_nums is None else gpu_nums
    config.config["BATCH_SIZE"] = 32
    BATCH_NUM = 2
    dataprep.init_data("eval")
    training = False
    stop_gradient = False

    attack_embed = args.embed
    attack_internal = args.internal
    assert attack_embed ^ attack_internal, "You must specify one attack method"

elif task_id == "TrainEmbed":
    config.config["MULTIGPU_NUM"] = 2 if gpu_nums is None else gpu_nums
    config.config["BATCH_SIZE"] = 128
    dataprep.init_data("train")
    training = True
    stop_gradient = True
elif task_id == "CollectStat":
    config.config["MULTIGPU_NUM"] = 1
    config.config["BATCH_SIZE"] = 64
    dataprep.init_data("train")
    training = True
    stop_gradient = True
    attack_embed = args.embed
    attack_internal = args.internal
    assert attack_embed ^ attack_internal, "You must specify one attack method"
else:
    assert False

confidence = 50
if model_name == "Imagenet_Denoise":
    confidence = 50
elif model_name == "Imagenet_Resnet_Normal":
    confidence = 50

config.config["confidence"] = confidence

def idk_atk(x, y):
    return x

tot_cnt_atk = 0
class INT_BARR_Attack:
    def __init__(self, x_ph, y_ph, target_layer,
                 default_wrapper_name=None, collect_cache_name=None,
                 stat_type=None):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        assert config.config["MULTIGPU_NUM"] == 1, "Currently not supporting Multi GPU Attack"

        self.x_ph = x_ph
        self.y_ph = y_ph

        assert stat_type in ["channel_min_max",
                             "neuron_min_max", "neuron_variance", "neuron_gaussian"]
        self.stat_type = stat_type
        self.target_layer = target_layer

        self.BATCH_SIZE = config.config["BATCH_SIZE"]
        self.rec = recorder("self_induced_attack_intbarr")

        self.collect_cache_name = collect_cache_name
        self.collect_cache_name += "_" + self.stat_type

        if default_wrapper_name is None:
            self.default_wrapper_name = "classifier"
        else:
            self.default_wrapper_name = default_wrapper_name

        print("Chosen Layer Wrapper Name: ", self.default_wrapper_name)
        print("Target Layer: ",  self.target_layer)

    def build(self, container):
        global tot_cnt_atk
        eps = 1e-9
        with tf.variable_scope("INT_BARR_Attack_%d" % tot_cnt_atk):
            tot_cnt_atk +=1
            self.adv_loss_sep = container.target_loss
            self.adv_loss = container.target_loss_sum
            self.lwp = get_layer_wrapper_by_name(self.default_wrapper_name)

            raw_feat = self.lwp.get_layer_by_id(self.target_layer)
            self.internal_feature = tf.where(tf.is_finite(
                raw_feat), raw_feat, tf.zeros_like(raw_feat, dtype=tf.float32))

            FEATURE_SHAPE = self.internal_feature.shape.as_list()[1:]
            assert len(FEATURE_SHAPE) != 4 or FEATURE_SHAPE[0] == FEATURE_SHAPE[1], \
                "Please double check whether you provide NHWC format image, Remove this line manually if this is false positive alert."

            VAR_SHAPE = [self.BATCH_SIZE, *FEATURE_SHAPE]

            DYNAMIC_BS = tf.shape(self.x_ph)[0]

            if self.stat_type == "channel_min_max":
                if len(FEATURE_SHAPE) == 1:
                    stat_axis = 0
                    STAT_SHAPE = [1, *FEATURE_SHAPE]
                elif len(FEATURE_SHAPE) == 3:
                    stat_axis = [0, 1, 2]
                    STAT_SHAPE = [1, 1, 1, FEATURE_SHAPE[2]]
                else:
                    assert False

                self.moving_max = tf.get_variable(
                    "stat_max",  dtype=tf.float32, initializer=-1e10*tf.ones(STAT_SHAPE))
                self.moving_min = tf.get_variable(
                    "stat_min", dtype=tf.float32, initializer=1e10*tf.ones(STAT_SHAPE))
                # mean = tf.math.reduce_std(tf.abs(x), axis=[0])

                self.moving_upt = [
                    tf.assign(self.moving_max,
                              tf.maximum(tf.reduce_max(self.internal_feature, axis=stat_axis, keep_dims=True), self.moving_max)),
                    tf.assign(self.moving_min,
                              tf.minimum(tf.reduce_min(self.internal_feature, axis=stat_axis, keep_dims=True), self.moving_min))]
                # Plus an eps in case of error
                self.moving_range = self.moving_max-self.moving_min + eps
                self.collectable = {
                    "max": self.moving_max, "min": self.moving_min}

            elif self.stat_type == "neuron_min_max":
                STAT_SHAPE = [1, *FEATURE_SHAPE]
                stat_axis = 0

                self.moving_max = tf.get_variable(
                    "stat_max",  dtype=tf.float32, initializer=-1e10*tf.ones(STAT_SHAPE))
                self.moving_min = tf.get_variable(
                    "stat_min", dtype=tf.float32, initializer=1e10*tf.ones(STAT_SHAPE))
                # mean = tf.math.reduce_std(tf.abs(x), axis=[0])

                self.moving_upt = [
                    tf.assign(self.moving_max,
                              tf.maximum(tf.reduce_max(self.internal_feature, axis=stat_axis, keep_dims=True), self.moving_max)),
                    tf.assign(self.moving_min,
                              tf.minimum(tf.reduce_min(self.internal_feature, axis=stat_axis, keep_dims=True), self.moving_min))]
                # Plus an eps in case of error
                self.moving_range = self.moving_max-self.moving_min + eps
                self.collectable = {
                    "max": self.moving_max, "min": self.moving_min}

            elif self.stat_type == "neuron_variance" or self.stat_type == "neuron_gaussian":
                STAT_SHAPE = [1, *FEATURE_SHAPE]
                stat_axis = 0

                self.moving_mean = tf.get_variable(
                    "stat_mean",  dtype=tf.float64, initializer=tf.zeros(STAT_SHAPE, dtype=tf.float64))
                self.moving_var = tf.get_variable(
                    "stat_var", dtype=tf.float64, initializer=tf.zeros(STAT_SHAPE, dtype=tf.float64))
                self.cnt = tf.get_variable(
                    "stat_cnt", dtype=tf.float64, initializer=tf.cast(eps, dtype=tf.float64))
                # mean = tf.math.reduce_std(tf.abs(x), axis=[0])

                self.moving_upt = [
                    tf.assign_add(self.moving_mean, tf.reduce_sum(tf.cast(
                        self.internal_feature, dtype=tf.float64), axis=stat_axis, keep_dims=True)),
                    tf.assign_add(self.moving_var, tf.reduce_sum(
                        tf.square(tf.cast(self.internal_feature, dtype=tf.float64)), axis=stat_axis, keep_dims=True)),
                    tf.assign_add(self.cnt, tf.cast(
                        DYNAMIC_BS, dtype=tf.float64)),
                ]

                # Use three stddev as range
                self.moving_range = (3 * 2) * tf.sqrt(self.moving_var /
                                                      self.cnt - (self.moving_mean / self.cnt) ** 2) + eps
                self.moving_range = tf.cast(self.moving_range, tf.float32)
                self.collectable = {
                    "mean": self.moving_mean, "var": self.moving_var, "cnt": self.cnt}

            else:
                assert False, "stat type not implemented"


    def collect_statistics(self, sess):
        def run_collect():
            _collect = sess.run(self.collectable)
            return _collect
        if self.collect_cache_name is not None:
            self.rec.delete_if_exist(self.collect_cache_name)
            self.rec.save_record_by_id(self.collect_cache_name, 0, run_collect, overwrite=True)
        else:
            assert False



with tf.Session() as sess:

    classifiers.init_model(model_name, confidence = confidence)

    BATCH_SIZE = config.config["BATCH_SIZE"]
    IMAGE_SIZE = config.config["IMAGE_SIZE"]
    build_classifier = config.config["build_classifier"]
    restore_classifier = config.config["restore_classifier"]
    next_batch = config.config["next_batch"]

    """First step : define your inputs"""
    content = tf.placeholder(
        tf.float32, shape=[None, *IMAGE_SIZE], name='content')
    label = tf.placeholder(tf.int64, shape=[None], name="label")

    """Second step : create attack so that it can bind to classifiers and do some processing"""
    #for atk_name in ["PGD_Linf"]:  # Handler: "Self_Feature_Induce_Attack",
    #    print("Building attacks:", atk_name)
    #    create_attack(content, label,  atk_name, sess)

    #atk_func = config.config["atk_func"][atk_name]

    """Third step : build classifier"""
     # tf.get_variable("training", initializer=True)
    #with tf.device("/cpu:0"):
    atk_name = "Self_Feature_Induce_Attack"

    with layer_1x1cnn_dense_embed("classifier", content, label, training, stop_gradient=stop_gradient) as lwp:
        print(lwp)
        model_container, _ = build_classifier(content, label)

    lwp.build()

    if task_id in ["Attack", "CollectStat"] :
        embeds = lwp.embed_lwp

        if attack_embed:
            atk_lwp = lwp.embed_lwp
        elif attack_internal:
            atk_lwp = lwp

    classes = []
    updates = []    
    if model_name=="Imagenet_Resnet_Adv_For_Analysis":
        layers = 4*4+1
    else:
        layers = 5
        
    for layer_id in range(layers):#[0, 1, 2, 3, 4]:
        for stat_type in ["neuron_gaussian", "neuron_variance", "neuron_min_max", "channel_min_max"]:
            if attack_embed:
                default_wrapper_name = "lwp_embeds"
            elif attack_internal:
                default_wrapper_name = "classifier"
            else:
                assert False

            if attack_internal:
                collect_cache_name = "try_internal#%s#%d" % (
                    model_name, layer_id)
            elif attack_embed:
                collect_cache_name = "try_internal_embed#%s#%d" % (
                    model_name, layer_id)
            _cls = INT_BARR_Attack(
                content, label, layer_id, default_wrapper_name, collect_cache_name, stat_type)
            _cls.build(model_container)
            classes.append(_cls)
            updates.append(_cls.moving_upt)

    sess.run(tf.global_variables_initializer())
    print("Restoring parameters")
    restore_classifier(sess)

    iteration = int(1.28e6 // BATCH_SIZE)

    for _ in tqdm(range(iteration)):
        x, y = next_batch()
        sess.run(updates, feed_dict={
                    content: x, label: y})

    for _cls in classes:
        _cls.collect_statistics(sess)
