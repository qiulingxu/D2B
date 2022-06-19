import tensorflow as tf
import numpy as np
import os

import config
import dataprep

from utils import container, get_scope_var, build_on_logits
from multigpu import multi_gpu




def add_layer_wrapper_embeds_func(model_name, func):
    config.config["classifier_func"][model_name]["layer_wrapper_embeds"] = func

def get_current_lwp_embeds_func():
    MODEL_NAME = config.config["MODEL_NAME"]
    if "layer_wrapper_embeds" in config.config["classifier_func"][MODEL_NAME]:
        return config.config["classifier_func"][MODEL_NAME]["layer_wrapper_embeds"]
    else:
        return None

def Imagenet_Denoise(type):
    import tensorflow as tf
    from tensorpack.tfutils import SmartInit
    
    from tensorpack.tfutils.tower import TowerContext

    if type == 1:
        from classifier_models.ImagenetDenoise.nets import ResNeXtDenoiseAllModel as Model
        depth = 101
        path = "X101-DenoiseAll.npz"
    elif type == 2:
        from classifier_models.ImagenetDenoise.nets import ResNetDenoiseModel as Model
        depth = 152
        path = "R152-Denoise.npz"
    elif type == 3:
        from classifier_models.ImagenetDenoise.nets import ResNetModel as Model
        depth = 152
        path = "R152.npz"
    elif type == 4:
        from classifier_models.ImagenetDenoise.nets import ResNetModel_For_Analysis as Model
        depth = 152
        path = "R152.npz"
    else:
        assert False
    from classifier_models.ImagenetDenoise.resnet_model import filter_record_array
    from classifier_models.ImagenetDenoise.resnet_model import build_layer_wrapper_embeds as build_lwp_embeds
    MODEL_NAME = config.config["MODEL_NAME"]
    add_layer_wrapper_embeds_func(MODEL_NAME, build_lwp_embeds)
    config.config["filter_record_array"] = filter_record_array

    def build_model(image, label):
        args = container()
        args.depth = depth
        image = tf.reverse(image, axis= [-1])
        reuse = config.config["reuse"]
        with TowerContext(tower_name='', is_training=False):
            with tf.variable_scope("", auxiliary_name_scope=False, reuse=reuse):
                model = Model(args)
                model.build_graph(image, label)
        return model.logits, model
    def restore_parameter(sess, var_list = None):
        file_path = os.path.join("checkpoints", path)
        sessinit = SmartInit(file_path)
        sessinit.init(sess)
    return build_model, restore_parameter

def Imagenet_Resnet():
    from classifier_models.ImagenetResnet import resnet_slim
    from classifier_models.ImagenetResnet.layerwrapbuilder import lwp_embed_build as build_lwp_embeds
    slim = tf.contrib.slim
    MODEL_NAME = config.config["MODEL_NAME"]
    add_layer_wrapper_embeds_func(MODEL_NAME, build_lwp_embeds)

    config.config["filter_record_array"] = resnet_slim.filter_record_array
    def build_model(image, label):
        reuse = config.config["reuse"]
        image = tf.reverse(image, axis=[-1])
        with slim.arg_scope(resnet_slim.resnet_arg_scope()):
            logits, desc = resnet_slim.resnet_v1_50(
                image, num_classes=1000, is_training=False, reuse=reuse)
        return logits, desc

    def restore_parameter(sess, var_list=None):
        file_path = os.path.join("checkpoints", "imagenet_resnet_v1_50.ckpt")
        if var_list is None:
            var_list = get_scope_var("resnet_v1")
        saver = tf.train.Saver(var_list)
        saver.restore(sess, file_path)
    return build_model, restore_parameter

def Imagenet_keras_model(name):
    def get_model(name):
        assert name in [ "resnet50","vgg19", "inception_v3", "mobilenet", "densenet121"]
        if name == "vgg19":
            from tensorflow.keras.applications.vgg19 import VGG19 as Model, preprocess_input
        elif name == "inception_v3":
            from tensorflow.keras.applications.inception_v3 import InceptionV3 as Model, preprocess_input
        elif name == "efficientnetb0":
            from tensorflow.keras.applications.efficientnet import EfficientNetB0 as Model, preprocess_input
        elif name == "densenet121":
            from tensorflow.keras.applications.densenet import DenseNet121 as Model, preprocess_input
        elif name == "mobilenet":
            from tensorflow.keras.applications.mobilenet import MobileNet as Model, preprocess_input
        elif name == "resnet50":
            from tensorflow.keras.applications.resnet import ResNet50 as Model, preprocess_input
        
        return Model, preprocess_input

    Model, preprocess_input = get_model(name)
    temp_model = None
    def build_model(image, label):
        nonlocal temp_model
        image = preprocess_input(image)
        model = Model(weights="imagenet", input_tensor=image)
        temp_model = model
        from getkeraslogit import KerasModelWrapper
        kmwrap = KerasModelWrapper(model)
        return kmwrap.get_logits(image), kmwrap

    def restore_parameter(sess, var_list=None):
        from tensorflow.python.keras.utils import data_utils
        if name == "densenet121":
            BASE_WEIGTHS_PATH = ('https://storage.googleapis.com/tensorflow/'
                                'keras-applications/densenet/')
            DENSENET121_WEIGHT_PATH = (
                BASE_WEIGTHS_PATH + 'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
            DENSENET121_WEIGHT_PATH_NO_TOP = (
                BASE_WEIGTHS_PATH +
                'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
            DENSENET169_WEIGHT_PATH = (
                BASE_WEIGTHS_PATH + 'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
            DENSENET169_WEIGHT_PATH_NO_TOP = (
                BASE_WEIGTHS_PATH +
                'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
            DENSENET201_WEIGHT_PATH = (
                BASE_WEIGTHS_PATH + 'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
            DENSENET201_WEIGHT_PATH_NO_TOP = (
                BASE_WEIGTHS_PATH +
                'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = data_utils.get_file(
                'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                DENSENET121_WEIGHT_PATH,
                cache_subdir='models',
                file_hash='9d60b8095a5708f2dcce2bca79d332c7')
        elif name == "mobilenet":
            rows = 224
            alpha = 1.0
            if alpha == 1.0:
                alpha_text = '1_0'
            elif alpha == 0.75:
                alpha_text = '7_5'
            elif alpha == 0.50:
                alpha_text = '5_0'
            else:
                alpha_text = '2_5'
            BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                                'keras-applications/mobilenet/')
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir='models')            
            
        elif name == "inception_v3":
            WEIGHTS_PATH = (
                'https://storage.googleapis.com/tensorflow/keras-applications/'
                'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
            weights_path = data_utils.get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        elif name == "vgg19":
            WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                            'releases/download/v0.1/'
                            'vgg19_weights_tf_dim_ordering_tf_kernels.h5')        
            weights_path = data_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')

        temp_model.load_weights(weights_path)
    return build_model, restore_parameter

def Imagenet_Resnet_Preprocess():
    from classifier_models.ImagenetResnet import resnet_slim
    from classifier_models.ImagenetResnet.layerwrapbuilder import lwp_embed_build as build_lwp_embeds
    slim = tf.contrib.slim
    MODEL_NAME = config.config["MODEL_NAME"]
    add_layer_wrapper_embeds_func(MODEL_NAME, build_lwp_embeds)

    config.config["filter_record_array"] = resnet_slim.filter_record_array

    def build_model(image, label):
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94

        mean = np.array([[[[_R_MEAN, _G_MEAN, _B_MEAN]]]])
        reuse = config.config["reuse"]
        image = image - mean
        #image = tf.reverse(image, axis=[-1])
        with slim.arg_scope(resnet_slim.resnet_arg_scope()):
            logits, desc = resnet_slim.resnet_v1_50(
                image, num_classes=1000, is_training=False, reuse=reuse)
        return logits, desc

    def restore_parameter(sess, var_list=None):
        file_path = os.path.join("checkpoints", "imagenet_resnet_v1_50.ckpt")
        if var_list is None:
            var_list = get_scope_var("resnet_v1")
        saver = tf.train.Saver(var_list)
        saver.restore(sess, file_path)
    return build_model, restore_parameter



def Imagenet_Vgg(ver = 16, pooling ="max"):
    from classifier_models.vgg.vgg16 import Vgg16
    MODEL_NAME = config.config["MODEL_NAME"]

    vclass = Vgg16(os.path.join("checkpoints", "vgg16.npy"), pooling=pooling)
    def build_model(image, label):
        vclass.build(image)
        return vclass.fc8, vclass

    def restore_parameter(sess, var_list=None):
        pass

    return build_model, restore_parameter

def CIFAR10_Madry(mode):
    from classifier_models.CIFAR10Madry.model import Model as Madrymodel
    model = Madrymodel("eval")
    new_var_list = None

    def build_model(image, label, reuse=False):
        nonlocal new_var_list
        before = get_scope_var(None)
        with tf.variable_scope("", reuse=reuse):
            model._build_model(image, label)
        after = get_scope_var(None)
        new_var_list = [i for i in set(after)-set(before)]
  
        return model.pre_softmax, {"model":model}

    def restore_parameter(sess, var_list=None):
        saver = tf.train.Saver(new_var_list)
        if mode == 0:
            path = os.path.join("checkpoints", "madrycifar10", "adv_trained","checkpoint-70000")
        elif mode == 1:
            path = os.path.join("checkpoints", "madrycifar10",
                                "naturally_trained", "checkpoint-70000")
        else:
            assert False
        saver.restore(sess, path)

    return build_model, restore_parameter

def MNIST_Madry(mode):
    from classifier_models.MNISTCNN.mnist import Model as Madrymodel
    model = Madrymodel()
    new_var_list = None

    def build_model(image, label, reuse=False):
        nonlocal new_var_list
        image = image / 255.0
        before = get_scope_var(None)
        with tf.variable_scope("", reuse=reuse):
            model.build_model(image, label)
        after = get_scope_var(None)
        new_var_list = [i for i in set(after)-set(before)]
  
        return model.pre_softmax, {"model":model}

    def restore_parameter(sess, var_list=None):
        saver = tf.train.Saver(new_var_list)
        if mode == 0:
            path = os.path.join("checkpoints", "madrymnist",
                                "adv_trained", "checkpoint-99900")
        elif mode == 1:
            path = os.path.join("checkpoints", "madrymnist",
                                "normal", "checkpoint-24900")
        saver.restore(sess, path)

    return build_model, restore_parameter

Handlers = {"Imagenet_Denoise":
                {"function": Imagenet_Denoise, "parameter": {"type": 1}, "dataset": "IMAGENET"},
            "Imagenet_Denoise_Resnet":
                {"function": Imagenet_Denoise, "parameter": {"type": 2}, "dataset": "IMAGENET"},
            "Imagenet_Resnet_Adv":
                {"function": Imagenet_Denoise, "parameter": {
                    "type": 3}, "dataset": "IMAGENET"},
            "Imagenet_VGG16":
                {"function": Imagenet_Vgg, "parameter": {"ver": 16, "pooling": "max"}, "dataset": "IMAGENET"},
            "Imagenet_VGG16_AVG":
                {"function": Imagenet_Vgg, "parameter": {"ver": 16, "pooling": "avg"}, "dataset": "IMAGENET"},                
            "Imagenet_Resnet_Normal":
                {"function": Imagenet_Resnet, "parameter": {}, "dataset": "IMAGENET"},
            "Imagenet_Resnet_Normal_P":
                {"function": Imagenet_Resnet_Preprocess, "parameter": {}, "dataset": "IMAGENET"},
            "CIFAR10_Resnet_Madry": 
                {"function": CIFAR10_Madry, "parameter": {"mode":0}, "dataset": "CIFAR10"},
            "CIFAR10_Resnet_Normal":
                {"function": CIFAR10_Madry, "parameter": {"mode":1}, "dataset": "CIFAR10"},
            "MNIST_CNN_Madry":
                {"function": MNIST_Madry, "parameter": {"mode":0}, "dataset": "MNIST"},
            "MNIST_CNN_Normal":
                {"function": MNIST_Madry, "parameter": {
                    "mode": 1}, "dataset": "MNIST"},
            "Imagenet_Resnet_Adv_For_Analysis":
                {"function": Imagenet_Denoise, "parameter": {
                    "type": 4}, "dataset": "IMAGENET"},
            "Imagenet_DenseNet":{
                "function": Imagenet_keras_model, "parameter":{"name": "densenet121"}, "dataset": "IMAGENET",
            },
            "Imagenet_MobileNet":{
                "function": Imagenet_keras_model, "parameter":{"name": "mobilenet"}, "dataset": "IMAGENET",
            },
            "Imagenet_VGG19":{
                "function": Imagenet_keras_model, "parameter":{"name": "vgg19"}, "dataset": "IMAGENET",
            }                        
           }

def init_model(name, confidence = 1):
    assert name in Handlers, "Unspecified Model Name"
    config.config["MODEL_NAME"] = name
    config.config["classifier_func"][name] = {}
    info = Handlers[name]
    callback_func = info["function"]
    params = info["parameter"]
    build_model, restore_classifier = callback_func(**params)
    build_model = multi_gpu(build_model, input_num=2, output_num=1, tot_output_num=2)
    def build_classifier(image, label, target_label=None, **kargs):
        if "pre_built_attack" in config.config:
            """ 
                Some attacker needs to preprocess the input in advance before feeding to classifier.
                Thus we build a multiplexing subgraph here.
                The input combining different source of input : natural one or adversarial one
                    selected from the assignment of switch_var.
            """

            config.config["create_attack_finish_flag"] = True
            pre_built_attack = config.config["pre_built_attack"]
            attack_list = pre_built_attack["list"]
            if len(attack_list)!=0:
                switch_var = pre_built_attack["switch"]
                switch_num = len(attack_list)+1
                switch_image = [image]
                switch_ops = {"default" : [tf.assign(switch_var,0)]}
                for idx,atk_info in enumerate(attack_list):
                    atk_name = atk_info["name"]
                    pre_built_func = atk_info["function"]
                    intermediate_image = pre_built_func()#image, label)
                    switch_image.append(intermediate_image)
                    switch_ops[atk_name] = [tf.assign(switch_var,idx+1)]
                config.config["pre_built_attack"]["switch_ops"] = switch_ops
                switch_image = tf.stack(switch_image, axis=0)
                switch_mask = tf.one_hot(switch_var, depth= switch_num)
                switch_mask = tf.reshape(switch_mask, [switch_num, 1, 1, 1, 1])
                image = tf.reduce_sum(switch_image * switch_mask, axis=0)
        
        logits, aux = build_model(image, label, **kargs)
        ### Set the confidence to a large one so one can fully express the power of domain
        cont = build_on_logits(logits, label=label, confidence=confidence, target_label=target_label)
        return cont, aux
    config.config["build_classifier"] = build_classifier
    config.config["restore_classifier"] = restore_classifier
    

if __name__=="__main__":
    from forexperiment import get_one_gpu
    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    for model_name in ["Imagenet_VGG19"]:  # Handlers.keys():
        tf.reset_default_graph()
        dummy_op = get_one_gpu()
        with tf.Session(config=config1) as sess:
            dummy_op(sess)
            print("Begin to check models: ", model_name)
            info = Handlers[model_name]
            config.set_data_set(info["dataset"])
            
            print("Preparing data:", info["dataset"])
            dataprep.init_data("eval")

            print("Building models:", info["function"])
            init_model(model_name)

            BATCH_SIZE = config.config["BATCH_SIZE"]
            IMAGE_SIZE = config.config["IMAGE_SIZE"]
            build_classifier = config.config["build_classifier"]
            restore_classifier = config.config["restore_classifier"]
            next_batch = config.config["next_batch"]

            content = tf.placeholder( tf.float32, shape=[None, *IMAGE_SIZE], name='content')
            label = tf.placeholder(tf.int64, shape=[None], name="label")
            model_container, _ = build_classifier(content, label)

            print("Restoring parameters")
            restore_classifier(sess)

            for test_batch in range(2):
                x_batch, y_batch = next_batch()
                _pred, _acc = sess.run([model_container.prediction,
                                  model_container.accuracy], feed_dict={
                                     content: x_batch, label: y_batch})
                print("Label:      ", y_batch)
                print("Prediction: ", _pred)
                print("Accuracy: ", _acc)


