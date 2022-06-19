from tensorflow.python.ops import math_ops
import numpy as np
import tensorflow as tf
import config

from ops import cnn, dense, batch_norm
from sync_batch import sync_batch_norm
from utils import build_on_logits
from record import recorder
from multigpu import multi_gpu, multi_gpu_decor, multi_adam, multi_sgd, multi_plus, multi_distribute
from classifiers import get_current_lwp_embeds_func
from tensorpack.tfutils.tower import TowerContext

IS_DEBUG = True


def get_layer_num(dct):
    cnt = 0
    for i in range(100):
        if "Layer_%d" % i in dct:
            cnt = i+1
    return cnt

@multi_gpu_decor(input_num=1, output_num=1, tot_output_num=1)
def sanitize_nan(input):
    return tf.where(tf.is_finite(input), x=input, y=tf.zeros_like(input, dtype=tf.float32))

class monitor_tf_objects():
    def __init__(self,):
        self.track_name = {
            "all": tf.GraphKeys.GLOBAL_VARIABLES,
            "train": tf.GraphKeys.TRAINABLE_VARIABLES,
            "update": tf.GraphKeys.UPDATE_OPS}
        self.temp_save = {}
        self.increment = {}

    def get_name(self, name):
        return tf.get_collection(name)

    def __enter__(self):
        for k in self.track_name.keys():
            self.temp_save[k] = self.get_name(self.track_name[k])
        return self

    def list_minus(self, list_a, list_b):
        """return elems in list_a but not in list_b"""
        list_result = []
        for elems in list_a:
            if elems not in list_b:
                list_result .append(elems)
        return list_result

    def __exit__(self, type, value, traceback):
        for k in self.track_name.keys():
            curr = self.get_name(self.track_name[k])
            self.increment[k] = self.list_minus(curr, self.temp_save[k])

    def get_objects(self, name):
        return self.increment[name]


class layer_base():
    def __init__(self, name=None):
        global LWP_UID, LWP_LOOPUP_DICT
        self.children = []
        if name is None:
            name = str("DEFNAME_LWP_UID_%d" % LWP_UID)
        LWP_LOOPUP_DICT[name] = self
        self.FIRST_INIT = True
        self.layers = [None ] * 128
        self.layer_num = 0

    def _is_init(self):
        return not self.FIRST_INIT

    def get_layer(self):
        return self.layers[:self.layer_num]

    def record(self, layer_input):
        assert not config.config["MULTI_GPU_ENV"]
        if not self._is_init():
            self.layers[self.layer_num] = layer_input
            self.layer_num += 1
        return layer_input

    def record_by_id(self, layer_input, idx):
        if not self._is_init():
            new_layer_num = idx + 1
            assert new_layer_num <= self.layer_num + 1
            if self.layers[idx] is None:
                self.layers[idx] = layer_input
            else:
                self.layers[idx] = tf.concat([self.layers[idx],layer_input], axis=0)
            self.layer_num = max(self.layer_num, new_layer_num)
        return layer_input

    def build(self,):
        pass

    def add_child(self, child):
        self.children.append(child)

    def get_child(self):
        return self.children

    def __enter__(self):
        self.father = config.config["ENV_LAYER_RECORD"][-1]
        self.father.add_child(self)
        config.config["ENV_LAYER_RECORD"].append(self)
        return self

    def __exit__(self, type, value, traceback):
        # Exception handling here
        config.config["ENV_LAYER_RECORD"] = config.config["ENV_LAYER_RECORD"][:-1]
        self.FIRST_INIT = False

    def get_layer_by_id(self, id):
        return self.layers[id]

    def get_dict_output(self, _lst, prefix =""):
        _dict = {}
        for i in range(self.layer_num):
            _dict["%sLayer_%d" % (prefix,i)] = _lst[i]
        return _dict

    def get_dict_layers(self, prefix=""):
        _dict = {}
        for i in range(self.layer_num):
            _dict["%sLayer_%d" % (prefix, i)] = sanitize_nan(self.layers[i])
        return _dict


class layer_record(layer_base):
    # Note that the record will only record nodes for the first time of running.
    # It is only supported for with ... : command
    def __init__(self, name):
        super().__init__(name)

    def record(self, layer_input):
        layer_input = super().record(layer_input)
        return layer_input

class hooked_layer_record(layer_base):
    def __init__(self, name, hook_layer):
        super().__init__(name)
        self.hook_layer = hook_layer
    
    def record(self, layer_input):
        layer_input = super().record(layer_input)
        print(layer_input , self.layer_num, self.hook_layer)
        if self.layer_num - 1 == self.hook_layer:
            layer_input = self.hook_call_back(layer_input)
        return layer_input
    
    def record_by_id(self, layer_input, idx):
        layer_input = super().record_by_id(layer_input, idx)
        if idx == self.hook_layer:
            layer_input = self.hook_call_back(layer_input)
        return layer_input
    
    def hook_call_back(self, input):
        assert False, "To be implemented"
        return input

class two_step_opt(hooked_layer_record):
    def hook_call_back(self, input):
        print("Building two step opt through hook layer")
        BS = config.config["BATCH_SIZE"]
        sp = input.shape.as_list()
        sp[0] = BS
        with tf.variable_scope("two_step_hooked_layer"):
            self.rang = tf.get_variable("range", shape=[1]+sp[1:])
            self.opt = tf.get_variable("int_var", shape=sp)
            self.org_sample = tf.get_variable("org_sample", shape=sp)
            self.atk_init = [tf.assign(self.opt, input), tf.assign(self.org_sample, input)]
            self.loss_simulate = tf.reduce_mean(tf.square(input - self.opt) )
            
        return self.opt

    def asgn(self, sess, rang):
        sess.run(tf.assign(self.rang, rang))

class clip_layer(hooked_layer_record):
    def __init__(self, name, hook_layer):
        super().__init__(name,hook_layer)
        #self.range = range
        
    def hook_call_back(self, input):
        BS = config.config["BATCH_SIZE"]
        sp = input.shape.as_list()
        sp[0] = BS
        with tf.variable_scope("clip_hooked_layer"):
            self.rang = tf.get_variable("range", shape=[1]+sp[1:])
            self.org_sample = tf.get_variable("org_sample", shape=sp)
            self.atk_init = [tf.assign(self.org_sample, input)]
            input = tf.clip_by_value(input, input-self.rang, input+self.rang)
            
        return input
    
    def asgn(self, sess, rang,):
        sess.run(tf.assign(self.rang, rang))

class layer_1x1cnn_dense_embed(layer_base):
    def __init__(self, name, content, label, training, stop_gradient=True):
        self.stop_gradient = stop_gradient
        self.training = training
        self.layer_logits = {}
        self.container = {}
        self.accuracy = {}
        self.label = label
        self.content = content
        self.rc = recorder("lwp_1x1cnn_dense_embed")
        self.embeds = []
        self.BATCH_SIZE = config.config["BATCH_SIZE"]

        self.MODEL_NAME = config.config["MODEL_NAME"]
        super().__init__(name)

    def _build(self, _input, training):
        # input, _updates, _vars = batch_norm(input,training = self.training, name= "bn", reuse= False)
        
        embed_num = min(256, _input.shape.as_list()[-1])

        _input = cnn(
            input=_input, name="cnn", filter_size=1, filter_num=embed_num, pool=False)

        # Global Average Pooling
        _input = tf.reduce_mean(_input, axis=[1, 2])
        embed = _input

        _input = sync_batch_norm(
            inputs=_input, is_training=training, scope="bn", offset=False, decay=0.995, epsilon=1e-5)
        #_input = batch_norm(input=_input, training=self.training, name="bn", offset=False, beta=0.95)

        CLASS_NUM = config.config["CLASS_NUM"]
        logits = dense(name="dense", input=_input,
                        hidden_num=CLASS_NUM, act_func=tf.identity)

        return logits, embed

    def record_by_id(self, layer_input, idx):
            
        if not self._is_init():
            new_layer_num = idx + 1
            assert new_layer_num <= self.layer_num + 1
            if self.layers[idx] is None:
                print("layer wrapper:", layer_input.name, layer_input.shape)
                if config.config["MULTIGPU_NUM"] == 1:
                    self.layers[idx] = layer_input
                else:
                    self.layers[idx] = [layer_input]
            else:
                self.layers[idx].append(layer_input)
            self.layer_num = max(self.layer_num, new_layer_num)
        return layer_input

    def get_embedding(self):
        return self.embeds

    def build(self):
        self.total_loss = 0

        _build = get_current_lwp_embeds_func()
        if _build is None:
            # Invoke default layer
            _build = self._build 
        _build = multi_gpu(_build, input_num=1, output_num=2, tot_output_num=2)


        @multi_gpu_decor(input_num=2, output_num=1, tot_output_num=2)
        def get_loss(logits, label):
            cont = build_on_logits(
                logits, label=label, label_smoothing=0.1)
            return cont.smooth_entropy_mean, cont.accuracy

        stop_gradient = multi_gpu(tf.stop_gradient, input_num=1, output_num=1, tot_output_num=1)
        with tf.device("/cpu:0"):
            with monitor_tf_objects() as self.mtf_obj:
                for i in range(self.layer_num):
                    with TowerContext(tower_name='', is_training=self.training):
                        with tf.variable_scope("layer_cnn_regression"):
                            with tf.variable_scope("layer_%d" % i):

                                if self.stop_gradient:
                                    _input = stop_gradient(self. layers[i])
                                else:
                                    _input = self. layers[i]

                                _input = sanitize_nan(_input)
                                logits, embed = _build(_input, training=self.training)
                                self.embeds. append(embed)
                                
                            self.layer_logits[i] = logits
                            self.container[i], self.accuracy[i] = get_loss(
                                logits, self.label)
                            self.accuracy[i] = tf.reduce_mean(self.accuracy[i])
                            #self.total_loss += self.container[i]
                            if i==0:
                                self.total_loss = self.container[i]
                            else:
                                self.total_loss = multi_plus(self.total_loss, self.container[i])
            with layer_record(name="lwp_embeds") as self.embed_lwp:
                for i in range(self.layer_num):
                    self.embed_lwp.record(self.embeds[i])

            print(self.mtf_obj.get_objects("update"))

            train_vars = self.mtf_obj.get_objects("train")
            # !!! Hack part Move it to config later
            if config.config["MODEL_NAME"] == "Imagenet_Denoise":
                BASE_LR = 0.1
                l2_reg = 0
                l2_pen_var = [tf.nn.l2_loss(v)
                              for v in train_vars if v.name.find('W') >= 0]
                self.init_step = 0
            elif config.config["MODEL_NAME"] == "Imagenet_Denoise_Resnet":
                BASE_LR = 0.1
                l2_reg = 1e-5
                l2_pen_var = [tf.nn.l2_loss(v)
                              for v in train_vars if v.name.find('W') >= 0]
                self.init_step = 0#40000
            elif config.config["MODEL_NAME"] in ["Imagenet_Resnet_Normal_P", "Imagenet_Resnet_Normal", "Imagenet_VGG16", "Imagenet_VGG16_AVG"]:
                BASE_LR = 0.1
                l2_reg = 1e-4
                l2_pen_var = [tf.nn.l2_loss(v)
                              for v in train_vars if v.name.lower().find('bias') < 0]
                self.init_step = 0#50000
            elif config.config["MODEL_NAME"] in ["Imagenet_Resnet_Adv", "Imagenet_Resnet_Adv_For_Analysis"]:
                BASE_LR = 0.1
                l2_reg = 1e-4
                l2_pen_var = [tf.nn.l2_loss(v)
                              for v in train_vars if v.name.lower().find('bias') < 0]
                self.init_step = 0  # 50000
            else:
                assert False
                BASE_LR = 0.1
                l2_reg = 1e-4
                l2_pen_var = [tf.nn.l2_loss(v)
                              for v in train_vars if v.name.lower().find('bias') < 0]
                self.init_step = 0  # 50000
            loss_reg = tf.add_n(l2_pen_var) * l2_reg
            self.total_loss = multi_plus(
                self.total_loss, multi_distribute(loss_reg))

            # !!!!!! Hack Value Please Modify that to 0 for final version
            global_step = tf.Variable(self.init_step, trainable=False)
            step_per_epoch = int(1e6 // self.BATCH_SIZE)
            boundaries = [3*step_per_epoch, 6*step_per_epoch, 9*step_per_epoch]
            self.total_batch_to_run = 12*step_per_epoch
            values = [BASE_LR, BASE_LR*0.1, BASE_LR*1e-2, BASE_LR*1e-3]
            learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries,
                                                                  values)

            # Please get trainable varaibles from the module
            with tf.control_dependencies(self.mtf_obj.get_objects("update")):

                #self.opt = multi_adam(
                #    1e-4, self.total_loss, var_list=self.mtf_obj.get_objects("train") )
                self.opt = multi_sgd(
                    learning_rate, self.total_loss, var_list=self.mtf_obj.get_objects("train"), global_step=global_step )
                #self.opt = tf.train.AdamOptimizer(
                #self.opt = tf.train.GradientDescentOptimizer(
                #                            learning_rate=1e-4).minimize(self.total_loss, 
                #                            var_list=train_vars, 
                #                            colocate_gradients_with_ops=True)

    def collect_statistics(self, iterator, iteration, sess, overwrite=False):
        def get_name(name):
            name = "%s/%s" % (self.MODEL_NAME, name)
            name = name.replace("//", "/")
            name = name.replace("/", "#")            
            return name

        all_var = self.mtf_obj.get_objects("all")

        def save_func():
            rst_dict = {}
            all_val = sess.run(all_var)
            for _var, _val in zip(all_var, all_val):
                name = get_name(_var.name)
                rst_dict[name] = _val
            return rst_dict
    
        def load_func():
            _dct = self.rc.load_record_by_id("%s#pretrained" % self.MODEL_NAME, 0)
            _dctn = {}
            for k in _dct.keys():
                _dctn[k.replace("##", "#")] = _dct[k]
            for _var in all_var:
                print("loading ",_var)
                name = get_name(_var.name)
                if name.find("moving_mean")>=0:
                    print(_dctn[name][0:1,0:1])
                    #print("Count Moving mean = 0: ",np.count_nonzero(_dctn[name] == 0.0))
                elif name.find("moving_var")>=0:
                    print(_dctn[name][0:1, 0:1])
                    #print("Count Moving variance = 1: ",np.count_nonzero(_dctn[name]==1.0))
                sess.run(tf.assign(_var, _dctn[name]))

        print(self.accuracy)
        exist_flag = self.rc.exists_record_by_id("%s#pretrained" % self.MODEL_NAME, 0,)
        print("Pre trained record exists: ", str(exist_flag))
        if not exist_flag or overwrite:
            if exist_flag:
                load_func()
                
            for i in range(self.init_step+1, self.total_batch_to_run):
                x, y = iterator()
                _acc, _loss, _ = sess.run([self.accuracy, self.total_loss, self.opt],
                                            feed_dict={self.content: x, self.label: y})

                if i % 100 == 0 and IS_DEBUG is True:
                    print("step %d: 1x1 layer wrapper regression: Loss %s, Acc %s" % (i, str(_loss), str(_acc)))

                if i%10000 == 0 and i!=0:
                    self.rc.save_record_by_id("%s#pretrained" %
                        self.MODEL_NAME, 0, save_func, overwrite= True)

            if i == self.total_batch_to_run:
                self.rc.save_record_by_id("%s#pretrained" %
                    self.MODEL_NAME, 0, save_func, overwrite=True)
        load_func()


        return 


def get_current_layer_wrapper():
    return config.config["ENV_LAYER_RECORD"][-1]


def get_layer_wrapper_by_name(name):
    return LWP_LOOPUP_DICT[name]


def reset():
    __init__()

def __init__():
    global LWP_LOOPUP_DICT, LWP_UID
    LWP_LOOPUP_DICT = {}
    LWP_UID = 0
    config.config["ENV_LAYER_RECORD"] = [layer_base()]


if "FIRST_RUN_FLAG" in globals():
    pass
else:
    FIRST_RUN_FLAG = True
    __init__()
