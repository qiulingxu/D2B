from classifiers import Handlers, init_model
from record import recorder
import tensorflow as tf
import numpy as np
import dataprep
import config
import os
import json
from utils import np_l2_dist, np_linf_dist
import time
from redirect_output import overide_write
import layer_supervisor
from tensorflow.distributions import Normal
from layer_supervisor import layer_1x1cnn_dense_embed, layer_record
from classifier_models.vgg.vgg16 import Vgg16

def filter(lst, func):
    rst = []
    for elem in lst:
        if func(elem):
            rst.append(elem)
    return rst


def filter_item(lst, model_name=None, layer=None, internal=None, embed=None, pgd=None, scale=None, alllayers=None, stattype=None, pgdscale=None, fs=None):
    if model_name is not None:
        lst = filter(lst, lambda x: x.find("mn_%s" % model_name) >= 0)
    if layer:
        lst = filter(lst, lambda x: x.find("#tl_%d" % layer) >= 0)
    if embed:
        lst = filter(lst, lambda x: x.find("embedsatk_") >= 0)
    if internal:
        lst = filter(lst, lambda x: x.find("internal_") >= 0)
    if pgd:
        lst = filter(lst, lambda x: x.find("pgdcompare") >= 0)
    if scale:
        lst = filter(lst, lambda x: x.find("sc_%.2e" % scale)>=0)
    if pgdscale:
        lst = filter(lst, lambda x: x.find("pgdsc_%.2e" % pgdscale) >= 0)
    if alllayers:
        lst = filter(lst, lambda x: x.find("#alayers")>=0 or x.find("#l")>=0)
    if stattype:
        lst = filter(lst, lambda x: x.find("#st%s" % stattype) >= 0)
    if fs:
        lst = filter(lst, lambda x: x.find("featurespace") >= 0)
    return lst


class dict_list:
    def __init__(self):
        self. dict = {}
    
    def add(self, key, val):
        if key in self.dict:
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]
        return self.dict
    
    def add_list(self, key, val_lst):
        for lv in val_lst:
            self.add(key, lv)
        
        

    def get_dict(self):
        return self.dict

    def average(self):
        rst = {}
        for key in self.dict.keys():
            lst = self.dict[key]
            rst[key] = np.mean(lst)
        return rst


def get_by_name(dct, name1):
    for key in dct.keys():
        if key.find(name1)>=0:
            return dct[key]
    assert False, "Name not found in %s" % (str(dct.keys()))
    return None

def init_guassian():
    global mean_ph, std_ph, dist, int_ph, quantile_tf
    mean_ph = tf.placeholder(tf.float32, shape=[1, None, None, None])
    std_ph = tf.placeholder(tf.float32, shape=[1, None, None, None])
    dist = Normal(loc=mean_ph, scale=std_ph)
    int_ph = tf.placeholder(tf.float32, shape=[None, None, None, None])

    quantile_tf = tf.clip_by_value(dist.cdf(
        int_ph), 0.0, 1.0)


def filter_dict(dct, string):
    rst = {}
    for k in dct.keys():
        if k.find(string) >=0:
            rst[k] = dct[k]
    return rst

def get_stat_x(sess, model_name, x_ph, x, y_ph, y, x_nat, lwp, model_container, internal=None, embed=None, rst=None, typs=None, target_ph =None, target_label=None):

    if rst is None:
        rst = dict_list()

    _pred, _acc, _loss_y, _loss = sess.run([model_container.prediction,
                            model_container.accuracy,
                            model_container.target_loss,
                            model_container.target_loss_sum], feed_dict={
        x_ph: x, y_ph: y})

    rst.add("accuracy", _acc)
    rst.add("attack_loss", _loss)
    rst.add_list("attack_loss_y", _loss_y)

    if target_label is not None:
        _loss, _tacc = sess.run([model_container.target_attack_loss_sum, model_container.target_accuracy], feed_dict={
            x_ph: x, y_ph: y, target_ph: target_label})
        rst.add("target_attack_loss", _loss)
        rst.add("target_accuracy", _tacc)

    rec_1 = recorder("self_induced_attack_intbarr", readonly=True)
    rn_1 = rec_1.list_record()
    if model_name != "Imagenet_Resnet_Adv_For_Analysis":
        layers = 5
    else:
        layers = 17
    acts = []
    for layer in range(0, layers):
        acts.append(lwp.get_layer_by_id(layer))

    filter_record_array = config.config["filter_record_array"]
    eps = 1e-9
    _acts_adv = sess.run(acts, feed_dict = {x_ph: x})
    _acts_nat = sess.run(acts, feed_dict = {x_ph: x_nat})
    for layer in range(0, layers):
        for typ in typs:#,"channel_min_max", "neuron_min_max", "neuron_variance"]:
            if internal:
                tp = 1
            elif embed:
                tp = 0

            if tp == 0:
                key_1 = "try_internal_embed#%s#%d" % (
                    model_name, layer)
            elif tp == 1:
                key_1 = "try_internal#%s#%d" % (
                    model_name, layer)
            key_1 += "_" + typ

            if key_1 in rn_1:
                try:
                    _record_1 = rec_1.load_record_by_id(key_1, 0)
                except Exception as e:
                    print(e)
                    continue
                _int_nat = _acts_nat[layer]
                _int_adv = _acts_adv[layer] 
                if typ == "neuron_variance":
                    _cnt = get_by_name(
                        _record_1, "cnt")
                    moving_range = (
                        3 * 2) * np.sqrt(get_by_name(_record_1, "var") / _cnt) + eps
                    moving_range = filter_record_array(moving_range, layer)
                elif typ == "neuron_min_max" or typ == "channel_min_max":
                    moving_range = get_by_name(
                        _record_1, "max") - get_by_name(_record_1, "min") + 1e-9
                    moving_range = filter_record_array(moving_range, layer)
                elif typ == "neuron_gaussian":
                    _cnt = get_by_name(
                        _record_1, "cnt")
                    mean = get_by_name(_record_1, "mean") / _cnt + eps
                    std = np.sqrt(get_by_name(_record_1, "var") / _cnt - np.square(mean) ) + eps
                    std = filter_record_array(std, layer)
                    mean = filter_record_array(mean, layer)
                    #print("mean.shape", mean.shape)
                    #print("std.shape", std.shape)
                    nat_quantile = sess.run(quantile_tf, feed_dict={
                                            mean_ph: mean, std_ph: std, int_ph: _int_nat})
                    adv_quantile = sess.run(quantile_tf, feed_dict={
                        mean_ph: mean, std_ph: std, int_ph: _int_adv})
                else:
                    assert False

                # np_l2_dist,"l2",
                for dist, name1 in zip([np_linf_dist, np_l2_dist], ["linf", "l2"]):

                    if typ == "neuron_gaussian":
                        rst.add("%s_%s" % (key_1, name1), dist(
                            nat_quantile, adv_quantile))
                        #for p in range(16):
                        #    print(dist(
                        #        nat_quantile[p:p+1], adv_quantile[p:p+1]), end="\t")
                        #print("##\n")
                    else:
                        rst.add("%s_%s" % (key_1, name1), dist(
                            _int_nat, _int_adv, scale=moving_range))
    for dist, name1 in zip([np_l2_dist, np_linf_dist], ["l2", "linf"]):
        rst.add("%s_%s" % ("pixel", name1), dist(
            x_nat, x, scale=255.0))
    return rst

if __name__ == "__main__":
    overide_write("testaccresult.txt", "a")
    #internal_embed_attack_sample
    rec_instance = recorder(
        "internal_embed_attack_sample_mult_vgg_502", readonly=True) #502 # 424")
    records = rec_instance.list_record()
    #batchsize = 8
    #batches = 14
    use_vgg = True
    batches = 10
    batchsize = 10
    # Handlers.keys():
    # "Imagenet_Resnet_Adv","Imagenet_Denoise_Resnet" Imagenet_Resnet_Normal
    for model_name in [ "Imagenet_Resnet_Adv"]: #"Imagenet_Resnet_Normal_P"]:  #
        if model_name == "Imagenet_Denoise":
            confidence = 50
        elif model_name == "Imagenet_Resnet_Normal" or model_name == "Imagenet_Resnet_Normal_P":
            confidence = 300
        else:
            confidence = 50

        print("Now testing model for %s" % model_name)
        print(records)
        records = filter_item(records, model_name)
        #records = filter_item(records, model_name, pgd=True) + filter_item(records, model_name, stattype="gauss", pgdscale=float(0.04))
        #records = filter_item(records, model_name, fs=True)
        #records = json.loads(open("humanprefvssucc_adv.json", "r").read())
        print(records)
        tf.reset_default_graph()
        layer_supervisor.reset()
        init_guassian()
        typs = ["neuron_gaussian", "neuron_min_max"]
        config1 = tf.ConfigProto()
        config1.gpu_options.allow_growth = True
        with tf.Session(config=config1) as sess:
            print("Begin to check models: ", model_name)
            info = Handlers[model_name]
            config.set_data_set(info["dataset"])
            if use_vgg:
                config.config["resnet_record_mode"] = "None"
            else:
                config.config["resnet_record_mode"] = "Main"
            print("Preparing data:", info["dataset"])
            dataprep.init_data("eval")

            print("Building models:", info["function"])
            init_model(model_name, confidence=confidence)

            BATCH_SIZE = config.config["BATCH_SIZE"]
            IMAGE_SIZE = config.config["IMAGE_SIZE"]
            build_classifier = config.config["build_classifier"]
            restore_classifier = config.config["restore_classifier"]
            next_batch = config.config["next_batch"]

            content = tf.placeholder( tf.float32, shape=[None, *IMAGE_SIZE], name='content')
            label = tf.placeholder(tf.int64, shape=[None], name="label")
            target_ph = tf.placeholder(tf.int64, shape=[None], name="label")
            if use_vgg:
                model_container, model_class = build_classifier(
                    content, label, target_ph)
                vgg16_class = Vgg16(os.path.join(
                    "checkpoints", "vgg16.npy"), pooling="max")
                with layer_record("classifier") as lwp:
                    vgg16_class.build(content)
                analyze_model = "Imagenet_VGG16"
            else:
                with layer_1x1cnn_dense_embed("classifier", content, label, False, stop_gradient=False) as lwp:
                    model_container, model_class = build_classifier(content, label, target_ph)
                analyze_model = model_name

            restore_classifier(sess)

            print("Restoring parameters")
            #model_name1 = ""
            for record in records:
                print("Index",rec_instance.get_or_create_record(record)["Index"])
                result = dict_list()
                for test_batch in range(batches):
                    start = batchsize*test_batch
                    end = batchsize*(test_batch+1)
                    #_data_nat = rec_instance.load_record_by_id(
                    #    "normal_data_%s" % model_name, test_batch)
                    _data_nat = rec_instance.load_record_by_num(
                        "normal_data_%s" % model_name, start, end)
                    x_batch_nat = _data_nat["x_batch"]
                    try:
                        _data_adv = rec_instance.load_record_by_num(
                            record, start, end)
                        
                        x_batch, y_batch= _data_adv["x_batch"], _data_adv["y_batch"]
                        if "target_label" in _data_adv:
                            target = _data_adv["target_label"]
                        else:
                            target = None
                    except Exception as e:
                        print(e)
                        continue

                    #result = get_stat(sess, rec_instance, record, model_name )
                    result = get_stat_x(sess, analyze_model, content, x_batch, label, y_batch, x_batch_nat, lwp,
                                        model_container, internal=True, rst=result, typs=typs, target_ph=target_ph, target_label=target)
                if "attack_loss_y" in result.dict:
                    for idx, lv in enumerate(result.dict["attack_loss_y"]):
                        print("sample %d lossval %.2f" % (idx, lv))
                result = result.average()
                print(record)
                for k in result.keys():
                    print("\t%s\t%s"%(k,str(result[k])))
                #time.sleep(5)
