import tensorflow as tf
import numpy as np

import multigpu
import classifiers
import dataprep
import config
from record import recorder
from layer_supervisor import layer_1x1cnn_dense_embed, layer_record
from classifier_models.vgg.vgg16 import Vgg16
from attacks import create_attack, _wrap_run_atk_with_switch, add_preprocess_func, add_restore_func
from tensorpack.tfutils.tower import TowerContext
import attacks
import os
import sys

from tensorflow.python import debug as tf_debug
from utils import save_pair_diff
from utils import np_l2_dist,np_linf_dist
import argparse

from redirect_output import overide_write
from testacc import get_stat_x, filter_dict

overide_write("try_internal_embed.log")
print("command arguments: ", sys.argv)

parser =  argparse.ArgumentParser(description=  "Internal Embedding Restriction")
parser.add_argument("--task", type=str, choices=["Attack","TrainEmbed","CollectStat"], help="First train embed layer, then attack.", default="TrainEmbed")
parser.add_argument("--model", type=str,
                    choices=["Imagenet_Resnet_Normal", "Imagenet_Resnet_Normal_P", "Imagenet_Denoise", "Imagenet_Denoise_Resnet", "Imagenet_Resnet_Adv", "Imagenet_VGG16", "Imagenet_Resnet_Adv_For_Analysis", "Imagenet_MobileNet", "Imagenet_VGG19", "Imagenet_DenseNet"], default="Imagenet_Resnet_Normal")
parser.add_argument("--targetlayer",  help="Which layer to attack. Only in attack mode.", default="[1]")
parser.add_argument("--scale", type=float,
                    help="The scale of noise. Only in attack mode.", default=0.03)
parser.add_argument("--images", type=str, help="The folders to save the images.", default="embeds")
parser.add_argument("--gpus", type=int, help="The number of gpus to use.", default=None)
parser.add_argument("--overwriteattack", action="store_true", help="Attack the batch even it is in the database")
parser.add_argument("--barrier", type=str, help="Which barrier function to use", default="polynomial")
parser.add_argument("--embed", help="Attack the embedding dimensionally reduced from internal", action="store_true")
parser.add_argument("--dynamic_step_size", help="automatically calibrate step size", type=bool, default=True)
parser.add_argument("--feature_smooth",
                    help="weight of smoothness of feature layer", type=float, default=0.)
parser.add_argument("--internal", help="Attack the internal", action="store_true")
parser.add_argument(
    "--stat", help="Attack the internal", default="neuron_min_max")
parser.add_argument("--targetattack", action="store_true")
parser.add_argument(
    "--pgdrand", help="Attack the internal", action="store_true")
parser.add_argument("--pgdpercent", help="set the scale to the percentage of pgd attack", type=float)
parser.add_argument("--smstep", type=int, default=0)
parser.add_argument("--fsstrength", type=int, default=0)
parser.add_argument("--fsscale", type=float, default=1.5)
parser.add_argument("--nonba", action="store_true")
parser.add_argument("--recname", type=str, default="")
parser.add_argument(
    "--novgg", help="Attack the internal", action="store_true")
parser.add_argument(
    "--batches", help="", type=int, default=2)
parser.add_argument("--denoise_first", type=int, default=0)
parser.add_argument("--resnetmode", type=str, default="Main",  choices=["Main","None","Shortcut"])
args = parser.parse_args()
task_id = args.task#"TrainEmbed"
barrier = args.barrier
novgg = args.novgg
resnetmode = str(args.resnetmode)
stat = args.stat
batches = int(args.batches)
smstep = int(args.smstep)
fsstrength = int(args.fsstrength)
denoise_first = args.denoise_first
fsscale = float(args.fsscale)
target_attack = bool(args.targetattack)
nonba = bool(args.nonba)
feature_smooth = float(args.feature_smooth)
recname = str(args.recname)
pgdpercent = args.pgdpercent
assert stat in ["channel_min_max", "neuron_min_max", "neuron_variance", "neuron_gaussian"]
dynamic_step_size_enable = bool(args.dynamic_step_size)
pgd_rand = args.pgdrand
alllayers= eval(args.targetlayer)#[0,1,2]
def formatstr():
    
    if attack_embed:
        prefix = "embedsatk_"
    elif attack_internal:
        prefix = "internal_"
    else:
        assert False
    if target_attack:
        prefix += "#targetatk_"
    if stat == "channel_min_max":
        prefix += "#stcmm_"
    elif stat == "neuron_min_max":
        prefix += "#stnmm_"
    elif stat == "neuron_variance":
        prefix += "#stnv_"
    elif stat == "neuron_gaussian":
        prefix += "#stgauss_"

    if feature_smooth != 0.:
        prefix +="featsmth%.2e_" %feature_smooth


    if denoise_first:
        prefix += "fo%d_"%denoise_first

    if dynamic_step_size_enable is True:
        step_size = dynamic_step_size
    else:
        step_size = defined["step_size"]
    alllayers_str = [str(i) for i in alllayers]
    if args.novgg:
        vggname= "#resmd%s" % config.config["resnet_record_mode"]
    else:
        vggname="#vgg"
    if barrier == "polynomial":
        return prefix+"mn_%s%s#bt_poly#l%s#pgdsc_%.2e#pgdperc%.2e#bpen%.2f#bpow%.2f#dynamss%s#ss_%.2e#ns_%d" % \
            (model_name, vggname, "-".join(alllayers_str), scale, pgdpercent,
            defined["bound_penalty_multiplier"], defined["barrier_power"],
             str(defined["dynamic_step_size"]), step_size, defined["num_steps"])
    elif barrier == "relu":
        return prefix+"mn_%s%s#bt_relu#l%s#pgdsc_%.2e#pgdperc%.2e#bpen%.2f#lkn%.2e#dynamss%s#ss_%.2e#ns_%d" % \
            (model_name, vggname, "-".join(alllayers_str), scale, pgdpercent,
             defined["bound_penalty_multiplier"], defined["leakiness"],
             str(defined["dynamic_step_size"]), step_size, defined["num_steps"])
    else:
        assert False


def formatstr_pgd():
    scale = defined_pgd["epsilon"] /255.0
    return "pgdcompare_"+"mn_%s#sc_%.2e#ss_%.2f#ns_%d_#rs%s" % \
        (model_name, scale, defined_pgd["step_size"],
         defined_pgd["num_steps"], defined_pgd["random_start"])

def formatstr_semantic():
    return "semanticcompare_"+"mn_%s#sp_%d" % (model_name, smstep)

def formatstr_featurespace():
    return "featurespacecompare_"+"mn_%s#fsstrength_%d_scale%.2e" % (model_name, fsstrength, fsscale)

def SelfFeatureInduce_Attack(image_ph, label_ph, sess, attack_name, model_name, **param):
    global defined
    param = {"step_size": 0.1, "num_steps": 2000, 
             "target_layer": 1, "bound_penalty_multiplier": 1, "stat_type": stat, "feature_smooth": feature_smooth, ** param}
    if attack_internal:
        param["collect_cache_name"] = "try_internal#%s" % (model_name)
    elif attack_embed:
        param["collect_cache_name"] = "try_internal_embed#%s" % (model_name)
    defined = param
    param["dynamic_step_size"] = False
    from attack_algs.self_induce_feature.internal_barrier_mult \
        import INT_BARR_Attack as fs_attack
    attack = fs_attack(image_ph, label_ph, ** param)

    def collect_func(iterator, iteration):
        attack.collect_statistics(iterator, iteration, sess)

    def build_attack(model_container):
        attack.build(model_container)

    def run_attack(x, y, target):
        return attack.attack(x, y, sess, target)

    return build_attack, run_attack, collect_func, attack


def Semantic_Attack(image_ph, label_ph, sess, attack_name, method="color", target_attack=None, steps=500):
    if method == "color":
        from attack_algs.semantic.cadv.sm_color import semantic_color as semantic_atk
        params = {
            "IMAGENET": {"model_path": os.path.join(".", "attack_algs", "semantic", "cadv", "latest_net_G.pth")},
            "CIFAR10": {"model_path": os.path.join(".", "attack_algs", "semantic", "cadv", "latest_net_G.pth")},
        }
    else:
        assert False
    param = params[config.config["DATA_SET"]]

    attack = semantic_atk(image_ph, label_ph, target_attack=target_attack, steps=steps, ** param)
    add_preprocess_func(attack_name, attack.pre_built)

    def build_attack(model_container):
        attack.build(model_container)

    def run_attack(x, y, target=None):
        return attack.attack(x, y, sess, target)
    return build_attack, run_attack


def FeatureSpace_Attack(image_ph, label_ph, sess, attack_name, method, strength, target_attack=None, scale = None):
    assert strength in [1, 2, 3]
    if method == "augmentation":
        from attack_algs.feature_space.fs_attack import feature_attack as fs_attack
        params = {
            "IMAGENET": {
                3: {"setting_name": "IMAGENET"},
                2: {"setting_name": "IMAGENET_SHALLOW"},
                1: {"setting_name": "IMAGENET_SHALLOWEST"}
            },
            "CIFAR10": {"setting_name": "CIFAR10"},
        }
    else:
        assert False
    param = params[config.config["DATA_SET"]]
    if config.config["DATA_SET"] == "IMAGENET":
        param = param[strength]
    param["encoder_path"] = os.path.join(
        "attack_algs", "feature_space",  "vgg19_normalised.npz")

    attack = fs_attack(image_ph, label_ph,
                       target_attack=target_attack, scale = scale, ** param)

    add_preprocess_func(attack_name, attack.pre_built)
    #add_restore_func(attack_name, attack.restore)

    def build_attack(model_container):
        attack.build(model_container)

    def run_attack(x, y, target=None):
        return attack.attack(x, y, sess, target)
    return build_attack, run_attack, attack.restore

def PGD_Attack(image_ph, label_ph, sess, attack_name, distance, epsilon, target_attack):
    global defined_pgd
    if distance == "Linf":
        from attack_algs.pgd_attack import LinfPGDAttack as PGD
        epsilon = 255.0 * epsilon
        params = {
            "IMAGENET": {"epsilon": epsilon, "num_steps": 1000, "step_size": epsilon / 4, "random_start": pgd_rand},
            "CIFAR10": {"epsilon": epsilon, "num_steps": 1000, "step_size": epsilon / 4, "random_start": pgd_rand},
        }
    else:
        assert False
    param = params[config.config["DATA_SET"]]
    defined_pgd = param
    attack = None

    def build_attack(model_container):
        nonlocal attack
        attack = PGD(image_ph, label_ph, model_container, target_attack=target_attack, ** param)

    def run_attack(x, y, target):
        return attack.generate(x, y, sess, target)
    return build_attack, run_attack



model_name = args.model #"Imagenet_Resnet_Normal"#"Imagenet_Denoise"
attacks.Handler["Self_Feature_Induce_Attack"]["parameter"] = {}
print("Begin to use models for testing attacks: ", model_name)

config.set_data_set("IMAGENET")

dynamic_step_size = None
gpu_nums = args.gpus
if task_id == "Attack":
    config.config["MULTIGPU_NUM"] = 1 if gpu_nums is None else gpu_nums
    config.config["BATCH_SIZE"] = 8
    BATCH_NUM = batches
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

if model_name == "Imagenet_Denoise":
    confidence = 50
elif model_name in ["Imagenet_Resnet_Normal", "Imagenet_Resnet_Normal_P"]:
    confidence = 300
else:
    if target_attack:
        confidence=300
    else:
        confidence = 50

config.config["confidence"] = confidence

def idk_atk(x, y):
    return x


scale = float(args.scale)
max_step_size = scale/4

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
#config1.gpu_options.per_process_gpu_memory_fraction = 1
with tf.Session(config=config1) as sess:

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
    target_ph = tf.placeholder(tf.int64, shape=[None], name="targetlabel")

    """Second step : create attack so that it can bind to classifiers and do some processing"""
    #for atk_name in ["PGD_Linf"]:  # Handler: "Self_Feature_Induce_Attack",
    #    print("Building attacks:", atk_name)
    #    create_attack(content, label,  atk_name, sess)

    #atk_func = config.config["atk_func"][atk_name]

    """Third step : build classifier"""
     # tf.get_variable("training", initializer=True)
    #with tf.device("/cpu:0"):
    atk_name = "Self_Feature_Induce_Attack"

    if task_id in ["Attack", "CollectStat"]:
        #target_layer = int(args.targetlayer)
        if attack_embed:
            default_wrapper_name = "lwp_embeds"
        elif attack_internal:
            default_wrapper_name = "classifier"
        else:
            assert False

        if barrier == "relu":
            multiplier = 1e6
        else:
            multiplier = 10000

        if args.novgg:
            bound_model_name = model_name
        else:
            bound_model_name = "Imagenet_VGG16"
        

        atk_build, atk_run, atk_collect, atk_class = SelfFeatureInduce_Attack(
            content, label, sess, atk_name, bound_model_name, bound_penalty_multiplier=multiplier,
            step_size=0.1, num_steps=80, default_wrapper_name=default_wrapper_name,
            target_layer=alllayers,
            barrier_power=200, leakiness=1e-5, barrier_type=barrier, stat_type=stat, target_attack=target_attack, max_step_size=max_step_size, denoise_first_only=denoise_first)

        pgd_build, pgd_run = PGD_Attack(
            content, label, sess, "PGD_Linf", "Linf", scale, target_attack=target_attack)
        if smstep>0:
            sm_build, sm_run = Semantic_Attack(
                content, label, sess, "Semantic_Color_Attack", target_attack=target_attack, steps=smstep)
        if fsstrength>0:
            fs_build, fs_run, fs_restore = FeatureSpace_Attack(
                content, label, sess, "Feature_Space_Argmentation", method="augmentation", target_attack=target_attack, strength=fsstrength, scale=fsscale)

    

    ## reset the filter_record_array
    
    
    if args.novgg:
        if model_name=="Imagenet_Resnet_Adv_For_Analysis":
            config.config["filter_record_array"] = config.dummy_filter_record_array
        else:
            config.config["resnet_record_mode"] = resnetmode#"Main"
        with layer_record("classifier") as lwp:
            model_container, _ = build_classifier(content, label, target_ph)
    else:
        config.config["filter_record_array"] = config.dummy_filter_record_array
        model_container, _ = build_classifier(content, label, target_ph)
        vgg16_class = Vgg16(os.path.join("checkpoints", "vgg16.npy"), pooling="max")
        with layer_record("classifier") as lwp:
            vgg16_class.build(content)
        

    if task_id in ["Attack", "CollectStat"] :
        atk_build(model_container)
        pgd_build(model_container)
        if smstep>0:
            sm_build(model_container)
        if fsstrength>0:
            fs_build(model_container)

        if attack_embed:
            atk_lwp = lwp.embed_lwp
        elif attack_internal:
            atk_lwp = lwp

    sess.run(tf.global_variables_initializer())
    print("Restoring parameters")
    restore_classifier(sess)
    
    if attack_embed:
        if task_id == "TrainEmbed":
            lwp.collect_statistics(next_batch, 300000, sess, overwrite=True)
        else:
            lwp.collect_statistics(next_batch, 300000, sess)


    if task_id == "CollectStat":
        atk_collect(next_batch, int(1.2e6) // BATCH_SIZE)

    if task_id == "Attack":
        if not nonba:
            atk_collect(next_batch, 5e4 // BATCH_SIZE)
            atk_run = _wrap_run_atk_with_switch(sess, atk_run, atk_name)
        if smstep>0:
            sm_run = _wrap_run_atk_with_switch(
                sess, sm_run, "Semantic_Color_Attack")
        if fsstrength>0:
            fs_run = _wrap_run_atk_with_switch(
                sess, fs_run, "Feature_Space_Argmentation")

        rec1 = recorder("select_test_sample")
        if recname=="":
            recname = "attack_samples"
        rec = recorder(recname)#502")

        internal_data = lwp.get_dict_layers("internal_")
        #internal_data.update(
        #    lwp.embed_lwp.get_dict_layers("embed_"))

        # Sync normal data across all batch
        for idx in range(BATCH_NUM):
            def get_np_data():
                x_batch, y_batch = next_batch()
                id_asgn = np.array(
                    [idx*BATCH_SIZE + i for i in range(BATCH_SIZE)])
                return {"x_batch": x_batch, "y_batch": y_batch, "idx": id_asgn}
            rec1.save_record_by_id("normal_data", idx, get_np_data)

        # Record different internal data for each model
        for idx in range(BATCH_NUM):
            def get_np_data():
                start = idx * BATCH_SIZE
                end = (idx+1) *BATCH_SIZE
                _data = rec1.load_record_by_num("normal_data", start, end)
                x_batch, y_batch = _data["x_batch"], _data["y_batch"]   
                id_asgn = np.array(
                    [idx*BATCH_SIZE + i for i in range(BATCH_SIZE)])

                target_label = np.random.randint(low=0, high=config.config["CLASS_NUM"], size=(BATCH_SIZE,))
                return {"x_batch": x_batch, "y_batch": y_batch, "idx": id_asgn, "target_label":target_label}
            rec.save_record_by_id("normal_data_%s" % model_name, idx, get_np_data)
        
        record_name_funcs = [formatstr_pgd ] 
        atk_runs = [pgd_run]
        names = ["pgd"]
        if not nonba:
            record_name_funcs .append(formatstr)   # formatstr_semantic,
            atk_runs. append(atk_run)  # , sm_run
            names.append("feature")  # , "semantic"""

        if smstep > 0:
            record_name_funcs.append(formatstr_semantic)
            atk_runs.append(sm_run)
            names.append("semantic")



        #record_name_funcs = [formatstr_semantic]
        #atk_runs = [sm_run]
        #names = ["semantic"]
        if fsstrength>0:
            record_name_funcs .append(formatstr_featurespace) 
            atk_runs.append(fs_run)
            names.append("featurespace")
        #moving_range = sess.run(atk_class.moving_range)
        #print("moving_range",moving_range)
        if "featurespace" in names:
            fs_restore(sess)

        for record_name_func, atk_run, name in zip(record_name_funcs, atk_runs, names):
            for idx in range(BATCH_NUM):
                start = idx *BATCH_SIZE
                end = (idx+1) *BATCH_SIZE
                _data = rec.load_record_by_num(
                    "normal_data_%s" % model_name, start, end)
                x_nat, y_batch, target = _data["x_batch"], _data["y_batch"], _data["target_label"]   
                id_asgn = np.array(
                    [idx*BATCH_SIZE + i for i in range(BATCH_SIZE)])
                print(x_nat.shape, y_batch.shape)
                if  name == "feature":
                    ### !!!!! This is a hack, should not appear in stable version, It overides the step size setting and monitor it.
                    pgd_data = rec.load_record_by_num(pgd_name, start, end)
                    scale_lst = [] 
                    if stat != "neuron_gaussian":
                        if args.novgg:
                            bound_model_name = model_name
                        else:
                            bound_model_name = "Imagenet_VGG16"
                        _stats = get_stat_x(sess, bound_model_name, atk_class.x_ph,
                                pgd_data["x_batch"], atk_class.y_ph, pgd_data["y_batch"], x_nat, atk_lwp, model_container, internal = attack_internal, embed = attack_embed, typs=[stat])
                        _stats = _stats.average()
                        for i in alllayers:
                            scale_lst .append( list(filter_dict(filter_dict(_stats, stat), "#%d" % i).values())[0])
                    else:
                        normal_quantile = sess.run(atk_class.quantiles, feed_dict={
                                                   atk_class.x_ph: x_nat})
                        pgd_quantile = sess.run(atk_class.quantiles, feed_dict={
                                                atk_class.x_ph: pgd_data["x_batch"]})
                        for i in range(len(alllayers)):
                            scale_lst.append( np_linf_dist(normal_quantile[i], pgd_quantile[i]) )

                    print("pgd scale", scale_lst)
                    scale_lst = [scale * pgdpercent for scale in scale_lst]
                    print("feature scale", scale_lst)
                    atk_class.set_scales(sess, scale_lst)

                    if dynamic_step_size_enable:
                        sess.run(atk_class.atk_init, feed_dict={atk_class.x_ph: x_nat, atk_class.y_ph: y_batch})
                        dynamic_step_size = atk_class.binary_dynam_learn_rate_search(x_nat, y_batch, sess, target)
                        dynamic_step_size = min(max_step_size, dynamic_step_size)
                        atk_class.step_size = dynamic_step_size

                if idx == 0:
                    record_name = record_name_func()             
                    if name == "pgd":
                        pgd_name = record_name   
                def atk_func():
                    print("Attack %s, iteration %d" % (record_name, idx))

                    _data = rec.load_record_by_num(
                        "normal_data_%s" % model_name, start, end)
                    x_nat, y_batch, target = _data["x_batch"], _data["y_batch"], _data["target_label"]
                    id_asgn = np.array(
                        [idx*BATCH_SIZE + i for i in range(BATCH_SIZE)])
                    print(x_nat.shape, y_batch.shape)

                    x_batch = atk_run(x_nat, y_batch, target)

                    
                    if target_attack:
                        accuracy = model_container.target_accuracy
                    else:
                        accuracy = model_container.accuracy

                    _int_data, _pred, _acc = sess.run([internal_data, model_container.prediction,
                                            accuracy], feed_dict=atk_class._input_dec(x_batch, y_batch, target))


                    image_path = args.images
                    for i in range(BATCH_SIZE):
                        save_pair_diff(x_nat[i], x_batch[i],
                                    path=os.path.join(image_path, record_name, "%d.png" % (idx*BATCH_SIZE+i)), dynamic=True)
                    print("Label:      ", y_batch)
                    print("Prediction: ", _pred)
                    print("Accuracy: ", _acc)
                    for dist in [np_l2_dist, np_linf_dist]:
                        print("Pixel Distance :", str(dist),
                              dist(x_nat , x_batch, scale=255.0))

                    #### Analyze internal given certain internal layer

                    """if attack_embed:
                        key = "embed_Layer_%d" % target_layer
                    elif attack_internal:
                        key = "internal_Layer_%d" % target_layer

                    for dist in [np_l2_dist, np_linf_dist]:
                        _int_nat = _data[key]
                        _int_adv = _int_data[key]
                        #print("Internal Distance :", str(dist),
                        #      dist(_int_nat , _int_adv , scale = moving_range))

                        #print(_int_nat[2]-_int_data_1[2])

                    ## do not record intermiate data for saving space"""
                    _int_data = {}

                    return {"x_batch": x_batch, "y_batch": y_batch, "target_label":target, "idx": id_asgn, **_int_data}
                rec.save_record_by_id(
                    record_name, idx, atk_func, overwrite=args.overwriteattack)
