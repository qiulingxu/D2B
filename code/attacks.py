import tensorflow as tf
import config
import os

from tqdm import tqdm
from layer_supervisor import layer_record


def add_preprocess_func(attack_name, func):
    """ 
    preprocess function should accept two parameters : image , label
        and return one modified image
    """
    assert not config.config["create_attack_finish_flag"], "It is not allowed to add preprocess for attack after building models"

    if "pre_built_attack" not in config.config:
        config.config["pre_built_attack"] = \
            {"switch": tf.get_variable("attack_switch", initializer=0),
             "list": []}
    config.config["pre_built_attack"]["list"].append(
        {"name": attack_name,
         "function": func}
    )


def add_restore_func(attack_name, func):
    config.config["atk_func"][attack_name]["restore"] = func


def add_collect_func(attack_name, func, iteration):
    config.config["atk_func"][attack_name]["collect"] = {
        "iteration": iteration, "func": func}


def build_all(model_container):
    for atk_name in config.config["atk_func"].keys():
        atk_func = config.config["atk_func"][atk_name]
        atk_func["build"](model_container)


def init_all(sess):
    for atk_name in config.config["atk_func"].keys():
        atk_func = config.config["atk_func"][atk_name]
        atk_func["restore"](sess)


def collect_all(iterator):
    for atk_name in config.config["atk_func"].keys():
        atk_func = config.config["atk_func"][atk_name]
        if "collect" in atk_func:
            clct_dict = atk_func["collect"]
            print("Collect data for %s: " % atk_name)
            clct_dict["func"](iterator, clct_dict["iteration"])


def PGD_Attack(image_ph, label_ph, sess, attack_name, distance, **kwargs):
    if distance == "Linf":
        from attack_algs.pgd_attack import LinfPGDAttack as PGD
        params = {
            "IMAGENET": {"epsilon": 8.0, "num_steps": 10, "step_size": 1.0, "random_start": True},
            "CIFAR10": {"epsilon": 8.0, "num_steps": 10, "step_size": 2.0, "random_start": True},
        }
    else:
        assert False
    param = params[config.config["DATA_SET"]]
    param.update(kwargs)
    attack = None

    def build_attack(model_container):
        nonlocal attack
        attack = PGD(image_ph, label_ph, model_container, ** param)

    def run_attack(x, y, target = None):
        return attack.generate(x, y, sess, target)
    return build_attack, run_attack


def Channel_Attack(image_ph, label_ph, sess, attack_name, channel="HSV"):
    if channel == "HSV":
        from attack_algs.channel_attack import hsv_attack as channel_atk
        params = {
            "IMAGENET": {},
            "CIFAR10": {},
        }
    elif channel == "RGB":
        from attack_algs.channel_attack import rgb_attack as channel_atk
        params = {
            "IMAGENET": {},
            "CIFAR10": {},
        }
    else:
        assert False
    param = params[config.config["DATA_SET"]]

    attack = channel_atk(image_ph, label_ph, ** param)
    add_preprocess_func(attack_name, attack.pre_built)

    def build_attack(model_container):
        attack.build(model_container)

    def run_attack(x, y):
        return attack.attack(x, y, sess)
    return build_attack, run_attack


def Semantic_Attack(image_ph, label_ph, sess, attack_name, method="color"):
    if method == "color":
        from attack_algs.semantic.cadv.sm_color import semantic_color as semantic_atk
        params = {
            "IMAGENET": {"model_path": os.path.join(".", "attack_algs", "semantic", "cadv", "latest_net_G.pth")},
            "CIFAR10": {"model_path": os.path.join(".", "attack_algs", "semantic", "cadv", "latest_net_G.pth")},
        }
    else:
        assert False
    param = params[config.config["DATA_SET"]]

    attack = semantic_atk(image_ph, label_ph, ** param)
    add_preprocess_func(attack_name, attack.pre_built)

    def build_attack(model_container):
        attack.build(model_container)

    def run_attack(x, y, target=None):
        return attack.attack(x, y, sess, target)
    return build_attack, run_attack


def FeatureSpace_Attack(image_ph, label_ph, sess, attack_name, method, strength):
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
        "attack_algs", "feature_space", "vgg19_normalised.npz")

    attack = fs_attack(image_ph, label_ph, ** param)

    add_preprocess_func(attack_name, attack.pre_built)
    add_restore_func(attack_name, attack.restore)

    def build_attack(model_container):
        attack.build(model_container)

    def run_attack(x, y):
        return attack.attack(x, y, sess)
    return build_attack, run_attack


def SelfFeatureInduce_Attack(image_ph, label_ph, sess, attack_name, **param):
    base_param = {"step_size": 0.1, "num_steps": 2000, "scale": 0.01,
                  "target_layer": 1, "bound_penalty_multiplier": 1, "stat_type": "neuron_min_max"}
    
    param = {**base_param, "barrier_type": "polynomial", "barrier_power": 10, ** param}
    # param = {**base_param, "barrier_type": "relu", "leakiness": 1e-3, ** param}

    from attack_algs.self_induce_feature.internal_barrier \
        import INT_BARR_Attack as fs_attack
    attack = fs_attack(image_ph, label_ph, ** param)

    def collect_func(iterator, iteration):
        attack.collect_statistics(iterator, iteration, sess)

    def build_attack(model_container):
        attack.build(model_container)

    def run_attack(x, y):
        return attack.attack(x, y, sess)

    add_collect_func(attack_name, collect_func, 10000)

    return build_attack, run_attack


Handler = {
    "PGD_Linf": {
        "function": PGD_Attack,
        "parameter": {"distance": "Linf"}},
    "HSV_Channel_Attack": {
        "function": Channel_Attack,
        "parameter": {"channel": "HSV"}},
    "RGB_Channel_Attack": {
        "function": Channel_Attack,
        "parameter": {"channel": "RGB"}},
    "Semantic_Color_Attack": {
        "function": Semantic_Attack,
        "parameter": {"method": "color"}},
    "Feature_Space_Argmentation": {
        "function": FeatureSpace_Attack,
        "parameter": {"method": "augmentation", "strength": 1}},
    "Self_Feature_Induce_Attack": {
        "function": SelfFeatureInduce_Attack,
        "parameter": {}},
}

Attack_Name_List = Handler.keys()

CREATE_ATTACK_WARNING = []

def _wrap_run_atk_with_switch(sess, run_attack, attack_name):
    def __enter_pre_built():
        if "pre_built_attack" in config.config:
            assert config.config["create_attack_finish_flag"], "Please build classifier before running attack."
            switch_ops = config.config["pre_built_attack"]["switch_ops"]
            if attack_name in switch_ops:
                sess.run(switch_ops[attack_name])
            else:
                if attack_name not in CREATE_ATTACK_WARNING:
                    print("Note that the pre built of %s is not given" %
                          attack_name)
                    CREATE_ATTACK_WARNING.append(attack_name)

    def __exit_pre_built():
        if "pre_built_attack" in config.config:
            switch_ops = config.config["pre_built_attack"]["switch_ops"]
            sess.run(switch_ops["default"])

    def run_attack_wrap(x, y, target=None):
        __enter_pre_built()
        result = run_attack(x, y, target=target)
        __exit_pre_built()
        return result
    return run_attack_wrap

def create_attack(image_ph, label_ph, attack_name, sess, iterator=None):
    """ 
        image placeholder
        label placeholder
        model container returned by build classifier
        attack_name registered
    """
    info = Handler[attack_name]
    function = info["function"]
    params = info["parameter"]

    def __idle(sess):
        #print(attack_name, "called idle" )
        pass

    config.config["atk_func"][attack_name] = {"restore": __idle}

    build_attack, run_attack = function(
        image_ph, label_ph, sess, attack_name, **params)

    run_attack_wrap = _wrap_run_atk_with_switch(sess, run_attack, attack_name)

    config.config["atk_func"][attack_name]["build"] = build_attack
    config.config["atk_func"][attack_name]["attack"] = run_attack_wrap


if __name__ == "__main__":
    import classifiers
    import dataprep
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model_name = "Imagenet_Denoise"  # "Imagenet_Resnet_Normal"
        print("Begin to use models for testing attacks: ", model_name)

        config.set_data_set("IMAGENET")
        dataprep.init_data("eval")
        classifiers.init_model(model_name)

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
        for atk_name in ["Self_Feature_Induce_Attack"]:  # Handler:
            print("Building attacks:", atk_name)
            create_attack(content, label,  atk_name, sess)

        atk_func = config.config["atk_func"][atk_name]

        """Third step : build classifier"""
        with layer_record("classifier") as lwp:
            model_container, _ = build_classifier(content, label)

        build_all(model_container)

        sess.run(tf.global_variables_initializer())
        print("Restoring parameters")
        restore_classifier(sess)

        init_all(sess)
        collect_all(next_batch)
        for test_batch in range(2):
            x_batch, y_batch = next_batch()
            x_batch = atk_func["attack"](x_batch, y_batch)
            _pred, _acc = sess.run([model_container.prediction,
                                    model_container.accuracy], feed_dict={
                content: x_batch, label: y_batch})
            print("Label:      ", y_batch)
            print("Prediction: ", _pred)
            print("Accuracy: ", _acc)
