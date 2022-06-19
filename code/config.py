def dummy_filter_record_array(x, layer):
    return x

def __init__():
    global config
    config = {}
    
    # for multi gpu module
    config["MULTIGPU_NUM"] = 1
    config["MULTI_GPU_ENV"] = False
    config["resnet_record_mode"] = "None"
    config["filter_record_array"] = dummy_filter_record_array
    config["reuse"] = False

def set_data_set(data_set):
    global config
    config["create_attack_finish_flag"] = False
    config["atk_func"] = {}
    config["classifier_func"] = {}
    if data_set == "IMAGENET":
        config["IMAGE_SIZE"] = (224, 224, 3)
        config["BATCH_SIZE"] = 8
        config["CLASS_NUM"] = 1000
        config["DATA_SET"] = "IMAGENET"
    elif data_set == "CIFAR10":
        config["IMAGE_SIZE"] = (32, 32, 3)
        config["BATCH_SIZE"] = 64
        config["CLASS_NUM"] = 10
        config["DATA_SET"] = "CIFAR10"
    elif data_set == "MNIST":
        config["IMAGE_SIZE"] = (28, 28, 1)
        config["BATCH_SIZE"] = 512
        config["CLASS_NUM"] = 10
        config["DATA_SET"] = "MNIST"
    else:
        assert False


if "FIRST_RUN_FLAG" in globals():
    pass
else:
    FIRST_RUN_FLAG = True
    __init__()
