import tensorflow as tf
import config
import functools


def split_input(input, num):
    x_lst = tf.split(x, num)
    return x_lst



def multi_gpu(func, input_num, output_num, tot_output_num):
    """ input_num : number of input tensors of target function, e.g. image and label are 2
        output_num : number of output tensors of target function.
        tot_output_num : number of total outputs of target function, the function cay carry non-tensor additional information in this area.

        Function Call Format: All the positional arguments should be tensor, put additional information in keyword arguments.
        Return Format: The first output_num of outputs are considered as tensors and concatenated, the rest is put in a list

        Hack Part, the tenosr will have a uncollected key 'fast_distribute_elems' for fast multi-gpu recalculation.
    """
    assert tot_output_num >= 0 and output_num >= 0 and output_num <= tot_output_num

    def multi_func(*arg, **karg):
        GPU_NUM = config.config["MULTIGPU_NUM"]
        if GPU_NUM == 1:
            return func(*arg, **karg)
        else:
            config.config["MULTI_GPU_ENV"] = True
            arg = list(arg)
            input_splits = []
            assert len(arg) == input_num
            for idx in range(input_num):

                assert isinstance(
                    arg[idx], (tf.Tensor, list, tuple)), "You can only provide tensors or list of Tensors for positional argument instead of %s" % (str(type(arg[idx])))
                if isinstance(arg[idx], tf.Tensor):
                    if "fast_distribute_elems" in arg[idx].__dict__.keys():
                        input_splits.append(arg[idx].fast_distribute_elems)
                    else:
                        input_splits.append(tf.split(arg[idx], GPU_NUM))
                else:
                    assert len(arg[idx]) == GPU_NUM
                    for j in arg[idx]:
                        assert isinstance(j, tf.Tensor), "You can only provide list of Tensors for positional argument instead of list of %s" % (
                            str(type(j)))
                    input_splits.append(arg[idx])

            outputs = []

            def append_output():
                nonlocal inputs
                for idx in range(input_num):
                    arg[idx] = inputs[idx]
                output = func(*arg, **karg)
                if isinstance(output, (list, tuple)):
                    assert len(output) == tot_output_num, "Output number mismatch, Expected %d but received %d" % (
                        tot_output_num, len(output))
                else:
                    assert tot_output_num == 1, "Output number mismatch, Expected %d but received %d" % (
                        tot_output_num, 1)
                if tot_output_num == 1:
                    output = [output]
                outputs .append(output)
            for gpu_id, inputs in enumerate(zip(*input_splits)):
                with tf.device(assign_to_device("/gpu:%d" % gpu_id)):
                    if gpu_id == 0:
                        reuse = config.config["reuse"]
                        config.config["reuse"] = False
                        with tf.variable_scope("", reuse=tf.get_variable_scope().reuse):
                            append_output()
                        config.config["reuse"] = reuse
                    else:
                        reuse = config.config["reuse"]
                        config.config["reuse"] = True
                        with tf.variable_scope("", reuse=True):
                            append_output()
                        config.config["reuse"] = reuse
            config.config["MULTI_GPU_ENV"] = False
            # The function returns a list, only the first one is output

            def reduce_output(ith_output):
                shape = ith_output[0].shape.as_list()
                dtype = ith_output[0].dtype
                for gpu_id in range(GPU_NUM):
                    assert shape == ith_output[gpu_id].shape.as_list() and dtype == ith_output[gpu_id].dtype 
                if len(shape) == 0:
                    # This is a scalar, we took the sum of it
                    #output_tensor = tf.reduce_sum(ith_output)
                    # we keep it unchanged
                    output_tensor = ith_output
                else:
                    output_tensor = tf.concat(ith_output, axis=0)
                    output_tensor. fast_distribute_elems = ith_output
                return output_tensor

            if tot_output_num == 0:
                return None
            else:
                outputs = list(zip(*outputs))
                if tot_output_num == 1:
                    outputs = outputs[0]
                    if output_num == 1:
                        outputs = reduce_output(outputs)
                    return outputs
                else:
                    for i in range(output_num):
                        outputs[i] = reduce_output(outputs[i])
                    return outputs

    return multi_func


def reduce_container(containers):
    l = len(containers)
    cta = containers[0]
    for k in cta.__dict__.keys():
        cta.__dict__[k] = [containers[i].__dict__[k] for i in range(l)]
    return cta       
        

def multi_gpu_decor(input_num, output_num, tot_output_num):
    return functools.partial(multi_gpu, 
                input_num=input_num, 
                output_num=output_num,
                tot_output_num=tot_output_num)


@multi_gpu_decor(2,1,1)
def multi_plus(a,b):
    return a+b

def multi_distribute(x):
    return [x for _ in range(config.config["MULTIGPU_NUM"])]

def multi_adam(*arg, **karg):
    return multi_opt(tf.train.AdamOptimizer, *arg, **karg)


def multi_sgd(*arg, **karg):
    return multi_opt(tf.train.GradientDescentOptimizer, *arg, **karg)

def multi_opt(opt, learning_rate, loss, var_list, global_step, **karg):
    grads = []

    for gpu_id in range(config.config["MULTIGPU_NUM"]):
        with tf.device("/gpu:%d" % gpu_id):
            optimizer = opt(learning_rate=learning_rate, **karg)
            grad = optimizer.compute_gradients(loss[gpu_id],
                                               var_list=var_list)
            grads.append(grad)
    tower_grads = average_gradients(grads)
    train_op = optimizer.apply_gradients(tower_grads, global_step=global_step)

    return train_op


# from https://wizardforcel.gitbooks.io/tensorflow-examples-aymericdamien/6.2_multigpu_cnn.html
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device  # "/" + ps_device
        else:
            return device

    return _assign

def sanitize_gradient(grad):
    grad = tf.where(tf.is_finite(grad), grad, tf.zeros_like(grad))
    return grad

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            g = sanitize_gradient(g)
            g = tf.clip_by_value(g, -1.0, 1.0)
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
