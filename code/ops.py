import tensorflow as tf
import numpy as np

def batch_norm(name, input, training, offset=True, beta=0.95):

    lshape = len(input.shape.as_list())

    epsilon = 1e-3
    beta = beta
    assert lshape in (2, 4)
    if lshape == 4:
        axis = [0, 1, 2]
    elif lshape == 2:
        axis = [0]
    shape = input.shape.as_list()
    shrink_shape = [shape[i] for i in range(lshape)]
    for i in axis:
        shrink_shape[i] = 1
    with tf.variable_scope(name):
        #assert training is not None and training != True and training != False

        mean_var = tf.get_variable(
            "moving_mean", shape=shrink_shape, trainable=False, initializer=tf.zeros_initializer())
        var_var = tf.get_variable(
            "moving_variance", shape=shrink_shape, trainable=False, initializer=tf.ones_initializer())  # variance variable

        if offset:
            linear_gamma = tf.get_variable(
                "gamma", shape=shrink_shape, trainable=True, initializer=tf.ones_initializer())
            linear_beta = tf.get_variable(
                "beta", shape=shrink_shape, trainable=True, initializer=tf.zeros_initializer())

        #mean_est = tf.reduce_mean(input, axis=axis, keep_dims=True)
        #var_est = tf.reduce_mean(tf.pow(input,2), axis=axis, keep_dims=True)

        mean_est, var_est = tf.nn.moments(input, axes=axis, keep_dims=True)

        update_mean = tf.assign(
            mean_var, beta*mean_var+(1-beta)*mean_est, name=name+"mean_asgn")
        update_var = tf.assign(
            var_var, beta*var_var+(1-beta)*var_est, name=name+"var_asgn")

        if offset:
            def normalized_input_train(): return linear_gamma * \
                (input-mean_est)*tf.rsqrt(var_est+epsilon)+linear_beta

            def normalized_input_test(): return linear_gamma * \
                (input-mean_var)*tf.rsqrt(var_var+epsilon)+linear_beta
        else:
            def normalized_input_train(): return (input-mean_est)*tf.rsqrt(var_est+epsilon)

            def normalized_input_test(): return (input-mean_var)*tf.rsqrt(var_var+epsilon)

        if training in [True, False]:
            if training:
                normalized_input = normalized_input_train()
            else:
                normalized_input = normalized_input_test()
        else:
            normalized_input = tf.cond(
                training, normalized_input_train, normalized_input_test)

        update_ops = [update_mean, update_var]
        #var_list = [mean_var, var_var]
        # if offset:
        #    var_list. extend( [linear_gamma, linear_beta] )

        for op in update_ops:
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)

    return normalized_input


def cnn(name, input, filter_size, filter_num, act_func=tf.nn.relu,
        pool=True, filter_stride=1, pool_stride=2, bias=True,
        initializer=None, padding="SAME"):

    shape = input.get_shape().as_list()

    if len(shape) != 4:
        assert False
    lst_ch = shape[3]

    #if initializer is None:
    #    val = np.sqrt(6/(filter_size*filter_size*lst_ch))
    #    initializer = tf.random_uniform_initializer(-val, val)

    f = tf.get_variable(name="%s_weight" % name,
                        shape=[filter_size, filter_size, lst_ch, filter_num],
                        initializer=initializer,
                        trainable=True)

    cnn = tf.nn.conv2d(input=input,
                       filter=f,
                       strides=[1, filter_stride, filter_stride, 1],
                       padding=padding)
    if bias:

        # b=tf.get_variable(name="%s_b"%name,
        #    shape=[1,1,1,filter_num],trainable=True)
        initial = tf.constant(0.0, shape=[1, 1, 1, filter_num])
        b = tf.get_variable(name="%s_bias" % name, initializer=initial)

        internal = act_func(cnn+b)
        weight = [f, b]
    else:
        internal = act_func(cnn)
        weight = [f]

    output = internal

    if pool:
        mx_pool = tf.nn.max_pool(value=internal,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, pool_stride, pool_stride, 1],
                                 padding="SAME")
        output = mx_pool

    updates = []

    return output


def dense(name, input, hidden_num, act_func=tf.nn.relu,
          initializer=None):

    shape = input.get_shape().as_list()
    batch_size = tf.shape(input)[0]
    if len(shape) != 2:
        j = 1
        for i in shape[1:]:
            j *= i
        input = tf.reshape(input, shape=[batch_size, j])
        shape = input.get_shape().as_list()
    elif len(shape) == 1:
        assert False

    lst_hd = shape[1]
    """if initializer is None:
        val = np.sqrt(6/lst_hd)
        initializer = tf.random_uniform_initializer(-val, val)"""

    w = tf.get_variable(name="%s_weight" % name,
                        shape=[lst_hd, hidden_num],
                        #initializer=initializer,
                        trainable=True)

    # b=tf.get_variable(name="%s_b"%name,
    #                  shape=[1,hidden_num],
    #                  trainable=True)
    initial = tf.constant(0.0, shape=[1, hidden_num])
    b = tf.get_variable(name="%s_bias" % name, initializer=initial)

    output = act_func(tf.matmul(input, w) + b)

    #weight = [w, b]
    #updates = []

    return output
