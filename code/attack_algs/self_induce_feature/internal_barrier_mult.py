from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import config
import random
import os

from tqdm import tqdm

from layer_supervisor import get_current_layer_wrapper, get_layer_wrapper_by_name
from utils import save_pair_diff
from record import recorder
from tensorflow.distributions import Normal

import shutil

IS_DEBUG = True

def prod_list(lst):
    p = 1
    for l in lst:
        p*=l
    return p

def sign_method_step(x, grad, step_size):
    x = x + step_size * np.sign(grad)
    return x


def adapt_sign_method_step(x, grad, step_size):
    grad = grad / np.sqrt((np.sum(np.square(x), axis= (1,2), keepdims=True) / (x.shape[1]*x.shape[2])))
    x = x + step_size * grad
    return x

class int_loss_layer:
    def __init__(self, x_ph, layer_wrapper, target_layer,  scale= 0.01, stat_type="neuron_variance", barrier_type="polynomial", barrier_power=200, collect_cache_name=None):
        self.x_ph = x_ph
        self.lwp = layer_wrapper
        self.target_layer = target_layer
        self.BATCH_SIZE = config.config["BATCH_SIZE"]
        self.rec = recorder("self_induced_attack_intbarr")
        self.barrier_type = barrier_type
        self.barrier_power = barrier_power
        self.collect_cache_name = collect_cache_name
        self.scale = scale
        self.stat_type = stat_type
        self.build()

    def build(self):
        raw_feat = self.lwp.get_layer_by_id(self.target_layer)

        self.internal_feature = tf.where(tf.is_finite(
            raw_feat), raw_feat, tf.zeros_like(raw_feat, dtype=tf.float32))

        DYNAMIC_BS = tf.shape(self.x_ph)[0]
        FEATURE_SHAPE = self.internal_feature.shape.as_list()[1:]
        VAR_SHAPE = [self.BATCH_SIZE, *FEATURE_SHAPE]

        FEATURE_SHAPE = self.internal_feature.shape.as_list()[1:]
        assert len(FEATURE_SHAPE)!=4 or FEATURE_SHAPE[0] == FEATURE_SHAPE[1], \
            "Please double check whether you provide NHWC format image, Remove this line manually if this is false positive alert."

        reduce_axis = list(range(1, len(self.internal_feature.shape.as_list())))

        self.benign_internal_feature = tf.get_variable(
            "bengin_feature", initializer=tf.zeros(VAR_SHAPE, tf.float32))
        self.atk_init = [self.benign_internal_feature[:DYNAMIC_BS].assign(
            self.internal_feature)]
            # Memorize the benign feature
        curr_benign = self.benign_internal_feature[:DYNAMIC_BS]
        self.smoothness_loss = 0
        loss_set = False
        eps = 1e-9
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
            self.moving_bound = (
                self.moving_max-self.moving_min)*self.scale + eps
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
            self.moving_bound = (
                self.moving_max-self.moving_min)*self.scale + eps
            self.collectable = {"max": self.moving_max, "min": self.moving_min}

        elif self.stat_type == "neuron_variance":
            STAT_SHAPE = [1, *FEATURE_SHAPE]
            stat_axis = 0
            
            self.moving_mean = tf.get_variable(
                "stat_mean",  dtype=tf.float64, initializer=-0.0*tf.ones(STAT_SHAPE, dtype=tf.float64))
            self.moving_var = tf.get_variable(
                "stat_var", dtype=tf.float64, initializer=0*tf.ones(STAT_SHAPE, dtype=tf.float64))
            self.cnt = tf.get_variable("stat_cnt", dtype=tf.float64, initializer=tf.cast(eps, dtype=tf.float64))
            # mean = tf.math.reduce_std(tf.abs(x), axis=[0])

            self.moving_upt = [
                tf.assign_add(self.moving_mean, tf.reduce_sum(tf.cast(self.internal_feature, dtype=tf.float64), axis=stat_axis, keep_dims=True)),
                tf.assign_add(self.moving_var, tf.reduce_sum(
                    tf.square(tf.cast(self.internal_feature, dtype=tf.float64)), axis=stat_axis, keep_dims=True)),
                tf.assign_add(self.cnt, tf.cast(DYNAMIC_BS, dtype=tf.float64)),
                ]

            # Use three stddev as range
            self.moving_range = (3 * 2)* tf.sqrt(self.moving_var / self.cnt - (self.moving_mean / self.cnt) ** 2) + eps
            self.moving_range = tf.cast(self.moving_range, tf.float32)
            self.moving_bound = self.moving_range * self.scale + eps
            self.collectable = {
                "mean": self.moving_mean, "var": self.moving_var, "cnt": self.cnt}

        elif self.stat_type == "neuron_gaussian":
            loss_set = True
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
            self.gauss_mean = self.moving_mean / self.cnt + eps
            self.standard_dev = tf.sqrt(self.moving_var /
                                        self.cnt - tf.square(self.gauss_mean)) + eps
            self.gauss_mean = tf.cast(self.gauss_mean, tf.float32)
            self.gauss_std = tf.cast(self.standard_dev, tf.float32)
            self.dist = Normal(loc=self.gauss_mean, scale=self.gauss_std)
            self.current_quantile = tf.clip_by_value(self.dist.cdf(
                self.internal_feature), 0.0, 1.0)  # curr_bengin)

            geps = 0
            lowest_quantile = self.dist.quantile(
                tf.maximum(self.current_quantile - self.scale, geps))
            highest_quantile = self.dist.quantile(tf.minimum(
                self.current_quantile + self.scale, 1 - geps))

            self.mem_cq = tf.get_variable(
                "current_quantile", initializer=tf.zeros(VAR_SHAPE, tf.float32))
            self.mem_lq = tf.get_variable(
                "low_quantile", initializer=tf.zeros(VAR_SHAPE, tf.float32))
            self.mem_hq = tf.get_variable(
                "high_quantile", initializer=tf.zeros(VAR_SHAPE, tf.float32))

            mem_lq = self.mem_lq[:DYNAMIC_BS]
            mem_hq = self.mem_hq[:DYNAMIC_BS]
            mem_cq = self.mem_cq[:DYNAMIC_BS]

            self.atk_init.extend([tf.assign(mem_lq, lowest_quantile),
                                    tf.assign(mem_hq, highest_quantile),
                                    tf.assign(mem_cq, self.current_quantile)])

            self.moving_range = self.mem_hq - self.mem_lq
            self.moving_diff = tf.abs(self.current_quantile - mem_cq)
            self.moving_range = tf.cast(self.moving_range, tf.float32)
            self.moving_bound = self.moving_range + eps

            self.collectable = {
                "mean": self.moving_mean, "var": self.moving_var, "cnt": self.cnt}
            self.supervised_bound = tf.reduce_max(tf.nn.relu((curr_benign - self.internal_feature)/(curr_benign - mem_lq)) +
                                                  tf.nn.relu((self.internal_feature-curr_benign) / (mem_hq - curr_benign)), axis=reduce_axis)
            if self.barrier_type == "polynomial":
                pnt = 0.5
                counting_pnt_h = (1-pnt) * mem_hq + pnt*curr_benign
                counting_pnt_l = (1-pnt) * mem_lq + pnt*curr_benign
                #self.linf_barrier_loss_sep = tf.reduce_sum(
                #    tf.pow(
                #        tf.nn.relu((curr_benign - self.internal_feature)/(curr_benign - mem_lq)) +
                #        tf.nn.relu((self.internal_feature-curr_benign)/(mem_hq - curr_benign)), self.barrier_power), axis=reduce_axis)
                self.linf_barrier_loss_sep = tf.reduce_sum(
                    tf.pow(
                        tf.nn.relu((counting_pnt_l - self.internal_feature)/(curr_benign - mem_lq)/pnt) +
                        tf.nn.relu((self.internal_feature-counting_pnt_h)/(mem_hq - curr_benign)/pnt), self.barrier_power), axis=reduce_axis)
                self.in_bound_threshold = 1.0
            else:
                assert False

        else:
            assert False, "stat type not implemented"

        if self.stat_type == "neuron_gaussian":
            diff = (self.current_quantile - mem_cq)/self.scale#self.internal_feature - curr_benign
        else:
            diff = (curr_benign - self.internal_feature) / self.moving_bound
        avg_diff = tf.nn.pool(diff, window_shape=[3,3], pooling_type="AVG", strides=[1,1], padding="VALID")
        avg_diff = tf.stop_gradient(avg_diff)
        diff_part = diff[:,1:-1,1:-1,:]
        self.smoothness_loss +=  tf.reduce_sum(tf.reduce_mean(tf.square(diff_part-avg_diff), axis=reduce_axis))

        if not loss_set:
            self.supervised_bound = tf.reduce_max(tf.abs(
                (self.internal_feature-curr_benign)/self.moving_range), axis=reduce_axis)

            if self.barrier_type == "polynomial":
                self.linf_barrier_loss_sep = tf.reduce_sum(
                    tf.pow((self.internal_feature-curr_benign)/self.moving_bound,
                            self.barrier_power), axis=reduce_axis)
                self.in_bound_threshold = 1.0
            elif self.barrier_type == "relu":
                # Try relu function
                beta = 0.95
                #fix_bias = prod_list(FEATURE_SHAPE)*self.leakiness*2
                self.linf_barrier_loss_sep = tf.reduce_sum(
                    (tf.nn.relu(self.internal_feature-(curr_benign + beta * self.moving_bound))/self.moving_bound) +
                    (tf.nn.relu((curr_benign-beta * self.moving_bound) -
                                self.internal_feature)/self.moving_bound),
                    axis=reduce_axis)  # + fix_bias

                self.in_bound_threshold = 1-beta

        self.linf_barrier_loss = tf.reduce_sum(self.linf_barrier_loss_sep)

    def load(self, sess):
        _dict = self.rec.load_record_by_id(
            self.collect_cache_name, 0)
        filter_record_array = config.config["filter_record_array"]
        for k in self.collectable.keys():
            if len(self.collectable[k].shape.as_list()) >= 2:
                _dict_val = filter_record_array(_dict[k], self.target_layer)
            else:
                _dict_val = _dict[k]
            sess.run(tf.assign(self.collectable[k], _dict_val))


class INT_BARR_Attack:
    def __init__(self, x_ph, y_ph, step_size, num_steps, target_layer,
                bound_penalty_multiplier, barrier_type, target_attack, barrier_power=None, 
                leakiness=None, default_wrapper_name=None, collect_cache_name=None, 
                dynamic_step_size=True, dynamic_step=True, stat_type = None, max_step_size = None, feature_smooth = 0., denoise_first_only=0):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        assert config.config["MULTIGPU_NUM"] == 1, "Currently not supporting Multi GPU Attack"
        
        self.x_ph = x_ph
        self.y_ph = y_ph
        self.step_size = step_size
        self.num_steps = num_steps
        self.target_layer = target_layer
        self.barrier_type = barrier_type
        self.feature_smooth = feature_smooth
        self.denoise_first_only =  denoise_first_only
        assert self.barrier_type in ["polynomial","relu"]
        if self.barrier_type == "polynomial":
            assert isinstance(barrier_power, int)
            self.barrier_power = barrier_power

        elif self.barrier_type == "relu":
            assert isinstance(leakiness, float)
            self.leakiness=leakiness

        assert stat_type in ["channel_min_max", "neuron_min_max", "neuron_variance", "neuron_gaussian"]
        self.stat_type = stat_type

        self.dynamic_step_size = dynamic_step_size
        self.dynamic_step = dynamic_step

        self.bound_penalty_multiplier = bound_penalty_multiplier
        self.BATCH_SIZE = config.config["BATCH_SIZE"]
        self.rec = recorder("self_induced_attack_intbarr")
        
        self.collect_cache_name = collect_cache_name
        #self.collect_cache_name += "_" + self.stat_type

        if default_wrapper_name is None:
            self.default_wrapper_name = "classifier"
        else:
            self.default_wrapper_name = default_wrapper_name

        self.optimize = sign_method_step
        self.max_step_size = max_step_size
        print("Chosen Layer Wrapper Name: ", self.default_wrapper_name)
        print("Target Layer: ",  self.target_layer,
              "bound_penalty: ", self.bound_penalty_multiplier)

        self.target_attack = target_attack

    def build(self, container):

        eps = 1e-9

        # self.internal_feature - curr_benign
        if os.path.exists(os.path.join(".","temp", "steps")):
            shutil.rmtree(
                os.path.join(".", "temp", "steps"), ignore_errors=True)
        with tf.variable_scope("INT_BARR_Attack"):
            if not self.target_attack:
                self.adv_loss_sep = container.target_loss
                self.adv_loss = container.target_loss_sum
            else:
                self.adv_loss_sep = container.target_attack_loss
                self.adv_loss = container.target_attack_loss_sum
                self.target_ph = container.target_label
                assert self.target_ph is not None

            self.lwp = get_layer_wrapper_by_name(self.default_wrapper_name)
            
            self.tot_loss = self.adv_loss_sep

            CONTENT_SHAPE = self.x_ph.shape.as_list()[1:]
            VAR_SHAPE = [self.BATCH_SIZE, *CONTENT_SHAPE]
            BS = tf.shape(self.x_ph)[0]
            self.original_content = tf.get_variable("benign_content", shape=VAR_SHAPE)
            curr_content = self.original_content[:BS]
            self.atk_init = [tf.assign(curr_content, self.x_ph)]

            diff = (self.x_ph - curr_content)
            #diff = diff / \
            #    (tf.reduce_max(diff, axis=[1, 2], keepdims=True) + 1e-3)
            """avg_diff = tf.nn.pool(diff, window_shape=[3,3], pooling_type="AVG", strides=[1,1], padding="VALID")
            avg_diff = tf.stop_gradient(avg_diff)
            diff_part = diff[:,1:-1,1:-1,:]
            self.tot_loss += 10. * tf.reduce_sum(tf.reduce_mean(tf.square((diff_part-avg_diff) / 2.), axis=[1,2,3]))"""
            
            # In consideration https://arxiv.org/pdf/1903.10396.pdf Log Barrier Function, While it may in high probability cause numerical instability
            # Try high order function

            

            self.int_losses = []
            self.linf_barrier_loss_sep = []
            self.linf_barrier_loss = 0
            self.layers = len(self.target_layer)
            self.supervised_bound = []
            # self.scale, 0.25, 0.25, 0.25]
            # min_max 4e-2
            self.scales = tf.get_variable("scales", dtype=tf.float32, initializer=tf.zeros([self.layers], dtype=tf.float32))
            self.scales_ph = tf.placeholder(dtype=tf.float32, shape = [self.layers])
            self.scales_asgn = tf.assign(self.scales, self.scales_ph)
            self.quantiles = []
            #variance 8e-2
            #self.scales = [0.3821126080497358,3.2089211540119456, 3.615826370445288, 3.381776613462789]
            # 4e-2
            #self.scales = [0.18594863416062785,1.9812006376022246, 1.9812912042820772, 1.7645465309333583]
            #self.scales = [_sc/2.0   for _sc in self.scales]
            for i, layerid in enumerate(self.target_layer):
                with tf.variable_scope("layer_%d"%i):
                    _ill = int_loss_layer(self.x_ph, self.lwp, layerid, self.scales[i], self.stat_type, self.barrier_type,
                                          self.barrier_power, self.collect_cache_name+"#"+str(layerid)+"_" + self.stat_type)
                    if self.stat_type == "neuron_gaussian":
                        self.quantiles.append(_ill.current_quantile)
                    self.tot_loss -= self.bound_penalty_multiplier * _ill.linf_barrier_loss_sep
                    if self.denoise_first_only == 0 or ( self.denoise_first_only != 0 and i<self.denoise_first_only):
                        self.tot_loss += self.feature_smooth * _ill.smoothness_loss
                    self.int_losses.append(_ill)
                    self.linf_barrier_loss_sep.append(
                        _ill.linf_barrier_loss_sep)
                    self.in_bound_threshold = _ill.in_bound_threshold
                    self.linf_barrier_loss += _ill. linf_barrier_loss
                    self.supervised_bound.append(_ill.supervised_bound)
                    self.atk_init.append(_ill.atk_init)
            
            self.grad=tf.gradients(self.tot_loss, self.x_ph)[0]

            #_,h,w,_ = self.grad.shape.as_list()
            #self.grad = tf.nn.avg_pool(self.grad, ksize=(1,2,2,1),strides=(1,2,2,1), padding="SAME")
            #self.grad = tf.image.resize(self.grad, size=(h,w))


            # self.check = tf.add_check_numerics_ops()

    def _input_dec(self, x, y = None, target = None):
        _dict = {self.x_ph: x}
        if self.target_attack and target is not None:
            _dict[self.target_ph] = target
        if y is not None:
            _dict[self.y_ph] = y 
        return _dict

    def binary_dynam_learn_rate_search(self, x, y, sess, target=None, start=0.0, end=128.0 , bin_times = 8, steps_probe = 100): #
        end = min(self.max_step_size, end) * 2
        bs = x.shape[0]
        for i in range(bin_times):
            mean = (start + end)/2
            over_bound = False
            x_adv = np.copy(x)
            for step in range(steps_probe):
                grad = sess.run(self.grad, feed_dict=self._input_dec(x_adv,y,target) )
                grad = np.nan_to_num(grad, nan=0.0)
                #x_adv = x_adv + mean * np.sign(grad)
                x_adv = self.optimize(x_adv, grad, mean)

            _bar_loss = sess.run(self.linf_barrier_loss_sep, feed_dict=self._input_dec(x_adv,y,target) )
            for i, _ill in enumerate(self.int_losses):
                for bid in range(bs):
                    if _bar_loss[i][bid]>_ill.in_bound_threshold:
                        over_bound = True
                        break
            if over_bound:
                end = mean
            else:
                start = mean
        est = (start+end)/2
        # further reduce for reliable optimization
        est = est / 2#4 * 3
        return est

    def set_scales(self, sess, scales):
        sess.run(self.scales_asgn, feed_dict={self.scales_ph: scales})

    def attack(self, x_nat, y, sess, target = None):
        if self.target_attack:
            assert target is not None
        BS=x_nat.shape[0]
        x=np.copy(x_nat)
        # Select the sample within the boundary
        mem_loss=[-np.inf for _ in range(BS)]
        final_x = np.copy(x_nat)
        sess.run(self.atk_init, feed_dict = self._input_dec(x,y,target))
        # _out_penalty,  = sess.run([self.linf_barrier_loss], feed_dict={self.x_ph: x,
        #                                                            self.y_ph: y})

        if self.dynamic_step_size:
            step_size = self.binary_dynam_learn_rate_search(x_nat, y, sess, target = target)
        else:
            step_size = self.step_size

        if self.max_step_size is not None:
            step_size = min(float(self.max_step_size), step_size)

        if self.dynamic_step:
            num_step = int(1e7)
            last_adv_loss = -1e20
            average_drop = 0
            loss_stat_decay = 0.95
        else:
            num_step = self.num_steps

        print("estimated optimal step size : %.2e"% step_size)
        
        
        for i in range(num_step):
            grad, _bar_loss, _adv_loss, _adv_loss_tot=sess.run([self.grad, self.linf_barrier_loss_sep, self.adv_loss_sep, self.adv_loss], feed_dict = self._input_dec(x,y,target))

            
            #grad = np.nan_to_num(grad, nan = 0.0)

            #### Filtering small signal
            #abs_grad = np.absolute(grad)
            #avg_mean = 1e-1 * np.max(abs_grad, axis=(1,2,3), keepdims=True)
            #grad = np.where(abs_grad < 1e-5, np.zeros_like(grad), grad)

            for j in range(BS):
                flag = True
                for k in range(self.layers):
                    if _bar_loss[k][j] > self.in_bound_threshold:
                        flag = False
                        break
                if _adv_loss[j] > mem_loss[j] and flag:
                    final_x[j:j+1, :, :, :] = x[j:j+1, :, :, :]
                    #print("step %d, update %d" % (i,j) )
                    mem_loss[j]=_adv_loss[j]

            # !!!!! Fuck What happens here, crash after changing x to x_final, check training setting, training setting ok. What is the shit? The reason
            # !!!!! is because there are mixed use of tenorpack data and numpy data 
            if IS_DEBUG and i % 50 == 0:
                _out_penalty, _adv_penalty, _bound = sess.run([self.linf_barrier_loss, self.adv_loss, self.supervised_bound],
                                                              feed_dict= self._input_dec(x,y,target))
                print("step %d: boundary penalty %.2f, adv_penalty: %.2f" % (i, _out_penalty, _adv_penalty))
                for j in range(self.layers):
                    print("\t, layer %d bound percent: %s" % (j, str(_bound[j])))


                save_pair_diff(x_nat[0], final_x[0],
                            path=os.path.join("temp", "steps", "%d.png" % i), dynamic=True)

            # print("PGD Est Step %d"%i)
            # l_infty=self.linf_estimation(x,y,sess)
            # print("Estimation L_inf: ",l_infty)
            # predefined step size
            #x=x + self.step_size * np.sign(grad)
            # dynamic step size
            #x = x + step_size * np.sign(grad)

            x = self.optimize(x, grad, step_size)
            # Adaptive Sign Method
            # grad_max = np.amax(np.abs(grad), axis=(1,2), keepdims=True)
            # grad = grad / (grad_max + 1e-5)
            # x = x + self.step_size * grad

            

            if IS_DEBUG and i % 200 == 0:
                print("Inspect gradient nan: ", np.all(np.isfinite(grad), axis=(1, 2, 3)),np.all(np.isfinite(x), axis=(1, 2, 3)))

            # x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x=np.clip(x, 0.0, 255.0)  # ensure valid pixel range

            _adv_loss_tot = np.sum(mem_loss)
            if self.dynamic_step:
                if self.target_attack:
                    drop_percent = (_adv_loss_tot - last_adv_loss) / max(abs(_adv_loss_tot + BS *
                                                        config.config["confidence"]), 10.0)  # control the confidence loss numerical problem
                else:
                    drop_percent = (_adv_loss_tot - last_adv_loss) / max(abs(_adv_loss_tot + BS * 5 *
                                                                         config.config["confidence"]), 10.0)  # control the confidence loss numerical problem
                average_drop = average_drop * loss_stat_decay + \
                    drop_percent * (1-loss_stat_decay)
                last_adv_loss = _adv_loss_tot
                if average_drop < 1e-4 or _adv_loss_tot >= -1e-2:
                    break

        if IS_DEBUG:
            for idx in range(self.BATCH_SIZE):
                save_pair_diff(x_nat[idx], final_x[idx],
                               path=os.path.join("temp", "INT_BARR_ATK", "%d.png" % idx), dynamic=True)
        return final_x

    def collect_statistics(self, iterator, iteration, sess):
        for i in range(self.layers):
            self.int_losses[i].load(sess)
