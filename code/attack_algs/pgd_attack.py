"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import config
import os
from utils import save_pair_diff
IS_DEBUG = True


class LinfPGDAttack:
    def __init__(self, x_ph, y_ph, container,  target_attack, epsilon, num_steps, step_size, random_start,  dynamic_step=True):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.target_attack = target_attack
        if not target_attack:
            loss = container.target_loss_sum
            self.adv_loss_sep = container.target_loss
            self.adv_loss = container.target_loss_sum
        else:
            loss = container.target_attack_loss_sum
            self.adv_loss_sep = container.target_attack_loss
            self.adv_loss = container.target_attack_loss_sum
            self.target_ph = container.target_label

        self.grad = tf.gradients(loss, x_ph)[0]

        self.x_ph = x_ph
        self.y_ph = y_ph
        self.BATCH_SIZE = config.config["BATCH_SIZE"]
        self.dynamic_step = dynamic_step
        print("pgd epsilon: ", epsilon, "step_size", step_size)

    def generate(self, x_nat, y, sess, target = None):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + \
                np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 255)  # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        if self.dynamic_step:
            num_step = int(1e7)
            last_adv_loss = -1e20
            average_drop = 0
            loss_stat_decay = 0.95
        else:
            num_step = self.num_steps

        for i in range(num_step):
            if self.target_attack:
                feed_dict = {self.x_ph: x, self.y_ph: y, self.target_ph: target}
            else:
                feed_dict = {self.x_ph: x, self.y_ph: y}
            grad, _adv_loss, _adv_loss_tot = sess.run([self.grad, self.adv_loss_sep, self.adv_loss], feed_dict=feed_dict)
            grad = np.nan_to_num(grad, nan=0.0)
            if self.dynamic_step:
                if _adv_loss_tot >= -1e-2:
                    break
                drop_percent = (_adv_loss_tot - last_adv_loss) / \
                    abs(_adv_loss_tot)
                average_drop = average_drop * loss_stat_decay + \
                    drop_percent * (1-loss_stat_decay)
                last_adv_loss = _adv_loss_tot
                if average_drop < 1e-3:
                    break

            #print("PGD Est Step %d"%i)
            # l_infty=self.linf_estimation(x,y,sess)
            #print("Estimation L_inf: ",l_infty)
            x = x + self.step_size * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255)  # ensure valid pixel range

            if i % 100 ==0 :
                print("step %d, loss: %.2f" %(i,_adv_loss_tot) ) 
            if i>10000:
                break


        if IS_DEBUG:
            for idx in range(self.BATCH_SIZE):
                save_pair_diff(x_nat[idx], x[idx],
                               path=os.path.join("temp", "PGD", "%d.png" % idx), dynamic=True)
        return x
