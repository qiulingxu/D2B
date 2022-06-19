import config
import tensorflow as tf

from ops import  cnn,dense,batch_norm
from sync_batch import sync_batch_norm


def lwp_embed_build(_input, training):
    embed_num = min(256, _input.shape.as_list()[-1])

    _input = sync_batch_norm(
        inputs=_input, is_training=training, scope="bn1", offset=False, decay=0.995, epsilon=1e-5)

    _input = cnn(
        input=_input, name="cnn", filter_size=1, filter_num=embed_num, pool=False)
    # Newly added
    _input = tf.nn.relu(_input)
    # Global Average Pooling
    
    _input = tf.reduce_mean(_input, axis=[1, 2])

    _input = sync_batch_norm(
        inputs=_input, is_training=training, scope="bn2", offset=False, decay=0.995, epsilon=1e-5)
    
    embed = _input
    #_input = batch_norm(input=_input, training=self.training, name="bn", offset=False, beta=0.95)

    CLASS_NUM = config.config["CLASS_NUM"]
    logits = dense(name="dense", input=_input,
                    hidden_num=CLASS_NUM, act_func=tf.identity)

    return logits, embed
