# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from third_party.baselines.a2c.utils import conv, fc, conv_to_fc
from third_party.baselines.common.distributions import make_pdtype
from third_party.baselines.common.input import observation_input
import numpy as np
import tensorflow as tf
import gin

@gin.configurable
def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = unscaled_images / 255. # Normalization
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

@gin.configurable
class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, reuse=False, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x, **conv_kwargs)

            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        greedy_action = self.pd.mode()
        neglogp0 = self.pd.neglogp(a0)

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, neglogp

        def eval_step(ob, *_args, **_kwargs):
            a = sess.run(greedy_action, {X:ob})
            return a

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.eval_step = eval_step
        self.value = value
