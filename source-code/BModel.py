import os
import tensorflow as tf
_config = tf.ConfigProto()
_config.gpu_options.allow_growth = True

class BModel(object):
    def __init__(self, config):
        self.config = config
        self.sess   = None

    def initialize_session(self):
        self.sess = tf.Session(config=_config)
        return self.sess
