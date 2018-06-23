import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):   # 输入的 in_size 就是特征值
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    Bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    W_plus_B = tf.matmul(inputs, Weights) + Bias  # 神经网络激活值  inputs(sample, feature) weight(feature, out_put)

    if activation_function is None:
        out_put = W_plus_B
    else:
        out_put = activation_function(W_plus_B)
    return out_put