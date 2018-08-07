import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from layer import add

mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # one-hot编码，更合理的计算欧式距离
print(mnist.train.labels)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_s: v_xs})
    # print(y_pre)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))      # argmax 按行（1）或者按列（0）算最大值   返回最大值的下标

    # 了解mnist数据结构的label 2018/6/26
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # cast 类似于映射（映射到一个你指定的类型）
    result = sess.run(accuracy, feed_dict={x_s: v_xs, y_s: v_ys})
    return result


# 不是TensorFlow的变量 所以使用 palcehoder
x_s = tf.placeholder(tf.float32, [None, 784])   # None 不规定样本数，但规定大小784
y_s = tf.placeholder(tf.float32, [None, 10])

prediction = add.add_layer(x_s, 784, 10, activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_s * tf.log(prediction), reduction_indices=[1]))  # 了解函数的机制！ 2018/6/23 14:37
# 交叉熵(深度学习常用概念，一般求目标与预测值之间的差距) 正负样本不均衡   可以使得模型的输出的分布尽量与训练样本分布一致
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x_s: batch_x, y_s: batch_y})

    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
sess.close()
