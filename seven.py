import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 卷积核的数字如何定义?   一般初始为很小的随机值，无需提前设计 所以均是从经验出发，没有很明确的理论依据，因为使用的是BP 算法 所以在训练中会更新W,B
# tensorflow中的save 只能保存变量  现在用处不大
# RNN 循环神经网络
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# compute_accuarcy
def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_s: v_x, keep_prob: 1.0})
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_y, 1))   # return boolean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    # Tensor("Mean_1:0", shape=(), dtype=float32)
    result = sess.run(accuracy)  # 使得tensor变成人类可以看懂的东西
    return result

# weight_variable   卷积核大小


def weight_variable(shape):
    inital_weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital_weight)
# bias_variable


def bias_variable(shape):
    inital_bias = tf.constant(0.1, shape=shape)
    return tf.Variable(inital_bias)
# conv2d   二维卷积
# conv2d(input, fliter, strides, padding, use_cudnn_on_gpu=None)
# fliter : [filter_height, filter_width, in_channels, out_channels]
# strides[0]：batch     strides[3]:channel   strides [1,x_move,y_move,1]
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
# pooling


def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_s = tf.placeholder(tf.float32, [None, 784]) # 28*28
y_s = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x_s, [-1, 28, 28, 1])     # [-1,28,28,1]   -1 不考虑输入维度 28*28的图片 1是channel

# convolutional layer1 +max_pooling    输入 28*28的图片  一层卷积后输出 28*28*32  feature_map   经过池化层 14*14*32
# [5,5,1,32]   卷积核大小 5*5  因为是灰色图片 所以为1  32 是卷积核个数
W_conv1 = weight_variable([5, 5, 1, 32])
bias_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+bias_conv1)
h_max_pooling1 = max_pooling(h_conv1)


# convolutional layer2 +max_pooling     输入 14*14*32   经过卷积后输出 14*14*64 feature    经过池化层 7*7*64
W_conv2 = weight_variable([5, 5, 32, 64])
bias_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_max_pooling1, W_conv2)+bias_conv2)
h_max_pooling2 = max_pooling(h_conv2)

h_pool2 = tf.reshape(h_max_pooling2, [-1, 7*7*64])
# full connected layer1 +dropout    输入的池化层的三维图片应该换成一维   全连接的神经网络处理数据均是一维
W_full = weight_variable([7*7*64, 1024])
bias_full = bias_variable([1024])
full_conn = tf.nn.relu(tf.matmul(h_pool2, W_full)+bias_full)
full_conn_drop = tf.nn.dropout(full_conn, keep_prob)


# full connected layer2 +prediction
W_full2 = weight_variable([1024, 10])
bias_full2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(full_conn_drop, W_full2) + bias_full2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_s * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist_data.train.next_batch(100)
        sess.run(train_step, feed_dict={x_s: batch_x, y_s: batch_y, keep_prob: 0.5})
        if i % 50 == 0:
            print(compute_accuracy(mnist_data.test.images[:1000], mnist_data.test.labels[:1000]))


