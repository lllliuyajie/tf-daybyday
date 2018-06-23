import tensorflow as tf
import numpy as np


# 神经网络的可视化 tensorboard


# 定义隐藏层
def add_layer(inputs, in_size, out_size, activation_function=None):   # 输入的 in_size 就是特征值
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('Bias'):
            Bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('W_plus_B'):
            W_plus_B = tf.matmul(inputs, Weights) + Bias  # 神经网络激活值  inputs(sample, feature) weight(feature, out_put)
        if activation_function is None:
            out_put = W_plus_B
        else:
            out_put = activation_function(W_plus_B)
        return out_put


X_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]   # linspace产生等差数列 插入新维度
print(X_data)
noise = np.random.normal(0, 0.05, X_data.shape).astype(np.float32)
print(noise)
y_data = np.square(X_data)-0.5 + noise

with tf.name_scope('input'):
    X_S = tf.placeholder(tf.float32, [None, 1], name='x_input')  # 1 代表只有一个特征值 None 代表可以输入无限个
    Y_S = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 定义隐藏层
l1 = add_layer(X_S, 1, 10, activation_function=tf.nn.relu)
# 定义输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y_S - prediction), reduction_indices=[1]))  # reduction_indices = [1] 按行处理  =[0] 按列处理

with tf.name_scope('trian'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={X_S: X_data, Y_S: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={X_S:  X_data, Y_S: y_data}))
sess.close()

