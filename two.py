import tensorflow as tf
import numpy as np


# palcehoder 占位符，暂时存储变量 从外部传入变量，而不是tf.Variable '内部变量'
# 激励函数就是产生非线性特征（光滑可微分）   梯度消失和梯度爆炸 CNN选择relu RNN tanh或者relu


# 定义隐藏层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    Bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    W_plus_B = tf.matmul(inputs, Weights) + Bias  # 神经网络激活值  inputs(sample, feature) weight(feature, out_put)

    if activation_function is None:
        out_put = W_plus_B
    else:
        out_put = activation_function(W_plus_B)
    return out_put


X_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]   # 插入新维度
print(X_data)
noise = np.random.normal(0, 0.05, X_data.shape).astype(np.float32)
print(noise)
y_data = np.square(X_data)-0.5 + noise

X_S = tf.placeholder(tf.float32, [None, 1])  # 1 代表只有一个特征值 None 代表可以输入无限个
Y_S = tf.placeholder(tf.float32, [None, 1])

# 定义隐藏层
l1 = add_layer(X_S, 1, 10, activation_function=tf.nn.relu)
# 定义输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y_S - prediction), reduction_indices=[1]))  # 最后一个参数表示处理维度

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={X_S: X_data, Y_S: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={X_S: X_data, Y_S: y_data}))
sess.close()
