import tensorflow as tf
import numpy as np

'''
1.使用图（graph）来表示计算任务 图中的节点称为operation
2.在被称之为会话（session）的上下文（context）中执行图
3.使用tensor（张量）表示数据
4.通过变量维护状态
5.使用feed 和 fetch可以为任意的操作赋值或者从中获取数据
'''

x_data = np.random.rand(10).astype(np.float32)   # 产生10个[0,1)区间的随机数
y_data = x_data * 0.1 + 0.3

Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 均匀分布随机数 （shape, minVal, maxVal）
# 定义变量（定义了某字符串是变量才是变量） Variable 最重要的是初始化 并且再run这一步激活
bias = tf.Variable(tf.zeros([1]))
y = Weight * x_data +bias

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 0.5为学习步长
train = optimizer.minimize(loss)

# init = tf.initialize_all_variables()    已经废弃的写法
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(401):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weight), sess.run(bias))  # weight 是张量无法直接输出，需要使用sess.run

sess.close()