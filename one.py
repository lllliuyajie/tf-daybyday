import tensorflow as tf
import numpy as np

x_data = np.random.rand(10).astype(np.float32)
y_data = x_data * 0.1 +0.3

Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))
y = Weight * x_data +bias

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(401):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weight), sess.run(bias))  # weight 是张量无法直接输出，需要使用sess.run

sess.close()