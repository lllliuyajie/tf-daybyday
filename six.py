import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from layer import add

# dropout 随机忽略掉一些神经元和神经连接 使神经网络变得不完整 解决过拟合


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_s: v_xs})
    # print(y_pre)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))      # argmax 按行（1）或者按列（0）算最大值
    # 了解mnist数据结构的label 2018/6/26
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      # cast 类似于映射（映射到一个你指定的类型）
    result = sess.run(accuracy, feed_dict={x_s: v_xs, y_s: v_ys})
    return result


digit = load_digits()
x = digit.data
y = digit.target
y = LabelBinarizer().fit_transform(y)    # 标签二值化(0,1)   inverse_transform(逆过程)
x_train, x_test, y_trian, y_test = train_test_split(x, y, test_size=0.3)


keep_prob = tf.placeholder(tf.float32)
x_s = tf.placeholder(tf.float32, [None, 64])
y_s = tf.placeholder(tf.float32, [None, 10])


l1 = add.add_2_layer(x_s, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add.add_2_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_s * tf.log(prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train, feed_dict={x_s: x_train, y_s: y_trian, keep_prob: 0.5})
    if i % 50 == 0:
        print('训练：'+sess.run(compute_accuracy(x_test, y_test)))
sess.close()

