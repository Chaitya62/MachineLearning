import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize

data_x = np.linspace(0, 1, 100000)
np.random.shuffle(data_x)
data_y = 20.1*data_x + 1.02



batch_size = 50

x = tf.placeholder('float64')
y = tf.placeholder('float64')

w = tf.Variable(np.random.random(1)[0], 'float64')
b = tf.Variable(np.random.random(1)[0], 'float64')

linear_model = tf.add(tf.mul(x, w), b)
alpha = 0.001
epoches = 100


cost = tf.square(linear_model-y)
cost = tf.reduce_sum(cost)
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)


def get_next_batch():
    yield data_x[:batch_size], data_y[:batch_size]

    start = batch_size
    for i in range(int(len(data_x)/batch_size)):
        en = min(start+batch_size, len(data_x))
        yield data_x[start:en], data_y[start:en]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoches):
        for epoch_x, epoch_y in get_next_batch():  
            sess.run(optimizer, {x: epoch_x, y: epoch_y})
            
        print(sess.run([w, b]))
        #print(sess.run(linear_model, {x: [1, 2, 4], y: [3, 5, 9]}))




