# coding=utf-8
import tensorflow as tf
import recnet.model.cnn_cell as cnn
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    prediction, train_step = cnn.get_cnn_cell()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={cnn.xs: batch_xs, cnn.ys: batch_ys, cnn.keep_prob: 0.5})
        if i % 50 == 0:

            accuracy = cnn.compute_accuracy(cnn.xs, cnn.ys, mnist.test.images, mnist.test.labels, cnn.keep_prob, sess, prediction)
            print(accuracy)
