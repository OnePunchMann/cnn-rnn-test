# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import datetime
import os

from recnet.configuratioin import height, width, batch_size
from recnet.model import rnn_cell as rnn

if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    optimizer, accuracy = rnn.get_rnn_cell()

    # 启动会话.开始训练
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    step = 0
    acc_rate = 0.98
    while 1:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, height, width])
        session.run(optimizer, feed_dict={rnn.x_: batch_x, rnn.y_: batch_y})
        # 每训练10次测试一次
        if step % 10 == 0:
            batch_x_test = mnist.test.images
            batch_y_test = mnist.test.labels
            batch_x_test = batch_x_test.reshape({-1, height, width})
            acc = session.run(accuracy, feed_dict={rnn.x_: batch_x_test, rnn.y_: batch_y_test})
            print(datetime.date.today().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 偏差满足要求，保存模型
            if acc >= acc_rate:
                model_path = os.getcwd() + os.sep + str(acc_rate) + "mnist.model"
                saver.save(session, model_path, global_step=step)
                break
        step += 1
    session.close()