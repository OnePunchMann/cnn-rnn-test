
import tensorflow as tf
import recnet.configuratioin as con

# input layer
x_ = tf.placeholder(tf.float32, [None, con.height, con.width])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape, w_alpha = 0.01):
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape, b_alpha = 0.1):
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)

def get_rnn_cell():

    #weight and bias
    w = weight_variable([con.rnn_size, con.out_size])
    b = bias_variable([con.out_size])

    x = tf.transpose(x_, [1, 0, 2])
    x = tf.reshape(x, [-1, con.width])
    x = tf.split(x, con.height)

    #LSTM and output
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(con.rnn_size)
    output, status = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    y_conv = tf.add(tf.matmul(output[-1], w),b)

    #optimizer and deviation
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return optimizer, accuracy