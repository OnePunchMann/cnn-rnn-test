import tensorflow as tf
from recnet.tfrecord.TFRecorderHandler import Parser, Generator
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

if __name__ == '__main__':

    # source = '/home/cheung/Desktop/data'
    # dest = '/home/cheung/Desktop/gen/'
    # gen = Generator(source, dest)
    # gen.generate()

    source = '/home/cheung/Desktop/gen/'
    dest = '/home/cheung/Desktop/redata/'
    par = Parser(source, dest)
    image, label = par.parse()
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(30):
            # image_down = np.asarray(image_down.eval(), dtype='uint8')
            # plt.imshow(image.eval())
            # plt.show()
            single, l = sess.run([image, label])  # 在会话中取出image和label
            img = Image.fromarray(single, 'RGB')  # 这里Image是之前提到的
            img.save(os.path.join(par.dest, str(i)) + '_''Label_' + str(l) + '.jpg')  # 存下图片
            # print(single,l)
        coord.request_stop()
        coord.join(threads)
