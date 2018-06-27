#-*- coding:utf-8 -*-
import tensorflow as tf

from recnet.input.poi_input import POI

class POIInput( ):

    def __init__(self, file):

        print('init')
        self.file = file
        self.poi = POI(file_source=file)

    def test(self):

        print('test')
        example, label = self.poi.decode()
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(coord=coord)
            for i in range(10):
                e_val, l_val = sess.run([example, label])
                # print("i is : %s , %s"(i, m, n))
                print(i, e_val, l_val)
            coord.request_stop()
            coord.join(threads=thread)


if __name__ == '__main__':

    file_name = '/home/cheung/data/test/'
    poiinput = POIInput(file_name)
    poiinput.test()