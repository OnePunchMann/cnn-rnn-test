# coding = utf-8
import tensorflow as tf
import os

class POI:

    def __init__(self, file_source):
        self.__source = file_source
        self.__file_names = self.__getName()
        self.__filename_queue = tf.train.string_input_producer(self.__file_names, shuffle=False)
        self.__reader = tf.TextLineReader()

    def __getName(self):
        name = []
        for root, dirs, files in os.walk(self.__source):
            for file in files:
                name.append(os.path.join(root + file))
        return name

    def __readLine(self):
        key, value = self.__reader.read(self.__filename_queue)
        return key,value

    def decode(self):
        key, value = self.__readLine()
        example, lable = tf.decode_csv(value, record_defaults=[['null'], ['null']])
        example_batch, label_batch = tf.train.shuffle_batch([example, lable], batch_size=20, capacity=200, min_after_dequeue=100, num_threads=2)
        return example_batch, label_batch
