from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy

INPUT_FILE = "samples.img"
IMAGE_WIDTH = 14
IMAGE_HEIGHT = 14
BATCH_SIZE = 20

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2
NUM_CLASSES = 10


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('<')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images():
    with open("samples.img", "br") as reader:
        global  IMAGE_HEIGHT
        global IMAGE_WIDTH
        reader.seek(0, os.SEEK_END)
        size = reader.tell()
        reader.seek(0, os.SEEK_SET)
        IMAGE_WIDTH = _read32(reader)
        IMAGE_HEIGHT = _read32(reader)
        numImages = (size - 8) / (IMAGE_WIDTH * IMAGE_HEIGHT*3+1)
        label = numpy.array(numpy.frombuffer(reader.read(1), dtype=numpy.uint8), dtype = numpy.uint8)
        values = numpy.frombuffer(reader.read(IMAGE_HEIGHT * IMAGE_WIDTH * 3), dtype=numpy.uint8)
        values = values.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        values = numpy.asarray(values, numpy.float32)
        data  =  [values]
        for i in range(1, int(numImages)):
            label = numpy.append(label, [numpy.frombuffer(reader.read(1), dtype=numpy.uint8)])
            values = numpy.frombuffer(reader.read(IMAGE_HEIGHT*IMAGE_WIDTH*3), dtype=numpy.uint8)
            values = values.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
            values = numpy.asarray(values, numpy.float32)
            data.append(values)
        return data,label, numImages


def inference(images):

    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(name='weights', initial_value=tf.random_normal([5, 5, 3, 64], stddev=5e-2),
                             dtype=tf.float32)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(name='biases', initial_value=tf.zeros([64]), dtype=tf.float32)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
    return conv1


def train():
    data, label, num_images = extract_images()
    with tf.Graph().as_default():
        images = tf.placeholder(tf.float32, [num_images, IMAGE_WIDTH ,IMAGE_HEIGHT,3])
        logits = inference(images)

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            start_time = time.time()
            sess.run(logits, feed_dict={images: data})
            duration = time.time() - start_time
            print(duration)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    tf.app.run()
