import argparse
import os.path
import time
from datetime import datetime

import numpy as np
from scipy import misc
import tensorflow as tf

CHECKPOINT_FILENAME = "point.ckpt"
CHECKPOINT_DIR = "./PointData"
IMAGE_WIDTH = 18
IMAGE_HEIGHT = 18
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
NUM_CLASSES = 12

FLAGS = tf.app.flags.FLAGS


def create_image_set(image, row,step):
    images = []
    for y in range(row,row+step):
        for x in range(int(FLAGS.left), int(FLAGS.right) - IMAGE_WIDTH):
            sub = image[y:y + IMAGE_HEIGHT, x:x + IMAGE_WIDTH]
            images.append(sub)
    return images


def inference(images):
    kernel1_size = 5
    filter1_size = 10
    level1 = 300
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(name='weights',
                             initial_value=tf.random_normal([kernel1_size, kernel1_size, 3, filter1_size], stddev=0.04,
                                                            dtype=tf.float32))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(name='biases', initial_value=tf.zeros([filter1_size]), dtype=tf.float32)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('local1') as scope:
        reshape = tf.reshape(norm1, [norm1.get_shape()[0].value, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(name='weights', initial_value=tf.random_normal([dim, level1], stddev=0.04),
                              dtype=tf.float32)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0.1, shape=[level1], dtype=tf.float32))
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(name='weights',
                              initial_value=tf.random_normal([level1, NUM_CLASSES], stddev=1.0 / level1),
                              dtype=tf.float32)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0, shape=[NUM_CLASSES], dtype=tf.float32))
        mat_mul = tf.matmul(local1, weights)
        softmax_linear = tf.add(mat_mul, biases, name=scope.name)

    return softmax_linear


def search():
    with tf.Graph().as_default():
        step = 1
        image = misc.imread(FLAGS.image)
        if not FLAGS.left:
            left = 0
        else:
            left = int(FLAGS.left)
        if not FLAGS.right:
            FLAGS.right = image.shape[1]
        else:
            right = int(FLAGS.right)
        if not FLAGS.top:
            top = 0
        else:
            top = int(FLAGS.top)
        if not FLAGS.bottom:
            FLAGS.bottom = image.shape[0]
        else:
            bottom = int(FLAGS.bottom)

        images = tf.placeholder(np.float32, [step*(right - left - IMAGE_WIDTH), IMAGE_WIDTH, IMAGE_HEIGHT, 3], name="images")
        logits = inference(images)
        reduced = tf.reduce_max(logits, reduction_indices=1)
        label = tf.arg_max(logits, dimension=1)
        found = tf.greater(label, 0)

        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
        saver.restore(sess, checkpoint_path)

        for y in range(top, bottom - IMAGE_HEIGHT, step):
            images_set = create_image_set(image, y,step)
            reduced_val, label_val, found_val, logits_value = sess.run(fetches=[reduced, label, found, logits],
                                                                       feed_dict={images: images_set})
            found_positive = np.nonzero(label_val)
            if len(found_positive[0]) > 0:
                for i in np.nditer(found_positive):
                    str_logistic = "{"
                    for li in range(NUM_CLASSES):
                        str_logistic += str(logits_value[i][li]) + ", "
                    str_logistic += "}"
                    print("(%d,%d) found label %d with strengths  %s" % (y, i + left, label_val[i], str_logistic))


# noinspection PyUnusedLocal
def main(argv=None):  # pylint: disable=runused-argument
    search()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--image", required=True, help='The image file')
    parser.add_argument('-l', "--left", help='The left border inside which perform the analise')
    parser.add_argument('-r', "--right", help='The right border inside which perform the analise')
    parser.add_argument('-t', "--top", help='The top border inside which perform the analise')
    parser.add_argument('-b', "--bottom", help='The top border inside which perform the analise')
    FLAGS = parser.parse_args()
    tf.app.run()
