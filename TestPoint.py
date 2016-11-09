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


def create_image_set(image, row):

    images = []
    for x in range(0, image.shape[1] - IMAGE_WIDTH):
       sub = image[row:row+IMAGE_HEIGHT,x:x + IMAGE_WIDTH]
       images.append(sub)
    return images


def add_weight_decay(variable, param):
    weight_decay = tf.mul(tf.nn.l2_loss(variable), param, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)


def inference(images):
    filter1 = 10
    level1 = 300
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(name='weights',
                             initial_value=tf.random_normal([7, 7, 3, filter1], stddev=0.04, dtype=tf.float32))
        add_weight_decay(kernel, 0)

        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(name='biases', initial_value=tf.zeros([filter1]), dtype=tf.float32)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        tf.histogram_summary('/activations/conv1', conv1)
        tf.scalar_summary('/sparsity/conv1', tf.nn.zero_fraction(conv1))

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('local1') as scope:
        reshape = tf.reshape(norm1, [norm1.get_shape()[0].value, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(name='weights', initial_value=tf.random_normal([dim, level1], stddev=0.04),
                              dtype=tf.float32)
        add_weight_decay(weights, 0.04)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0.1, shape=[level1], dtype=tf.float32))
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        tf.histogram_summary('/activations/local1', local1)
        tf.scalar_summary('/sparsity/local1', tf.nn.zero_fraction(local1))

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(name='weights',
                              initial_value=tf.random_normal([level1, NUM_CLASSES], stddev=1.0 / level1),
                              dtype=tf.float32)
        add_weight_decay(weights, 0.04)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0, shape=[NUM_CLASSES], dtype=tf.float32))
        mat_mul = tf.matmul(local1, weights)
        softmax_linear = tf.add(mat_mul, biases, name=scope.name)
        tf.histogram_summary('/activations/softmax_linear', softmax_linear)
        tf.scalar_summary('/sparsity/softmax_linear', tf.nn.zero_fraction(softmax_linear))

    return softmax_linear


def search(filename):
    with tf.Graph().as_default():
        image = misc.imread(filename)
        images = tf.placeholder(np.float32, [image.shape[1]-IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name="images")
        logits = inference(images)
        reduced = tf.reduce_max(logits, reduction_indices=1)
        label = tf.arg_max(logits, dimension=1)
        found = tf.greater(reduced, 8)

        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
        saver.restore(sess, checkpoint_path)


        for y in range(0, image.shape[0] - IMAGE_HEIGHT):
            images_set = create_image_set(image, y)
            reduced_val, label_val, found_val = sess.run(fetches=[reduced, label, found], feed_dict={images: images_set})
            found_positivi = np.nonzero(found_val);
            if (len(found_positivi[0]) > 0):
                for i in np.nditer(found_positivi):
                    print("(%d,%d) found label %d with strength  %f"%(y,i, label_val[i],reduced_val[i]))


# noinspection PyUnusedLocal
def main(argv=None):  # pylint: disable=runused-argument
    if len(argv) < 2:
        print("You must supply a file")
        quit()

    search(argv[1])


if __name__ == '__main__':
    tf.app.run()
