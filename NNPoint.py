import argparse
import os.path
import time
from datetime import datetime
from scipy import  misc

import numpy as np
import tensorflow as tf

from BachData import BachData

INPUT_FILE = "samples.img"
CHECKPOINT_FILENAME = "point.ckpt"
TRAIN_DIR = "./tensordboard_data"
CHECKPOINT_DIR = "./PointData"
IMAGE_WIDTH = 18
IMAGE_HEIGHT = 18
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
NUM_CLASSES = 1
BACH_VALID_SIZE = 300
BACH_WRONG_SIZE = 600

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
MAX_STEPS = 1000000


def evaluation(logits, labels):
    int_label = tf.cast(labels, tf.int32)
    correct = tf.nn.in_top_k(logits, int_label, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('<')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images():
    with open("samples.img", "rb") as reader:
        with open("samples.label", "rb") as label_reader:
            global IMAGE_HEIGHT
            global IMAGE_WIDTH
            global NUM_CLASSES
            reader.seek(0, os.SEEK_END)
            size = reader.tell()
            reader.seek(0, os.SEEK_SET)
            print(size)
            num_images = int(size / (IMAGE_WIDTH * IMAGE_HEIGHT * 3))
            data = np.frombuffer(reader.read(size), dtype=np.uint8)
            data = np.reshape(data, (num_images, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
            label = np.frombuffer(label_reader.read(num_images), dtype=np.uint8)
            label_valid = []
            label_wrong = []
            for i in range(0, int(num_images)):
                if label[i] > 0:
                    misc.imsave("tmp/" + str(label[i]) + "-img_orig_" + str(i) + ".png", data[i])
                    if label[i] > NUM_CLASSES:
                        NUM_CLASSES = label[i]
                    label_valid = np.append(label_valid, i)
                else:
                    label_wrong = np.arange(i + 1, num_images)
                    break
            NUM_CLASSES += 1

    bachData = BachData(data, label, label_valid, NUM_CLASSES, BACH_VALID_SIZE)
    bachWrongData = BachData(data, label, label_wrong, NUM_CLASSES, BACH_WRONG_SIZE)

    return bachData, bachWrongData


def add_weight_decay(variable, param):
    weight_decay = tf.mul(tf.nn.l2_loss(variable), param, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)


def inference(images):
    kernel1_size = 5
    filter1_size = 30
    level1 = 300
    with tf.variable_scope('conv1') as scope:
        kernelWeights = tf.Variable(name='weights',
                                    initial_value=tf.random_normal([kernel1_size, kernel1_size, 3, filter1_size],
                                                                   stddev=0.04, dtype=tf.float32))
        add_weight_decay(kernelWeights, 0)

        conv = tf.nn.conv2d(images, kernelWeights, [1, 1, 1, 1], padding='VALID')
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
        add_weight_decay(weights, 0.04)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0.1, shape=[level1], dtype=tf.float32))
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(name='weights',
                              initial_value=tf.random_normal([level1, NUM_CLASSES], stddev=1.0 / level1),
                              dtype=tf.float32)
        add_weight_decay(weights, 0.04)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0, shape=[NUM_CLASSES], dtype=tf.float32))
        mat_mul = tf.matmul(local1, weights)
        softmax_linear = tf.add(mat_mul, biases, name=scope.name)

    return softmax_linear, conv1


def calculate_loss(logits, labels):
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,labels))

    # labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), cross_entropy


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    return loss_averages_op


def _train(total_loss, global_step):
    tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdadeltaOptimizer()
        # opt = tf.train.AdagradOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train():
    bach_data, bach_wrong_data = extract_images()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images = tf.placeholder(np.float32, [BACH_VALID_SIZE + BACH_WRONG_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3],
                                name="images")
        label_sparse = tf.placeholder(np.float32, [None, NUM_CLASSES], name="labels")
        y__ = tf.placeholder(tf.float32, [int(BACH_VALID_SIZE + BACH_WRONG_SIZE)], name="y__")
        logits, conv = inference(images)
        eval_func = evaluation(logits, y__)
        loss, cross_entropy = calculate_loss(logits, label_sparse)
        train_op = _train(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        sess = tf.Session()
        start = 0
        if FLAGS.START:
            init = tf.initialize_all_variables()
            sess.run(init)
        else:
            saver.restore(sess, checkpoint_path)
            print("Model restored.")
            start = global_step.eval(sess)

        tf.train.start_queue_runners(sess=sess)

        # summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)
        data, label_spar, labels = bach_data.get_bach_data()
        data_wrong, labels_wrong_sparse, labels_wrong = bach_wrong_data.get_bach_data()
        data = np.append(data, data_wrong, 0)
        label_spar = np.append(label_spar, labels_wrong_sparse, 0)
        labels = np.append(labels, labels_wrong)
        for step in range(start, MAX_STEPS):
            start_time = time.time()
            _, loss_value = sess.run(fetches=[train_op, loss],
                                     feed_dict={images: data,
                                                label_sparse: label_spar,
                                                y__: labels})
            duration = time.time() - start_time

            if step % 200 == 0:
                data, label_spar, labels = bach_data.get_bach_data()
                data_wrong, labels_wrong_sparse, labels_wrong = bach_wrong_data.get_bach_data()
                data = np.append(data, data_wrong, 0)

                label_spar = np.append(label_spar, labels_wrong_sparse, 0)
                labels = np.append(labels, labels_wrong)
                num_examples_per_step = BACH_VALID_SIZE + BACH_WRONG_SIZE
                examples_per_sec = num_examples_per_step / duration
                eval_func_value = sess.run((eval_func),
                                           feed_dict={images: data, label_sparse: label_spar,
                                                      y__: labels})
                format_str = '%s: step %d, loss = %.5f, eval= %d, (%.1f examples/sec)'
                print(format_str % (datetime.now(), step, loss_value, eval_func_value, examples_per_sec))

            # Save the model checkpoint periodically.
            if step % 200 == 199 or (step + 1) == MAX_STEPS:
                saver.save(sess, checkpoint_path)


# noinspection PyUnusedLocal
def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.DeleteRecursively(TRAIN_DIR)
    tf.gfile.MakeDirs(TRAIN_DIR)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', "--START", help='Start a learning session')
    FLAGS = parser.parse_args()
    tf.app.run()
