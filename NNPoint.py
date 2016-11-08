import os.path
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

INPUT_FILE = "samples.img"
CHECKPOINT_FILENAME = "point.ckpt"
TRAIN_DIR = "./tensordboard_data"
CHECKPOINT_DIR = "./PointData"
IMAGE_WIDTH = 18
IMAGE_HEIGHT = 18
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
NUM_CLASSES = 12
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.5  # Initial learning rate.

MAX_STEPS = 100000

num_images = 0


def evaluation(logits, labels):
    int_label = tf.cast(labels, tf.int32)
    correct = tf.nn.in_top_k(logits, int_label, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('<')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images():
    with open("samples.img", "rb") as reader:
        global IMAGE_HEIGHT
        global IMAGE_WIDTH
        global num_images
        reader.seek(0, os.SEEK_END)
        size = reader.tell()
        reader.seek(0, os.SEEK_SET)
        IMAGE_WIDTH = _read32(reader)
        IMAGE_HEIGHT = _read32(reader)
        print(size)
        print(IMAGE_WIDTH)
        num_images = int((size - 8) / (IMAGE_WIDTH * IMAGE_HEIGHT * 3 + 1))
        label = np.array(np.frombuffer(reader.read(1), dtype=np.uint8), dtype=np.uint8)
        values = np.frombuffer(reader.read(IMAGE_HEIGHT * IMAGE_WIDTH * 3), dtype=np.uint8)
        values = values.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        values = np.asarray(values, np.float32)
        data = [values]
        for i in range(1, int(num_images)):
            label = np.append(label, [np.frombuffer(reader.read(1), dtype=np.uint8)])
            values = np.frombuffer(reader.read(IMAGE_HEIGHT * IMAGE_WIDTH * 3), dtype=np.uint8)
            values = values.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
            values = np.asarray(values, np.float32)
            data.append(values)

        return data, label, num_images


def _activation_summary(x):
    tf.histogram_summary('/activations', x)
    tf.scalar_summary('/sparsity', tf.nn.zero_fraction(x))


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

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _train(total_loss, global_step):
    # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    #                                 global_step,
    #                                 1000,
    #                                 LEARNING_RATE_DECAY_FACTOR,
    #                                 staircase=True)
    lr = 0.5

    tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdadeltaOptimizer(lr)
        # opt = tf.train.AdagradOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    #
    # # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train():
    global num_images
    data, label, num_images = extract_images()
    label_sparse = np.zeros([num_images, NUM_CLASSES])
    for i in range(0, num_images):
        label_sparse[i, label[i]] = 1
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        images = tf.placeholder(np.float32, [num_images, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name="images")
        labels = tf.placeholder(np.float32, [None, NUM_CLASSES], name="labels")
        y__ = tf.placeholder(tf.float32, [int(num_images)], name="y__")
        logits, conv = inference(images)
        eval_func = evaluation(logits, y__)
        loss, cross_entropy = calculate_loss(logits, labels)
        train_op = _train(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        sess = tf.Session()
        if FLAGS.START:
            init = tf.initialize_all_variables()
            # Start running operations on the Graph.
            sess.run(init)
        else:
            saver.restore(sess, checkpoint_path)
            print("Model restored.")


        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)

        for step in range(0, MAX_STEPS):
            start_time = time.time()

            _, loss_value, cross_entropy_value, logits_value = sess.run(fetches=[train_op, loss, cross_entropy, logits],
                                                                        feed_dict={images: data, labels: label_sparse,
                                                                                   y__: label})
            duration = time.time() - start_time

            #            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 200 == 1:
                num_examples_per_step = num_images
                examples_per_sec = num_examples_per_step / duration
                summary_str, logits_val, eval_func_value = sess.run((summary_op, logits, eval_func),
                                                                    feed_dict={images: data, labels: label_sparse,
                                                                               y__: label})
                summary_writer.add_summary(summary_str, step - 40)
                format_str = '%s: step %d, loss = %.5f, (%.1f examples/sec)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec))
                format_str = 'Evaluation: %d (%d)'
                calc = np.argmax(logits_value, 1)
                amax = np.amax(logits_value,1)
                val = num_images - np.count_nonzero(calc == label)
                print(format_str % (val, eval_func_value))
                print(logits_value[0])
            # if step % 200 == 0:
            #     conv_val = sess.run(conv, feed_dict={images: data, labels: label_sparse, y__: label})
            #     conv_val = conv_val.swapaxes(1, 3)
            #     saveImageConv(data, conv_val, label);

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == MAX_STEPS:
                saver.save(sess, checkpoint_path)


# noinspection PyUnusedLocal
def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.DeleteRecursively(TRAIN_DIR)
    tf.gfile.MakeDirs(TRAIN_DIR)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S',"--START", help='Start a learning session')
    FLAGS = parser.parse_args()
    tf.app.run()
