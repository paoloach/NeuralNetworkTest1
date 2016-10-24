from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

INPUT_FILE = "samples.img"
IMAGE_WIDTH = 14
IMAGE_HEIGHT = 14
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3
NUM_CLASSES = 12
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

MAX_STEPS = 100000

FLAGS = tf.app.flags.FLAGS
num_images = 0

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('<')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images():
    with open("samples.img", "rb") as reader:
        global IMAGE_HEIGHT
        global IMAGE_WIDTH
        reader.seek(0, os.SEEK_END)
        size = reader.tell()
        reader.seek(0, os.SEEK_SET)
        IMAGE_WIDTH = _read32(reader)
        IMAGE_HEIGHT = _read32(reader)
        print (size)
        print (IMAGE_WIDTH)
        numImages = (size - 8) / (IMAGE_WIDTH * IMAGE_HEIGHT * 3 + 1)
        label = np.array(np.frombuffer(reader.read(1), dtype=np.uint8), dtype=np.uint8)
        values = np.frombuffer(reader.read(IMAGE_HEIGHT * IMAGE_WIDTH * 3), dtype=np.uint8)
        values = values.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
        values = np.asarray(values, np.float32)
        data = [values]
        for i in range(1, int(numImages)):
            label = np.append(label, [np.frombuffer(reader.read(1), dtype=np.uint8)])
            values = np.frombuffer(reader.read(IMAGE_HEIGHT * IMAGE_WIDTH * 3), dtype=np.uint8)
            values = values.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
            values = np.asarray(values, np.float32)
            data.append(values)
        return data, np.asarray(label, np.float32), numImages


def _activation_summary(x):
    tf.histogram_summary('/activations', x)
    tf.scalar_summary('/sparsity', tf.nn.zero_fraction(x))


def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(name='weights',
                             initial_value=tf.random_normal([5, 5, 3, 64], stddev=5e-2, dtype=tf.float32))
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(name='biases', initial_value=tf.zeros([64]), dtype=tf.float32)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm1, [norm1.get_shape()[0].value, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(name='weights', initial_value=tf.random_normal([dim, 384], stddev=0.04), dtype=tf.float32)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0.1, shape=[384], dtype=tf.float32))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(name='weights', initial_value=tf.random_normal([384, NUM_CLASSES], stddev=1 / 384.0),
                              dtype=tf.float32)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0, shape=[NUM_CLASSES], dtype=tf.float32))
        matMul = tf.matmul(local3, weights)
        softmax_linear = tf.add(matMul, biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def calculate_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
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
    """
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    1000,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
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
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images = tf.placeholder(np.float32, [num_images, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name="images")
        labels = tf.placeholder(np.float32, [num_images], name="labels")
        logits = inference(images)
        loss = calculate_loss(logits, labels)
        train_op = _train(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(MAX_STEPS):
            start_time = time.time()

            _, loss_value = sess.run(fetches=[train_op, loss], feed_dict={images: data, labels: label})
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = num_images
                examples_per_sec = num_examples_per_step / duration

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec)')
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec))

                # if step % 100 == 0:
                #     summary_str = sess.run(summary_op, feed_dict={images: data, labels: label})
                #     summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
