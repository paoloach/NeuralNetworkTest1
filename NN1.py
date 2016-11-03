from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import math
import tensorflow as tf
import tensorflow.contrib.learn as tf_learn
import tensorflow.contrib.layers as tf_layer
import numpy as np
from scipy import ndimage,misc
from six.moves import xrange  # pylint: disable=redefined-builtin

INPUT_FILE = "samples.img"
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

FLAGS = tf.app.flags.FLAGS
num_images = 0

TRAIN_DIR = "./tensordboard_data"
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
def evaluation(logits, labels):
    int_label = tf.cast(labels, tf.int32)
    correct = tf.nn.in_top_k(logits,int_label, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('<')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def printLabelStatistics(labels):
    stat = dict()
    for label in labels:
        if label not in stat:
            stat[label] = 0
        stat[label] += 1
    print(stat)


def extract_images():
    with open("samples.img", "rb") as reader:
        global IMAGE_HEIGHT
        global IMAGE_WIDTH
        reader.seek(0, os.SEEK_END)
        size = reader.tell()
        reader.seek(0, os.SEEK_SET)
        IMAGE_WIDTH = _read32(reader)
        IMAGE_HEIGHT = _read32(reader)
        print(size)
        print(IMAGE_WIDTH)
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

        printLabelStatistics(label)
        return data, np.asarray(label, np.float32), numImages


def _activation_summary(x):
    tf.histogram_summary('/activations', x)
    tf.scalar_summary('/sparsity', tf.nn.zero_fraction(x))


def add_weight_decay(variable, param):
    weight_decay = tf.mul(tf.nn.l2_loss(variable), param, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)


def inference(images):
    filter1 = 10
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(name='weights',
                             initial_value=tf.random_normal([5, 5, 3, filter1], stddev=0.04, dtype=tf.float32))
        add_weight_decay(kernel,0)

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
        weights = tf.Variable(name='weights', initial_value=tf.random_normal([dim, 384], stddev=0.04), dtype=tf.float32)
        add_weight_decay(weights, 0.04)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0.1, shape=[384], dtype=tf.float32))
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        tf.histogram_summary('/activations/local1', local1)
        tf.scalar_summary('/sparsity/local1', tf.nn.zero_fraction(local1))

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(name='weights', initial_value=tf.random_normal([384, NUM_CLASSES], stddev=1 / 384.0),
                              dtype=tf.float32)
        add_weight_decay(weights, 0.04)
        biases = tf.Variable(name='biases', initial_value=tf.constant(value=0, shape=[NUM_CLASSES], dtype=tf.float32))
        matMul = tf.matmul(local1, weights)
        softmax_linear = tf.add(matMul, biases, name=scope.name)
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
    return tf.add_n(tf.get_collection('losses'), name='total_loss'),cross_entropy


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
    lr = 2

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(total_loss)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        #opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdadeltaOptimizer(lr)
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

    #return train_step

def saveImageConv(data,conv):
    if tf.gfile.Exists("conv"):
        tf.gfile.DeleteRecursively("conv")
    tf.gfile.MakeDirs("conv")
    allConv = np.reshape(conv, (conv.shape[0],conv.shape[1] * conv.shape[2], conv.shape[3]))
    for imageCount in range(0, conv.shape[0]):
        imagefile = "conv/conv_"+str(imageCount) + ".png"
        misc.imsave(imagefile,allConv[imageCount])
        imagefile = "conv/orig_"+str(imageCount)+".png"
        misc.imsave(imagefile, data[imageCount])

def train1():
    global num_images
    data, label, num_images = extract_images()
    labelSparse = np.zeros([num_images, NUM_CLASSES])
    for i in range(0, int(num_images)):
        labelSparse[i, label[i]] = 1
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        images = tf.placeholder(np.float32, [num_images, IMAGE_WIDTH, IMAGE_HEIGHT, 3], name="images")
        labels = tf.placeholder(np.float32, [None, NUM_CLASSES], name="labels")
        y__ = tf.placeholder(tf.float32, [int(num_images)], name="y__")
        logits, conv = inference(images)
        loss, cross_entropy = calculate_loss(logits, labels)
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

        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)

        for step in xrange(MAX_STEPS):
            start_time = time.time()

            _, loss_value, cross_entropy_value, logits_value = sess.run(fetches=[train_op, loss, cross_entropy,logits],
                                                   feed_dict={images: data, labels: labelSparse,y__: label})
            duration = time.time() - start_time

            #            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = num_images
                examples_per_sec = num_examples_per_step / duration
                summary_str = sess.run(summary_op, feed_dict={images: data, labels: labelSparse,y__: label})
                summary_writer.add_summary(summary_str, step - 40)
                format_str = ('%s: step %d, loss = %.5f, (%.1f examples/sec)')
                print(format_str % (datetime.now(), step, loss_value,examples_per_sec))
            if step % 200 == 199:
                conv_val = sess.run(conv, feed_dict={images: data, labels: labelSparse, y__: label})
                conv_val = conv_val.swapaxes(1, 3)
                saveImageConv(data, conv_val);

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def train2():
    global num_images
    data, label, num_images = extract_images()
    labels = np.zeros([num_images, NUM_CLASSES])
    for i in range(0, int(num_images)):
        labels[i, label[i]] = 1
    image_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3
    data_flat = np.array(data).reshape(num_images, image_size)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images = tf.placeholder(np.float32, [None, image_size], name="images")

        W = tf.Variable(tf.zeros([image_size, NUM_CLASSES]), name="W")
        b = tf.Variable(tf.zeros([NUM_CLASSES]), name="b")
        y = tf.matmul(images, W, name="matmul_image_W") + b

        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="y_")

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        tf.histogram_summary('/activations/w', W)
        tf.scalar_summary('/sparsity/w', tf.nn.zero_fraction(W, name="summary_W"))
        tf.histogram_summary('/activations/b', b)
        tf.scalar_summary('/sparsity/b', tf.nn.zero_fraction(b, name="summary_B"))

        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)
        for step in range(1000):
            _, cross_entropy_value = sess.run([train_step, cross_entropy], feed_dict={images: data_flat, y_: labels})
            if step % 10 == 0:
                summary_str = sess.run(summary_op, feed_dict={images: data_flat, y_: labels})
                summary_writer.add_summary(summary_str, step)
                result = tf.argmax(y, 1)
                correct_prediction = tf.equal(result, tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                accuracy_value = sess.run([accuracy], feed_dict={images: data_flat, y_: labels})
                print(accuracy_value)




def train3():
    global num_images
    data, label, num_images = extract_images()
    labels = np.zeros([num_images, NUM_CLASSES])
    for i in range(0, int(num_images)):
        labels[i, label[i]] = 1
    image_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3
    data_flat = np.array(data).reshape(num_images, image_size)
    hidden1Neurons = 128
    hidden2Neurons = 32
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        images = tf.placeholder(np.float32, [None, image_size], name="images")

        w1 = tf.Variable(
            tf.truncated_normal([image_size, hidden1Neurons],
                                stddev=1.0 / math.sqrt(float(image_size))), name="w1")
        b1 = tf.Variable(tf.zeros([hidden1Neurons]), name="b1")
        hidden1 = tf.nn.relu(tf.matmul(images, w1) + b1)

        w2 = tf.Variable(
            tf.truncated_normal([hidden1Neurons, hidden2Neurons], stddev=1.0 / math.sqrt(float(hidden1Neurons))),
            name="w2")
        b2 = tf.Variable(tf.zeros([hidden2Neurons]), name="b2", trainable=True)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)

        w3 = tf.Variable(
            tf.truncated_normal([hidden2Neurons, NUM_CLASSES], stddev=1.0 / math.sqrt(float(hidden2Neurons))),
            name='w3')
        b3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='b3')
        y = tf.matmul(hidden2, w3) + b3

        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="y_")
        y__ = tf.placeholder(tf.float32, [int(num_images)], name="y__")
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

        loss_op = [w1,b1,w2,b2,w3,b3,y]
        with tf.control_dependencies(loss_op):
            optimizer = tf.train.GradientDescentOptimizer(1)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_step = optimizer.minimize(cross_entropy, global_step=global_step)
        tf.histogram_summary('/activations/w', w1)
        tf.scalar_summary('/sparsity/w', tf.nn.zero_fraction(w1, name="summary_W"))
        tf.histogram_summary('/activations/b', b1)
        tf.scalar_summary('/sparsity/b', tf.nn.zero_fraction(b1, name="summary_B"))

        summary_op = tf.merge_all_summaries()
        eval = evaluation(y, y__)
        sess = tf.InteractiveSession()
        # Train
        tf.initialize_all_variables().run()
        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)
        for step in range(1000):
            _, cross_entropy_value = sess.run([train_step, cross_entropy],
                                              feed_dict={images: data_flat, y_: labels, y__: label})
            if (step % 10 == 0):
                summary_str = sess.run(summary_op, feed_dict={images: data_flat, y_: labels})
                summary_writer.add_summary(summary_str, step)
                result = tf.argmax(y, 1)
                correct_prediction = tf.equal(result, tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                accuracy_value = sess.run([eval], feed_dict={images: data_flat, y_: labels, y__: label})
                print(cross_entropy_value)


def train4():
    global num_images
    data, label, num_images = extract_images()
    labels = np.zeros([num_images, 1],dtype=np.int32)
    for i in range(0, int(num_images)):
        labels[i] = label[i];
    image_size = IMAGE_WIDTH * IMAGE_HEIGHT * 3
    data_flat = np.array(data,dtype=np.uint8).reshape(num_images, image_size)
    with tf.Graph().as_default():
        feature_columns = [tf_layer.real_valued_column("", dimension=int(image_size),dtype=tf.uint8)]
        classifier = tf_learn.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=[100, 200, 100],
                                                    n_classes=NUM_CLASSES,
                                                    model_dir=TRAIN_DIR)
        for i in range(0,100):
            classifier.fit(x=data_flat, y=labels, steps=100)
            accuracy_score = classifier.evaluate(x=data_flat, y=labels)["accuracy"]
            print('Accuracy: {0:f}'.format(accuracy_score))
            print(classifier.predict(data_flat[0]))


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.DeleteRecursively(TRAIN_DIR)
    tf.gfile.MakeDirs(TRAIN_DIR)
    train1()


if __name__ == '__main__':
    tf.app.run()
