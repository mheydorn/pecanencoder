""" 
Auto Encoder to detect foregn material in pecans.
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import glob
import TensorflowUtils as utils
import cv2

#Import dataset
imgPaths = glob.glob("dataset/pecan_images_no_contrast/clean2/*") # Some images
filenameQ = tf.train.string_input_producer(imgPaths)

# Define a subgraph that takes a filename, reads the file, decodes it, and                                                                                     
# enqueues it.                                                                                                                                                 
filename = filenameQ.dequeue()
image_bytes = tf.read_file(filename)
decoded_image = tf.image.decode_png(image_bytes)
image_queue = tf.FIFOQueue(128, dtypes = [tf.uint8], shapes = [(96,96,3)])
#decoded_image = tf.dtypes.cast(decoded_image, tf.float32)
enqueue_op = image_queue.enqueue(decoded_image)

NUM_THREADS = 8
queue_runner = tf.train.QueueRunner(
    image_queue,
    [enqueue_op] * NUM_THREADS,  # Each element will be run from a separate thread.                                                                                       
    image_queue.close(),
    image_queue.close(cancel_pending_enqueues=True),
    )

tf.train.add_queue_runner(queue_runner)

# Training Hyper Parameters
learning_rate = 5e-4
num_steps = 30000
batch_size = 100

display_step = 1000
examples_to_show = 10

#Other Hyper Parameters
IMAGE_SIZE = 96
OUTPUT_CHANNELS = 3

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def encode_decode(image, keep_prob):
    with tf.variable_scope("encode_decode"):

        #conv1
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([3, 3, 3, 16])
            b_conv1 = bias_variable([16])
            h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)


        # conv2
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([3, 3, 16, 32])
            b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(max_pool_2x2(h_conv2))

        # conv3
        with tf.variable_scope('conv3'):
            W_conv3 = weight_variable([3, 3, 32, 64])
            b_conv3 = bias_variable([64])
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)


        # conv4
        with tf.variable_scope('conv4'):
            W_conv4 = weight_variable([3, 3, 64, 128])
            b_conv4 = bias_variable([128])
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

        #Upscale
        deconv_shape1 = h_pool3.get_shape()

        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, 128], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(h_pool4, W_t1, b_t1, output_shape=tf.shape(h_pool3))
        fuse_1 = tf.add(conv_t1, h_pool3, name="fuse_1")

        deconv_shape2 = h_pool2.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(h_pool2))
        fuse_2 = tf.add(conv_t2, h_pool2, name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], OUTPUT_CHANNELS])
        W_t3 = utils.weight_variable([16, 16, OUTPUT_CHANNELS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([OUTPUT_CHANNELS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

    return conv_t3

#Define placeholders
keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, OUTPUT_CHANNELS], name="annotation")
outputImage = encode_decode(image, keep_probability)

#Define train_op, loss, and optimizer
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

loss = tf.reduce_mean(tf.math.abs(tf.math.subtract(annotation, outputImage)))

trainable_var = tf.trainable_variables()
train_op = train(loss, trainable_var)

#Create and initilize session
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator() #Coordinator for the queue runner which reads in the images
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #Training loop
    MAX_ITERATION = 1000
    for itr in xrange(MAX_ITERATION):

        imgs = image_queue.dequeue_many(batch_size)
        train_images = imgs.eval()
        train_labels = train_images.copy()


        feed_dict = {image: train_images, annotation: train_labels, keep_probability: 0.85}

        output, _, train_loss = sess.run([outputImage, train_op, loss], feed_dict=feed_dict)
        
        print("loss =", train_loss )
        if itr % 10 == 0:
            print("Saving")
            saver.save(sess, "model@" + str(itr) + ".ckpt", itr)
            plt.subplot(1, 2, 1)
            plt.imshow(np.clip(output[0].astype(np.uint8), a_min = 0, a_max = 255 ))
            plt.subplot(1, 2, 2)
            plt.imshow(np.clip(train_labels[0].astype(np.uint8), a_min = 0, a_max = 255 ))
            plt.show()
            

























