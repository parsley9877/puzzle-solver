import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
from scrambler import ImageScrambler
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

mnist = keras.datasets.mnist
(X, y), (X_test, y_test) = mnist.load_data()
X = X.reshape(X.shape[0], 28, 28, 1)
n = 10
segment_shape = (14, 14, 1)
sampled_image = (((X[10].reshape(28, 28, 1) / 255.0) - 0.5) / 0.5).astype(np.float32)

scrambler_object = ImageScrambler(shape=(28, 28), block_shape=segment_shape)
scrambled_image = scrambler_object.scramble(sampled_image)

if not os.path.exists('./Scrambled Image'):
    os.mkdir('./Scrambled Image')

if not os.path.exists('./Log Scrambler'):
    os.mkdir('./Log Scrambler')

fig1, ax1 = plt.subplots()
ax1.imshow(sampled_image.reshape([28, 28]), cmap='gray')
fig1.savefig('./Scrambled Image/original.png')

fig2, ax2 = plt.subplots()
ax2.imshow(scrambled_image.reshape([28, 28]), cmap='gray')
fig2.savefig('./Scrambled Image/scrambled.png')

log_obj = open('./Log Scrambler/log.txt', mode='w')
log_obj.write('Original Image Shape: ' + str(sampled_image.shape) + '\n')
log_obj.write('Segments Shape: ' + str(segment_shape) + '\n')
log_obj.close()
