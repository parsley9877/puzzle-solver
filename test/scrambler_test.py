import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Tuple
from scrambler import ImageScrambler

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


def write_summary(path: str, sampled_image: np.ndarray, scrambled_image: np.ndarray, segment_shape: Tuple) -> None:

    if not os.path.exists(os.path.join(path, 'Scrambled Image')):
        os.mkdir(os.path.join(path, 'Scrambled Image'))

    if not os.path.exists(os.path.join(path, 'Log Scrambler')):
        os.mkdir(os.path.join(path, 'Log Scrambler'))

    fig1, ax1 = plt.subplots()
    if len(sampled_image.shape) == 2:
        ax1.imshow(sampled_image.reshape([sampled_image.shape[0], sampled_image.shape[1], 1]))
    else:
        ax1.imshow(sampled_image.reshape([sampled_image.shape[0], sampled_image.shape[1], sampled_image.shape[2]]))
    fig1.savefig(os.path.join(path, 'Scrambled Image', 'original.png'))

    fig2, ax2 = plt.subplots()
    if len(scrambled_image.shape) == 2:
        ax2.imshow(scrambled_image.reshape([scrambled_image.shape[0], scrambled_image.shape[1], 1]))
    else:
        ax2.imshow(scrambled_image.reshape([scrambled_image.shape[0], scrambled_image.shape[1], scrambled_image.shape[2]]))
    fig2.savefig(os.path.join(path, 'Scrambled Image', 'scrambled.png'))

    log_obj = open(os.path.join(path, 'Log Scrambler', 'log.txt'), mode='w')
    log_obj.write('Original Image Shape: ' + str(sampled_image.shape) + '\n')
    log_obj.write('Segments Shape: ' + str(segment_shape) + '\n')
    log_obj.close()

def get_one_sample(dataset) -> np.ndarray:
    (X, y), (X_test, y_test) = dataset.load_data()
    n = np.random.choice(X.shape[0], 1)
    if len(X.shape) == 4:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        sampled_image = (((X[n[[0]]].reshape(X.shape[1], X.shape[2], X.shape[3]) / 255.0) - 0.5) / 0.5).astype(np.float32)
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        sampled_image = (((X[n[[0]]].reshape(X.shape[1], X.shape[2], 1) / 255.0) - 0.5) / 0.5).astype(np.float32)
    return sampled_image


if not os.path.exists('./cifar10-results'):
    os.mkdir('cifar10-results')
cifar = keras.datasets.cifar10
n = 76
segment_shape = (2, 2, 3)
sampled_image = get_one_sample(cifar)
scrambler_object = ImageScrambler(shape=(32, 32), block_shape=segment_shape)
scrambled_image = scrambler_object.scramble(sampled_image)
write_summary(path='./cifar10-results', sampled_image=sampled_image, scrambled_image=scrambled_image, segment_shape=segment_shape)

if not os.path.exists('./mnist-results'):
    os.mkdir('mnist-results')
mnist = keras.datasets.mnist
n = 32
segment_shape = (7, 7, 1)
sampled_image = get_one_sample(mnist)
scrambler_object = ImageScrambler(shape=(28, 28), block_shape=segment_shape)
scrambled_image = scrambler_object.scramble(sampled_image)
write_summary(path='./mnist-results', sampled_image=sampled_image, scrambled_image=scrambled_image, segment_shape=segment_shape)