import matplotlib.pyplot as plt
from typing import Iterable, Union
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

class ImageSaver():
    """
    Description: a class for handling image saving
    Note: How to handle images with pixel value [-1, 1] ?
    """
    def __init__(self) -> None:
        pass

    def save(self, path: str, images: Union[Iterable[tf.Tensor], tf.Tensor], names: Iterable[str]) -> None:
        """
        Description: it saves a batch or list of images in a desired directory, with a desired naming convention.

        :param path: path to save batch or list of tensor images (H, W, C)
        :param images: a batch or python list of tensor images
        :param names: an iterable of strings for naming saved files
        :return: None
        """
        for img_idx, name in enumerate(names):
            self._save_single_image(path, name, images[img_idx])

    def _save_single_image(self, path: str, file_name: str, image: tf.Tensor) -> None:
        """
        Description: saves a single image with an specific file name, in a desired path

        :param path: path to save a tensor image (H, W, C)
        :param file_name: name of the file
        :param image: a tensor image
        :return: None
        """
        assert tf.rank(image) == 3 and (image.shape[2] == 1 or image.shape[2] == 3)
        fig, ax = plt.subplots()
        if image.shape[2] == 1:
            ax.imshow(image.numpy(), cmap='gray')
        else:
            ax.imshow(image.numpy())
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        fig.savefig(os.path.join(path, file_name))
        plt.close()

# if __name__ == '__main__':

    # (X, y), (X_test, y_test) = keras.datasets.mnist.load_data()
    # batch_image = (((X[0:20].reshape([20, 28, 28, 1])) / 255.0) - 0.5 ) / 0.5
    # print(batch_image.dtype)
    #
    # path_mnist = './mnist_saved_images'
    # path_mnist_numpy = os.path.join(path_mnist, 'from_numpy')
    # path_mnist_numpy_uint = os.path.join(path_mnist_numpy, 'uint')
    # path_mnist_numpy_float = os.path.join(path_mnist_numpy, 'float')
    # path_mnist_tensor = os.path.join(path_mnist, 'from_tensor')
    # path_mnist_tensor_uint = os.path.join(path_mnist_tensor, 'uint')
    # path_mnist_tensor_float = os.path.join(path_mnist_tensor, 'float')
    #
    # if not os.path.exists(path_mnist):
    #     os.mkdir(path_mnist)
    #     os.mkdir(path_mnist_numpy)
    #     os.mkdir(path_mnist_numpy_uint)
    #     os.mkdir(path_mnist_numpy_float)
    #     os.mkdir(path_mnist_tensor)
    #     os.mkdir(path_mnist_tensor_uint)
    #     os.mkdir(path_mnist_tensor_float)

    # dataset = tf.data.Dataset.from_tensor_slices((X)).batch(batch_size=20)
    # for elem in dataset:
    #     tensor_batch_image = elem
    #     break

    # names = [str(x) for x in np.arange(20)]
    #
    # saver = ImageSaver()
    #
    # saver.save(path_mnist_numpy_uint, batch_image, names)


