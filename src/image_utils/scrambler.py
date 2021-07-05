import numpy as np
from typing import Tuple, Iterable, Union
import tensorflow as tf
from tensorflow import keras
import os
from image_saver import ImageSaver


class ImageScrambler():
    """
    Description: Class for handling image scrambling
    """

    def __init__(self, shape: Tuple, block_shape: Tuple) -> None:
        """

        :param shape: shape of the original image
        :param block_shape: shape of the segments
        """
        assert shape[0] % block_shape[0] == 0 and shape[1] % block_shape[1] == 0
        self.shape = shape
        self.block_shape = block_shape

    def _scramble(self, image: Union[tf.Tensor]) -> tf.Tensor:
        """
        Description: scramble a rank 3 input image (H, W, C)

        :param image: single image to be scrambled
        :return: scrambled image
        """
        assert tf.rank(image) == 3
        horizontal_num = self.shape[1] // self.block_shape[1]
        vertical_num = self.shape[0] // self.block_shape[0]
        scrambled_image = np.zeros_like(image)
        num_segments = horizontal_num * vertical_num
        rand_segments = np.random.choice(num_segments, size=num_segments, replace=False)
        patches = []
        for init_segment, dist_segment in enumerate(rand_segments):
            init_patch = self._get_patch(init_segment)
            dist_patch = self._get_patch(dist_segment)
            new_segment = image.numpy()[init_patch[0]:init_patch[0] + self.block_shape[0], init_patch[1]:init_patch[1] + self.block_shape[1], :]
            scrambled_image[dist_patch[0]:dist_patch[0] + self.block_shape[0], dist_patch[1]:dist_patch[1] + self.block_shape[1], :] = new_segment

        return tf.convert_to_tensor(scrambled_image, dtype=tf.float32)

    def _get_patch(self, segment_num: int) -> Tuple[int, int]:
        """
        Description: returns the pinch point of a patch with an specific segment number

        :param segment_num: segment number
        :return: pinch point of the corresponding patch (x, y)
        """
        horizontal_num = self.shape[1] / self.block_shape[1]
        vertical_num = self.shape[0] / self.block_shape[0]
        row_patch = segment_num // horizontal_num
        col_patch = segment_num % horizontal_num
        row_patch_idx = row_patch * self.block_shape[0]
        col_patch_idx = col_patch * self.block_shape[1]
        return int(row_patch_idx), int(col_patch_idx)

    def scramble(self, image: tf.Tensor, num: Iterable[int]) -> Iterable:
        """
        Description: returns the scrambled version of a batch of images (B, H, W, C).
        From each image in batch, generates the desired number of scrambled versions.

        :param image: batch of images (B, H, W, C)
        :param num: an iterable with same number of elements with number of images
        :return: a list of all scrambled images
        """
        scrambled_images = []
        for idx, number in enumerate(num):
            for _ in range(number):
                scrambled_images.append(self._scramble(image[idx]))
        return scrambled_images

    def set_shape(self, new_shape):
        """

        :param new_shape: set a new shape config for class
        """
        self.shape = new_shape

    def set_block_shape(self, new_block_shape):
        """

        :param new_block_shape: set a new block_shape config for class
        """
        self.block_shape = new_block_shape

# if __name__ == '__main__':
    # s = ImageScrambler(shape=(28, 28), block_shape=(7, 4))
    # image = np.arange(28*28).reshape([28, 28, 1])
    # print(image.reshape([28, 28]))
    # print(s.scramble(image).reshape([28, 28]))

    # (X, y), (X_test, y_test) = keras.datasets.cifar10.load_data()
    # X = ((X.reshape([X.shape[0], 32, 32, 3]) / 255.0 ) - 0.5 ) / 0.5
    # dataset = tf.data.Dataset.from_tensor_slices((X)).batch(batch_size=20)
    #
    # for batch in dataset:
    #     batch_image = tf.cast(batch, tf.float32)
    #     break
    #
    #
    # path_mnist = './mnist_saved_images'
    # path_mnist_original = os.path.join(path_mnist, 'original batch')
    # path_mnist_scrambled = os.path.join(path_mnist, 'scrambled batch')
    #
    # if not os.path.exists(path_mnist):
    #     os.mkdir(path_mnist)
    #     os.mkdir(path_mnist_original)
    #     os.mkdir(path_mnist_scrambled)
    #
    # names_original = [str(x) for x in np.arange(20)]
    # names_scrambled = [str(x) for x in np.arange(100)]
    #
    # saver = ImageSaver()
    # scrambler = ImageScrambler(shape=(32, 32), block_shape=(8, 8))
    #
    # scrambld_images_list = scrambler.scramble(image=batch_image, num=[int(x) for x in np.ones(20) * 5])
    #
    # scrambled_dataset = tf.data.Dataset.from_tensor_slices(scrambld_images_list).batch(batch_size=100)
    #
    # for batch in scrambled_dataset:
    #     scrambled_batch = tf.cast(batch, tf.float32)
    #     break
    #
    #
    # saver.save(path_mnist_original, batch_image, names_original)
    # saver.save(path_mnist_scrambled, scrambled_batch, names_scrambled)




