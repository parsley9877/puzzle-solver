import numpy as np
from typing import Tuple

class ImageScrambler():

    def __init__(self, shape: Tuple, block_shape: Tuple) -> None:
        assert shape[0] % block_shape[0] == 0 and shape[1] % block_shape[1] == 0
        self.shape = shape
        self.block_shape = block_shape

    def scramble(self, image: np.ndarray) -> np.ndarray:
        horizontal_num = self.shape[1] // self.block_shape[1]
        vertical_num = self.shape[0] // self.block_shape[0]
        scrambled_image = np.zeros_like(image)
        num_segments = horizontal_num * vertical_num
        rand_segments = np.random.choice(num_segments, size=num_segments, replace=False)
        patches = []
        for init_segment, dist_segment in enumerate(rand_segments):
            init_patch = self.get_patch(init_segment)
            dist_patch = self.get_patch(dist_segment)
            new_segment = image[init_patch[0]:init_patch[0] + self.block_shape[0], init_patch[1]:init_patch[1] + self.block_shape[1], :]
            scrambled_image[dist_patch[0]:dist_patch[0] + self.block_shape[0], dist_patch[1]:dist_patch[1] + self.block_shape[1], :] = new_segment

        return scrambled_image

    def get_patch(self, segment_num: int) -> Tuple[int, int]:
        horizontal_num = self.shape[1] / self.block_shape[1]
        vertical_num = self.shape[0] / self.block_shape[0]
        row_patch = segment_num // horizontal_num
        col_patch = segment_num % horizontal_num
        row_patch_idx = row_patch * self.block_shape[0]
        col_patch_idx = col_patch * self.block_shape[1]
        return int(row_patch_idx), int(col_patch_idx)


if __name__ == '__main__':
    s = ImageScrambler(shape=(9, 9), block_shape=(3, 3))
    image = np.arange(81).reshape([9, 9, 1])
    print(image.reshape([9, 9]))
    print(s.scramble(image).reshape([9, 9]))
