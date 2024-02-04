# -*- coding: utf-8 -*-
import random
from typing import Optional

import PIL.Image
import numpy as np


def draw_forgetting_box(image, size_ratio=Optional[tuple], mask_value=0):
    assert isinstance(image, np.ndarray)

    if size_ratio is None:
        size_ratio = (random.randint(25, 75) / 100, random.randint(25, 75) / 100)
    w, h = image.shape[:2]
    masked_image = image.copy()
    w_cut, h_cut = (int(w * size_ratio[0]), int(h * size_ratio[1]))  # round down
    x_min, y_min = random.randint(0, w - w_cut), random.randint(0, h - h_cut)
    # x_max, y_max = x_min + w_cut - 1, y_min + h_cut - 1
    if image.shape == 3:
        masked_image[x_min:x_min + w_cut, y_min:y_min + h_cut, :] = mask_value
    else:
        masked_image[x_min:x_min + w_cut, y_min:y_min + h_cut] = mask_value
    # return image, masked_image, round((x_min + x_max) / 2 / w, 6), round((y_min + y_max) / 2 / h, 6), \
    #        round(w_cut / w, 6), round(h_cut / h, 6)
    # center x y w h in ratio
    return image, masked_image, round(x_min / w, 6), round(y_min / h, 6), \
           round(w_cut / w, 6), round(h_cut  / h, 6)


if __name__ == '__main__':
    img = np.arange(0, 20).reshape([1, -1]).repeat(20, axis=0)
    # img = np.repeat(np.arange(10).reshape([1, -1]), 10, axis=0)
    image, masked_image, x_min, y_min, w_cut, h_cut = draw_forgetting_box(img, size_ratio=(0.3, 0.3), mask_value=-1)
    np.set_printoptions(threshold=np.inf)
    print(masked_image)
    print(x_min, y_min, "\n", w_cut, h_cut)
