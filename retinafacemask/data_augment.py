import random
from typing import Tuple

import albumentations as albu
import numpy as np

from retinafacemask.box_utils import matrix_iof


def _crop(image, boxes, labels, landm, img_dim):
    height, width = image.shape[:2]
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            unclear_variable = 0
        else:
            unclear_variable = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((unclear_variable, t, unclear_variable + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = value >= 1
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1] : roi[3], roi[0] : roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] = boxes_t[:, :2] - roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] = boxes_t[:, 2:] - roi[:2]

        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        landms_t = landms_t.reshape([-1, 10])

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return image, boxes, labels, landm, pad_image_flag


def _mirror(image: np.ndarray, boxes: np.ndarray, landms: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = image.shape[1]
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width = image.shape[:2]
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:height, 0:width] = image
    return image_t


class Preproc:
    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        if targets.shape[0] == 0:
            raise ValueError("this image does not have gt")

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landmarks = targets[:, 4:-1].copy()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landmarks, self.img_dim)
        image_t = _pad_to_square(image_t, self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        height, width = image_t.shape[:2]

        boxes_t[:, 0::2] = boxes_t[:, 0::2] / width
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / height

        landm_t[:, 0::2] = landm_t[:, 0::2] / width
        landm_t[:, 1::2] = landm_t[:, 1::2] / height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))

        image_t = albu.Compose(
            [
                albu.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=(0.5, 1.5), p=0.5),
                albu.HueSaturationValue(hue_shift_limit=18, val_shift_limit=0, p=0.5),
                albu.Resize(height=self.img_dim, width=self.img_dim, p=1),
                albu.Normalize(p=1),
            ]
        )(image=image_t)["image"]

        image_t = image_t.transpose(2, 0, 1)

        return image_t, targets_t
