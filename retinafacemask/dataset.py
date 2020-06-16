import json
import random
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb
from torch.utils import data

from retinafacemask.data_augment import Preproc
from retinafacemask.utils import random_color


def l2_measure(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b) ** 2).sum())


def extract_mask_points(points: np.ndarray) -> np.ndarray:
    target_mask_polygon_points = np.zeros((16, 2), dtype=np.int32)

    target_mask_polygon_points[0] = points[28].astype(np.int32)
    target_mask_polygon_points[1:] = points[1:16].astype(np.int32)

    return target_mask_polygon_points


def extract_target_points_and_characteristic(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    avg_left_eye_point = points[36:42].mean(axis=0)
    avg_right_eye_point = points[42:48].mean(axis=0)
    avg_mouth_point = points[48:68].mean(axis=0)

    left_face_point = points[1]
    right_face_point = points[15]

    d1 = l2_measure(left_face_point, avg_mouth_point)
    d2 = l2_measure(right_face_point, avg_mouth_point)

    x1, y1 = avg_left_eye_point
    x2, y2 = avg_right_eye_point
    alpha = np.arctan((y2 - y1) / (x2 - x1 + 1e-5))

    s1 = alpha * 180 / np.pi
    s2 = d1 / (d2 + 1e-5)

    target_mask_polygon_points = extract_mask_points(points)

    return target_mask_polygon_points, s1, s2


class WiderFaceDetection(data.Dataset):
    def __init__(
        self,
        label_path: str,
        image_path: str,
        preproc: Optional[Preproc] = None,
        add_masks_prob: Optional[float] = None,
    ) -> None:
        self.add_mask_prob = add_masks_prob

        self.preproc = preproc
        self.image_path = Path(image_path)

        with open(label_path) as f:
            self.labels = json.load(f)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray]:
        labels = self.labels[index]
        file_name = labels["file_name"]
        image = load_rgb(self.image_path / file_name)

        annotations = np.zeros((0, 15))

        for label in labels["annotations"]:
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label["x_min"]
            annotation[0, 1] = label["y_min"]
            annotation[0, 2] = label["x_min"] + label["width"]
            annotation[0, 3] = label["y_min"] + label["height"]

            landmarks = label["landmarks"]
            # landmarks
            annotation[0, 4] = landmarks[0]  # l0_x
            annotation[0, 5] = landmarks[1]  # l0_y
            annotation[0, 6] = landmarks[3]  # l1_x
            annotation[0, 7] = landmarks[4]  # l1_y
            annotation[0, 8] = landmarks[6]  # l2_x
            annotation[0, 9] = landmarks[7]  # l2_y
            annotation[0, 10] = landmarks[9]  # l3_x
            annotation[0, 11] = landmarks[10]  # l3_y
            annotation[0, 12] = landmarks[12]  # l4_x
            annotation[0, 13] = landmarks[13]  # l4_y
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)

            if "dlib_landmarks" in label and self.add_mask_prob is not None and self.add_mask_prob < random.random():
                points = label["dlib_landmarks"]
                target_points, _, _ = extract_target_points_and_characteristic(np.array(points).astype(np.int32))
                image = cv2.fillPoly(image, [target_points], color=random_color())

        target = np.array(annotations)
        if self.preproc is not None:
            image, target = self.preproc(image, target)

        return torch.from_numpy(image).float(), target.astype(np.float32)


def detection_collate(batch: Tuple) -> Tuple[torch.Tensor, list]:
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    images = []
    for sample in batch:
        for tup in sample:
            if torch.is_tensor(tup):
                images.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annotations = torch.from_numpy(tup).float()
                targets.append(annotations)

    return torch.stack(images, 0), targets
