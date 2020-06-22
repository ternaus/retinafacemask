import json
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import albumentations as albu
import cv2
import numpy as np
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
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
        self, label_path: str, image_path: str, image_size: int, add_masks_prob: Optional[float] = None
    ) -> None:
        self.add_mask_prob = add_masks_prob

        self.preproc = Preproc(img_dim=image_size, rgb_means=[0.485, 0.456, 0.406])
        self.image_size = image_size
        self.image_path = Path(image_path)

        with open(label_path) as f:
            self.labels = json.load(f)

        self.valid_annotation_indices = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        labels = self.labels[index]
        file_name = labels["file_name"]
        image = load_rgb(self.image_path / file_name)

        # annotations will have the format
        # 4: box, 10 landmarks, 1: landmarks / no landmarks, 1: mask / no_mask

        annotations = np.zeros((0, 16))

        for label in labels["annotations"]:
            annotation = np.zeros((1, 16))
            # bbox
            annotation[0, 0] = label["x_min"]
            annotation[0, 1] = label["y_min"]
            annotation[0, 2] = label["x_min"] + label["width"]
            annotation[0, 3] = label["y_min"] + label["height"]

            if label["landmarks"]:
                landmarks = np.array(label["landmarks"])
                # landmarks
                annotation[0, 4:14] = landmarks[self.valid_annotation_indices]
                if annotation[0, 4] < 0:
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1

            if "dlib_landmarks" in label and self.add_mask_prob is not None and random.random() < self.add_mask_prob:
                points = label["dlib_landmarks"]
                target_points, _, _ = extract_target_points_and_characteristic(np.array(points).astype(np.int32))
                image = cv2.fillPoly(image, [target_points], color=random_color())
                annotation[0, 15] = 1
            else:
                annotation[0, 15] = 0

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)

        image, target = self.preproc(image, target)

        image = albu.Compose(
            [
                albu.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=(0.5, 1.5), p=0.5),
                albu.HueSaturationValue(hue_shift_limit=18, val_shift_limit=0, p=0.5),
                albu.Resize(height=self.image_size, width=self.image_size, p=1),
                albu.Normalize(p=1),
            ]
        )(image=image)["image"]

        return {
            "image": tensor_from_rgb_image(image),
            "annotation": target.astype(np.float32),
            "file_name": file_name,
        }


def detection_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    annotation = []
    images = []
    file_names = []

    for sample in batch:
        images.append(sample["image"])
        annotation.append(torch.from_numpy(sample["annotation"]).float())
        file_names.append(sample["file_name"])

    return {"image": torch.stack(images), "annotation": annotation, "file_name": file_names}
