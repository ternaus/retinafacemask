import json
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.utils.data as data

from retinafacemask.data_augment import Preproc


class WiderFaceDetection(data.Dataset):
    def __init__(self, label_path: str, image_path: str, preproc: Optional[Preproc] = None) -> None:
        self.preproc = preproc
        self.image_path = Path(image_path)

        with open(label_path) as f:
            self.labels = json.load(f)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray]:
        labels = self.labels[index]
        file_name = labels["file_name"]
        image = cv2.imread(str(self.image_path / file_name))

        # height, width = image.shape[:2]

        # labels = self.words[index]
        annotations = np.zeros((0, 15))
        # if len(labels["annotations"]) == 0:
        #     return annotations

        for idx, label in enumerate(labels["annotations"]):
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
