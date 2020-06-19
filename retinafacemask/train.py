import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

import apex
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from iglovikov_helper_functions.metrics.map import recall_precision
from pytorch_lightning.logging import NeptuneLogger
from torch.utils.data import DataLoader
from torchvision.ops import nms

from retinafacemask.box_utils import decode
from retinafacemask.data_augment import Preproc
from retinafacemask.dataset import WiderFaceDetection, detection_collate
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class RetinaFaceMask(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = object_from_dict(self.hparams["model"])

        if hparams["sync_bn"]:
            self.model = apex.parallel.convert_syncbn_model(self.model)

        self.loss_weights = self.hparams["loss_weights"]

        self.priors = object_from_dict(self.hparams["prior_box"], image_size=[840, 840])
        self.loss = object_from_dict(self.hparams["loss"], priors=self.priors)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # skipcq: PYL-W0221
        return self.model(batch)

    def train_dataloader(self):
        return DataLoader(
            WiderFaceDetection(
                label_path=self.hparams["train_annotation_path"],
                image_path=self.hparams["train_image_path"],
                preproc=Preproc(self.hparams["image_size"][0], self.hparams["rgb_mean"]),
                add_masks_prob=self.hparams["add_masks_prob"],
            ),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=detection_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            WiderFaceDetection(
                label_path=self.hparams["val_annotation_path"],
                image_path=self.hparams["val_image_path"],
                preproc=Preproc(self.hparams["image_size"][0], self.hparams["rgb_mean"]),
                add_masks_prob=None,
            ),
            batch_size=self.hparams["val_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=detection_collate,
        )

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"], params=[x for x in self.model.parameters() if x.requires_grad]
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]  # skipcq: PYL-W0201

        return self.optimizers, [scheduler]

    # skipcq: PYL-W0613, PYL-W0221
    def training_step(self, batch, batch_idx):
        images = batch["image"]
        targets = batch["annotation"]

        out = self.forward(images)

        loss_localization, loss_classification, loss_landmarks = self.loss(out, targets)

        total_loss = (
            self.loss_weights["localization"] * loss_localization
            + self.loss_weights["classification"] * loss_classification
            + self.loss_weights["landmarks"] * loss_landmarks
        )

        logs = {
            "classification": loss_classification,
            "localization": loss_localization,
            "landmarks": loss_landmarks,
            "train_loss": total_loss,
            "lr": self._get_current_lr(),
        }

        return {"loss": total_loss, "log": logs}

    # skipcq: PYL-W0613, PYL-W0221
    def validation_step(self, batch, batch_idx):
        images = batch["image"]

        image_height = images.shape[2] * 2
        image_width = images.shape[3] * 2

        annotations = batch["annotation"]

        file_names = batch["file_name"]

        out = self.forward(images)

        loc, confidence, _ = out

        confidence = F.softmax(confidence, dim=-1)
        batch_size = loc.shape[0]

        predictions_coco = []

        scale = torch.Tensor([image_width, image_height, image_width, image_height]).to(loc.device)

        for batch_id in range(batch_size):
            boxes = decode(
                loc.data[batch_id], self.priors.to(images.device), self.hparams["test_parameters"]["variance"]
            )
            scores = confidence[batch_id][:, 1]

            valid_index = torch.where(scores > 0.1)[0]
            boxes = boxes[valid_index]
            scores = scores[valid_index]

            boxes *= scale

            # do NMS
            keep = nms(boxes, scores, self.hparams["val_parameters"]["iou_threshold"])
            boxes = boxes[keep, :].cpu().numpy()

            if boxes.shape[0] == 0:
                continue

            scores = scores[keep].cpu().numpy()

            file_name = file_names[batch_id]

            for box_id, bbox in enumerate(boxes):
                x_min, y_min, x_max, y_max = bbox

                x_min = np.clip(x_min, 0, x_max - 1)
                y_min = np.clip(y_min, 0, y_max - 1)

                predictions_coco += [
                    {
                        "id": str(hash(f"{file_name}_{box_id}")),
                        "image_id": file_name,
                        "category_id": 1,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": scores[box_id],
                    }
                ]

        gt_coco = []

        for batch_id, annotation_list in enumerate(annotations):
            for annotation in annotation_list:
                x_min, y_min, x_max, y_max = annotation[:4]
                file_name = file_names[batch_id]

                gt_coco += [
                    {
                        "id": str(hash(f"{file_name}_{batch_id}")),
                        "image_id": file_name,
                        "category_id": 1,
                        "bbox": [
                            x_min.item() * image_width,
                            y_min.item() * image_height,
                            (x_max - x_min).item() * image_width,
                            (y_max - y_min).item() * image_height,
                        ],
                    }
                ]

        return {"predictions": predictions_coco, "gt": gt_coco}

    def validation_epoch_end(self, outputs: List) -> Dict[str, Any]:
        result_predictions: List[dict] = []
        result_gt: List[dict] = []

        for output in outputs:
            result_predictions += output["predictions"]
            result_gt += output["gt"]

        _, _, average_precision = recall_precision(result_gt, result_predictions, 0.5)

        logs = {"epoch": self.trainer.current_epoch, "mAP@0.5": average_precision}

        return {"val_loss": average_precision, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    neptune_logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name=hparams["project_name"],
        experiment_name=f"{hparams['experiment_name']}",  # Optional,
        tags=["pytorch-lightning", "mlp"],  # Optional,
        upload_source_files=[],
    )

    pipeline = RetinaFaceMask(hparams)

    Path(hparams["checkpoint_callback"]["filepath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"],
        logger=neptune_logger,
        checkpoint_callback=object_from_dict(hparams["checkpoint_callback"]),
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
