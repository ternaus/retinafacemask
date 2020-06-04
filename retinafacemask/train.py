import argparse
from pathlib import Path

import apex
import pytorch_lightning as pl
import torch
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from torch.utils.data import DataLoader
from retinafacemask.data_augment import preproc
from retinafacemask.dataset import WiderFaceDetection, detection_collate
import yaml
import os


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

        self.loss = object_from_dict(self.hparams["loss"])
        self.loss_weights = self.hparams["loss_weights"]

        priorbox = object_from_dict(self.hparams["prior_box"])
        with torch.no_grad():
            priors = priorbox.forward()
            self.priors = priors.cuda()



    def forward(self, batch) -> torch.Tensor:  # skipcq: PYL-W0221
        return self.model(batch)

    def train_dataloader(self):
        return DataLoader(
            WiderFaceDetection(self.hparams["train_annotation_path"],
                               preproc(self.hparams["prior_box"]["image_size"], self.hparams["rgb_mean"])),
            batch_size=self.hparams["val_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=detection_collate
        )

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"], params=filter(lambda x: x.requires_grad, self.model.parameters()),
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]  # skipcq: PYL-W0201

        return self.optimizers, [scheduler]

    # skipcq: PYL-W0613, PYL-W0221
    def training_step(self, batch, batch_idx):
        images, targets = next(batch)

        out = self.forward(images)

        loss_localization, loss_classification, loss_landmarks = self.loss(out, self.priors, targets)

        total_loss = self.loss_weights["localization"] * loss_localization + \
                     self.loss_weights["classification"] * loss_classification + \
                     self.loss_weights["landmarks"] * loss_landmarks

        logs = {
            "classification": loss_classification,
            "localization": loss_localization,
            "landmarks": loss_landmarks,
            "train_loss": total_loss,
            "lr": self._get_current_lr()
        }

        return {"loss": total_loss, "log": logs}

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()




def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    # neptune_logger = NeptuneLogger(
    #     api_key=os.environ["NEPTUNE_API_TOKEN"],
    #     project_name=hparams["project_name"],
    #     experiment_name=f"{hparams['experiment_name']}",  # Optional,
    #     tags=["pytorch-lightning", "mlp"],  # Optional,
    #     upload_source_files=[],
    # )

    pipeline = RetinaFaceMask(hparams)

    Path(hparams["checkpoint_callback"]["filepath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"],
        # logger=neptune_logger,
        checkpoint_callback=object_from_dict(hparams["checkpoint_callback"]),
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
