"""
from https://github.com/ternaus/Pytorch_Retinaface/blob/master/models/retinaface.py
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from retinafacemask.net import FPN, SSH
from torch import nn
from torchvision.models import _utils
from typing import Dict, Tuple


class ClassHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(in_channels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_anchors: int = 3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool,
        in_channels: int,
        return_layers: Dict[str, int],
        out_channels: int,
        phase: str = "train",
    ) -> None:
        """

        Args:
            phase: train or test.
        """
        super().__init__()
        self.phase = phase
        # backbone = None
        # if name == "mobilenet0.25":
        #     backbone = MobileNetV1()
        #     if cfg["pretrain"]:
        #         checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device("cpu"))
        #         from collections import OrderedDict
        #
        #         new_state_dict = OrderedDict()
        #         for k, v in checkpoint["state_dict"].items():
        #             name = k[7:]  # remove module.
        #             new_state_dict[name] = v
        #         # load params
        #         backbone.load_state_dict(new_state_dict)
        # elif name == "Resnet50":

        if name == "Resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError(f"Only Resnet50 backbone is supported but got {name}")

        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channels_stage2 = in_channels
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = out_channels
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, in_channels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, in_channels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, in_channels=out_channels)

    @staticmethod
    def _make_class_head(fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(in_channels, anchor_num))
        return classhead

    @staticmethod
    def _make_bbox_head(fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(in_channels, anchor_num))
        return bboxhead

    @staticmethod
    def _make_landmark_head(fpn_num: int = 3, in_channels: int = 64, anchor_num: int = 2) -> nn.ModuleList:
        landmarkhead = nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(LandmarkHead(in_channels, anchor_num))
        return landmarkhead

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == "train":
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
