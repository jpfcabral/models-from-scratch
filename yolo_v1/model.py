import torch
from torch import nn
import pytorch_lightning as pl

from yolo_v1.utils import intersection_over_union

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(pl.LightningModule):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()

        self.in_channels = in_channels
        self.split_size = 7
        self.num_boxes = 2
        self.num_classes = 20

        self.darknet = nn.Sequential(
            # First block
            CNNBlock(self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second block
            CNNBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block
            CNNBlock(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fouth Block
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),

            CNNBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fifth Block
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            CNNBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),

            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),

            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.split_size * self.split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.split_size * self.split_size * (self.num_classes + self.num_boxes * 5)),
        )
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x, start_dim=1))

    def loss(self, logits, target):
        logits = logits.reshape(-1, self.split_size, self.split_size, self.num_classes + self.num_boxes)

        iou_b1 = intersection_over_union(logits[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(logits[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        box_predictions = exists_box * ((bestbox * logits[..., 26:30]+ (1 - bestbox) * logits[..., 21:25]))

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),torch.flatten(box_targets, end_dim=-2),)

        pred_box = (bestbox * logits[..., 25:26] + (1 - bestbox) * logits[..., 20:21])

        object_loss = self.mse(torch.flatten(exists_box * pred_box), torch.flatten(exists_box * target[..., 20:21]),)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * logits[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * logits[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        class_loss = self.mse(
            torch.flatten(exists_box * logits[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss