import torch
from torch import nn
import pytorch_lightning as pl

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

def test():
    model = Yolov1()
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

test()