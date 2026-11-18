"""Classic model: VGG-style CNN on log-mel spectrograms, trained from scratch (Piczak 2015; Simonyan & Zisserman 2015)."""
from torch import nn


def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class ClassicCNN(nn.Module):
    def __init__(self, n_classes=50, widths=(32, 64, 128, 256)):
        super().__init__()
        chans = (1,) + widths
        self.body = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans, widths)])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(widths[-1], n_classes))

    def forward(self, z):
        return self.head(self.body(z).mean((2, 3)))  # global average pooling
