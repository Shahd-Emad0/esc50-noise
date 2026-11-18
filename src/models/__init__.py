"""Model registry: every spectrogram model = waveform -> LogMel -> (SpecAugment in training) -> network."""
import torch
import torchaudio
from torch import nn

from ..datasets.esc50 import LogMel
from .cnn import ClassicCNN
from .crnn_attention import CRNNAttention
from .efficientnet import efficientnet_b0


class SpecClassifier(nn.Module):
    def __init__(self, net, specaug=True):
        super().__init__()
        self.features, self.net = LogMel(), net
        self.specaug = nn.Sequential(torchaudio.transforms.FrequencyMasking(24, iid_masks=True),
                                     torchaudio.transforms.TimeMasking(48, iid_masks=True)) if specaug else nn.Identity()

    def forward(self, wave):
        z = self.features(wave)
        return self.net(self.specaug(z) if self.training else z)


def build_model(cfg):
    nets = {"cnn": lambda: ClassicCNN(),
            "crnn_attention": lambda: CRNNAttention(attention=cfg["attention"]),
            "efficientnet_b0": lambda: efficientnet_b0(cfg["pretrained"])}
    return SpecClassifier(nets[cfg["model"]](), cfg["specaug"])


def load_trained(cfg, path, device):
    cfg = {**cfg, "pretrained": False}  # weights come from the checkpoint
    m = build_model(cfg)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device).eval()
