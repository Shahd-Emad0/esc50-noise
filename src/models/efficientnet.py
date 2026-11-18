"""Modern pretrained CNN: EfficientNet-B0 (Tan & Le 2019) via timm, ImageNet weights or random init.
timm sums the RGB stem filters into one input channel for the 1-channel log-mel input."""
import timm


def efficientnet_b0(pretrained, n_classes=50):
    return timm.create_model("efficientnet_b0", pretrained=pretrained, in_chans=1, num_classes=n_classes, drop_rate=0.3)
