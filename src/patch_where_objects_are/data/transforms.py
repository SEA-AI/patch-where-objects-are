"""Detection image augmentations.

Each pipeline takes one PIL image and returns a normalized ``(C, H, W)``
float tensor plus the parameters needed to transform its boxes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomHorizontalFlip:
    """Flip with probability ``p``; returns the image and whether it flipped."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> tuple[Image.Image, bool]:
        """Return the (possibly) flipped image and whether it was flipped."""
        if random.random() < self.p:
            return TF.hflip(img), True
        return img, False


class ColorJitter:
    """Brightness / contrast / saturation / hue jitter, applied with probability ``p``.

    Same semantics as ``torchvision.transforms.ColorJitter`` for the ranges.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        p: float = 0.8,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Return the jittered image (``img`` unchanged if not applied)."""
        if random.random() >= self.p:
            return img
        b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)
        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        img = TF.adjust_saturation(img, s)
        return TF.adjust_hue(img, h)


class RandomMotionBlur:
    """Gaussian blur with a random odd kernel size, applied with probability ``p``."""

    def __init__(self, kernel_sizes: Sequence[int] = (3, 5, 7), p: float = 0.2) -> None:
        self.kernel_sizes = list(kernel_sizes)
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Return the blurred image (``img`` unchanged if not applied)."""
        if random.random() >= self.p:
            return img
        k = random.choice(self.kernel_sizes)
        return img.filter(ImageFilter.GaussianBlur(radius=k // 2))


class RandomWeather:
    """Fog (blend with white, intensity 0.05-0.25), applied with probability ``p``."""

    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Return the fogged image (``img`` unchanged if not applied)."""
        if random.random() >= self.p:
            return img
        intensity = random.uniform(0.05, 0.25)
        return Image.blend(
            img, Image.new("RGB", img.size, (255, 255, 255)), alpha=intensity
        )


def to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> ImageNet-normalized float32 ``(C, H, W)`` tensor."""
    return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)


class _Augment:
    """Flip, color jitter, motion blur, fog (in this order)."""

    def __init__(self) -> None:
        self.hflip = RandomHorizontalFlip(p=0.5)
        self.jitter = ColorJitter()
        self.motion_blur = RandomMotionBlur()
        self.weather = RandomWeather()

    def __call__(self, img: Image.Image) -> tuple[Image.Image, bool]:
        img, flipped = self.hflip(img)
        img = self.jitter(img)
        img = self.motion_blur(img)
        return self.weather(img), flipped


class DetectionTransformPipeline:
    """Resize to ``img_size`` x ``img_size``, augment (train), normalize.

    Returns ``(tensor, hflipped)`` with ``tensor`` of shape ``(C, H, W)``. With
    ``augment=False`` (val / test) only resize and normalize are applied and
    ``hflipped`` is always False.
    """

    def __init__(self, img_size: int, augment: bool = True) -> None:
        self.img_size = img_size
        self.augment = _Augment() if augment else None

    def __call__(self, img: Image.Image) -> tuple[torch.Tensor, bool]:
        """Resize, augment and normalize ``img``; returns ``(tensor, hflipped)``."""
        img = TF.resize(
            img, [self.img_size, self.img_size], interpolation=Image.BILINEAR
        )
        flipped = False
        if self.augment is not None:
            img, flipped = self.augment(img)
        return to_normalized_tensor(img), flipped
