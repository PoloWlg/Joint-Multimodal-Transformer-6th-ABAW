"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import numbers
from typing import Tuple, Optional, List, Any, Union

import torch
from torch import Tensor
from torchvision.transforms.functional import normalize
import numpy as np
import cv2
import PIL
import PIL.Image
import random
import importlib
from torchaudio.transforms import AmplitudeToDB
from .intensity import RandomColorAugment
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchaudio


class ComposeWithInvert(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, invert=False):
        if invert:
            for t in reversed(self.transforms):
                img = t(img, invert)
        else:
            for t in self.transforms:
                img = t(img, invert)
        return img


class ComposeAudioSpectro(object):

    def __init__(self, l_transforms: list):
        self.transforms = l_transforms

    def __call__(self, spectro):
        for t in self.transforms:
            spectro = t(spectro)

        return spectro


class NumpyToTensor:
    # convert numpy to tensor, or tensor to tensor
    def __init__(self):
        pass

    def __call__(self, clip, invert):

        if invert:
            # convert to TODO
            clip = clip.permute(1, 2, 3, 0)
            clip = clip.mul(255).to(torch.uint8)
        else:
            # convert from img int8 T, W, H, C to float 0-1 image C T W H
            clip = clip.astype(np.float32) / 255
            clip = torch.from_numpy(clip).permute(3, 0, 1, 2)

        return clip


class ColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of
    leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency
    (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter
        brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness),
             1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter
        contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast),
            1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter
        saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation),
             1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given
             [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be
            non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval
            with negative values,
            or use an interpolation that generates negative values before using
            this function.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be "
                                 f"non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple "
                            f"with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float],
               Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def sample_params(self) -> Tuple[Tensor, Optional[float], Optional[float],
                                     Optional[float], Optional[float]]:
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        return fn_idx, brightness_factor, contrast_factor, saturation_factor,\
               hue_factor

    def forward(self,
                img,
                fn_idx: torch.Tensor,
                brightness_factor: float,
                contrast_factor: float,
                saturation_factor: float,
                hue_factor: float
                ):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = TF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = TF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = TF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = TF.adjust_hue(img, hue_factor)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s


class RandomColorJitter(torch.nn.Module):
    """Randomly ColorJitter.
    The same as transforms.ColorJitter but it is applied randomly over
    samples. instead over all samples.
    p: probability of aplying ColorJitter.
    """

    def __init__(self,
                 brightness=0.,
                 contrast=0.,
                 saturation=0.,
                 hue=0.,
                 p=0.1):
        super().__init__()
        self.p = p
        self.colorjitter = ColorJitter(brightness=brightness,
                                       contrast=contrast,
                                       saturation=saturation,
                                       hue=hue
                                       )

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly color-jittered.
        """

        if self.p == 1 or torch.rand(1) < self.p:
            params = self.colorjitter.sample_params()
            fn_idx, brightness_factor, contrast_factor, saturation_factor, \
            hue_factor = params

            img_ = self.colorjitter(img, fn_idx, brightness_factor,
                                    contrast_factor, saturation_factor,
                                    hue_factor)


            return img_

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


def RandomColorAugmentation(clip):
    augmentation = RandomColorAugment()
    for clip_i in range(clip.shape[0]):
        img = clip[clip_i, :, :, 0:3]
        img = augmentation(img)
        clip[clip_i, :, :, 0:3] = np.array(img)
    return clip


def more_random_vision_augmentation(clip: np.ndarray,
                                    crop_size: int) -> np.ndarray:
    assert clip.ndim == 4, clip.ndim  # 8, h, w, 3

    aug = transforms.Compose([
        transforms.RandomRotation(degrees=[-6, 6]),
        transforms.RandomResizedCrop(size=crop_size, scale=(0.8, 1.0),
                          ratio=(1. / 1., 1. / 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        RandomColorJitter(brightness=0.4,
                          contrast=0.4,
                          saturation=0.4,
                          hue=0.1,
                          p=0.8),
    ])
    for clip_i in range(clip.shape[0]):
        img = clip[clip_i, :, :, 0:3]
        img = PIL.Image.fromarray(img)
        img = aug(img)
        clip[clip_i, :, :, 0:3] = np.array(img)
    return clip


class RandomTimeStretch(torch.nn.Module):
    """
    Audio data augmentation.
    """
    def __init__(self, n_freq: int = 201, p=0.5):
        super().__init__()

        self.p = p
        self.op = torchaudio.transforms.TimeStretch(n_freq=n_freq)

    def __call__(self, spectro):

        if self.p == 1 or torch.rand(1) < self.p:
            if torch.rand(1) < 0.5:
                return self.op(spectro, overriding_rate=1.2)
            else:
                return self.op(spectro, overriding_rate=0.9)

        return spectro


class RandomTimeMasking(torch.nn.Module):
    """
    Audio data augmentation.
    """
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p
        self.op = torchaudio.transforms.TimeMasking(time_mask_param=80)

    def __call__(self, spectro):

        if self.p == 1 or torch.rand(1) < self.p:
            return self.op(spectro)

        return spectro


class RandomFrequencyMasking(torch.nn.Module):
    """
    Audio data augmentation.
    """
    def __init__(self, p=0.5):
        super().__init__()

        self.p = p
        self.op = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)

    def __call__(self, spectro):

        if self.p == 1 or torch.rand(1) < self.p:
            return self.op(spectro)

        return spectro


def more_random_audio_spectrogram_augmentation(
        spectro: torch.Tensor) -> torch.Tensor:

    audio_transforms = ComposeAudioSpectro([
        RandomTimeMasking(p=0.6),
        RandomFrequencyMasking(p=0.6)
    ])

    spectro = audio_transforms(spectro)

    return spectro


class Normalize:
    """Normalize an tensor image or video clip with mean and standard deviation.
       Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        # forward is an in place operation!
        # invert is not
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.mean_t = None
        self.std_t = None

    def __call__(self, clip, invert: bool = False):
        if self.mean_t is None:
            dtype = clip.dtype
            if len(clip.shape) == 4:
                self.mean_t = torch.as_tensor(self.mean, dtype=dtype, device=clip.device)[:, None, None, None]
                self.std_t = torch.as_tensor(self.std, dtype=dtype, device=clip.device)[:, None, None, None]
            else:
                self.mean_t = torch.as_tensor(self.mean, dtype=dtype, device=clip.device)[:, None, None]
                self.std_t = torch.as_tensor(self.std, dtype=dtype, device=clip.device)[:, None, None]

        if invert:
            clip = clip.clone()
            clip.mul_(self.std_t).add_(self.mean_t)
        else:
            # image of size (C, H, W) to be normalized.
            # clip = normalize(clip, self.mean, self.std)
            clip.sub_(self.mean_t).div_(self.std_t)

        return clip


class AmpToDB:

    def __init__(self):
        self.amplitude_to_DB = AmplitudeToDB('power', 80)

    def __call__(self, features, invert: bool = False):

        if invert:
            pass  # do nothing
        else:
            features = self.amplitude_to_DB(features)

        return features


class RandomClipFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip, invert):

        if invert:
            pass  # do nothing
        else:
            if random.random() < self.p:
                # T W H C
                # assert clip.shape[3] == 3 # last channel is RGB
                # for every image apply cv2 flip
                for i in range(clip.shape[0]):
                    clip[i] = cv2.flip(clip[i], 1)

        return clip