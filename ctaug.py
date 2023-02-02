import math
import torch
import torch.nn as nn
from torch.distributions import categorical
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from typing import Optional, List, Tuple, Dict

from torch_randaug import _apply_op


def apply_op(img: Tensor, op_meta: Dict, interpolation: InterpolationMode, fill: Optional[List[float]]) -> Tensor:
    """
    Different behavior with RA - e.g., blending after some operations
    """
    op_name, params = [*op_meta.items()][0]
    magnitude = params[0].item()
    if op_name == "AutoContrast":
        img_t = F.autocontrast(img.clone())
        img = img_t * magnitude + img * (1 - magnitude)  # blend
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, magnitude)
    elif op_name == "Cutout":
        img = cutout_ctaug(img, magnitude)
    elif op_name == "Equalize":
        img_t = F.equalize(img.clone())
        img = img_t * magnitude + img * (1 - magnitude)  # blend
    elif op_name == "Invert":
        img_t = F.invert(img.clone())
        img = img_t * magnitude + img * (1 - magnitude)  # blend
    elif op_name == "Identity":
        pass
    elif op_name == "Posterize":
        img = F.posterize(img, round(magnitude))
    elif op_name == "Rescale":
        rescale_method = params[1]
        c, h, w = img.shape
        crop_width = round(magnitude * img.shape[1])
        img = F.center_crop(img, crop_width)
        img = F.resize(F.to_pil_image(img), (h, w), rescale_method)
        img = F.to_tensor(img)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, magnitude)
    elif op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "Smooth":
        smooth_kernel = torch.ones((3, 1, 3, 3), dtype=torch.uint8)
        smooth_kernel[:, :, 1, 1] = 5
        with torch.no_grad():  # img size [3, h, w]
            img_t = nn.functional.conv2d(img.clone().unsqueeze(0), smooth_kernel, padding=1, groups=3).squeeze(0)
        img = img_t * magnitude + img * (1 - magnitude)  # blend
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[round(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, round(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


class CTAugment(nn.Module):
    def __init__(self, depth=2, th=0.85, decay=0.99, num_bins=17, img_size=(96, 96)):
        """
        depth: number of selected transformations for each input image
        """
        super().__init__()
        # TODO: cutout, invert, rescale, smooth
        self.depth = depth
        self.th = th
        self.decay = decay
        self.num_bins = num_bins
        self.img_size = img_size

        self.op_meta = augmentation_space(num_bins)
        self.num_augs = len(self.op_meta)
        self.bins = torch.ones((self.num_augs, num_bins))  # bins for augmentation parameters
        self.rescale_options = [InterpolationMode.LANCZOS, InterpolationMode.BICUBIC, InterpolationMode.BILINEAR,
                                InterpolationMode.BOX, InterpolationMode.HAMMING, InterpolationMode.NEAREST]
        self.rescale_bins = torch.ones(6)

        self.aug_index = []
        self.bin_index = []
        self.rescale_index = []
    
    def update(self, preds, labels):
        with torch.no_grad():
            # compute update weight and apply to bins
            w = (1 -  1 / (2 * len(labels)) * (preds - labels).abs().sum()).cpu()
            for ai, bi in zip(self.aug_index, self.bin_index):
                self.bins[ai][bi] = self.decay * self.bins[ai][bi] + (1 - self.decay) * w
            # clear index
            self.aug_index = []
            self.bin_index = []
            # update if rescaling was chosen
            if len(self.rescale_index) > 0:
                for ri in self.rescale_index:
                    self.rescale_bins[ri] = self.decay * self.rescale_bins[ri] + (1 - self.decay) * w
                self.rescale_index = []
    
    def forward(self, img):
        augs = self.sample()
        for op_meta in augs:
            image = apply_op(img, op_meta, InterpolationMode.NEAREST, fill=None)
        return image

    def sample(self):
        # sample augmentations from categorical distribution according to bins
        augs = []
        aug_index = torch.randint(low=0, high=self.num_augs, size=(self.depth,))  # sample uniformly at random
        for ai in aug_index:
            aug = {}
            # get selected augentation name and sample its strength
            key = list(self.op_meta.keys())[ai]
            probs = self.bins[ai].clone()
            probs[probs<=self.th] = 1e-6
            probs = probs / probs.sum()
            try:
                bin_sampler = categorical.Categorical(probs)
            except ValueError as e:
                print(probs)
                print(e)
            bin_index = bin_sampler.sample()  # index of bin
            # register index
            self.aug_index.append(ai)
            self.bin_index.append(bin_index)
            aug[key] = [self.op_meta[key][bin_index]]
            if key == "Rescale":
                rescale_index = torch.randint(low=0, high=len(self.rescale_options), size=(1,))
                self.rescale_index.append(rescale_index)
                aug[key].append(self.rescale_options[rescale_index])
            augs.append(aug)
        return augs


def augmentation_space(num_bins: int) -> Dict[str, Tensor]:
    return {
        # op_name: (magnitudes, signed)
        "AutoContrast": torch.linspace(0., 1., num_bins),
        "Brightness": torch.linspace(0., 1., num_bins),
        "Color": torch.linspace(0., 1., num_bins),
        "Contrast": torch.linspace(0., 1., num_bins),
        "Cutout": torch.linspace(0, 0.5, num_bins),
        "Equalize": torch.linspace(0., 1., num_bins),
        "Invert": torch.linspace(0., 1., num_bins),
        "Identity": torch.linspace(0., 1., num_bins),
        "Posterize": torch.linspace(0., 8., num_bins),
        "Rescale": torch.linspace(0.5, 1., num_bins),
        "Rotate": torch.linspace(-45.0, 45.0, num_bins),
        "Sharpness": torch.linspace(0., 1., num_bins),
        "ShearX": torch.linspace(-0.3, 0.3, num_bins),
        "ShearY": torch.linspace(-0.3, 0.3, num_bins),
        "Smooth": torch.linspace(0., 1., num_bins),
        "Solarize": torch.linspace(0., 1., num_bins),
        "TranslateX": torch.linspace(-0.3, 0.3, num_bins),
        "TranslateY": torch.linspace(-0.3, 0.3, num_bins),
    }


def cutout_ctaug(img, magnitude):
    """
    Sets a random square patch of side-length (L×image width) pixels to gray.
    """
    c, h, w = img.size()
    # patch size and location
    patch_size = round(float(w * magnitude))
    x = torch.randint(low=0, high=w - patch_size, size=(1,))
    y = torch.randint(low=0, high=h - patch_size, size=(1,))
    # gray value
    if img.dtype == torch.float32:
        if img.min() >= 0:  # 0 ~ 1
            value = 0.5
        else:  # -1 ~ 1
            value = 0.0
    elif img.dtype == torch.uint8:  # 0 ~ 255
        value = 127
    else:
        raise Exception("Not supported tensor dtype.")
    # cutout
    img[:, y:y+patch_size, x:x+patch_size] = value

    return img
