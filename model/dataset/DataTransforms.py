import math
import torch
import numpy as np
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class ToCUDA(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        item["img"] = item["img"].cuda() if item["img"] else None
        item["seg"] = item["seg"].cuda() if item["seg"] else None
        item["lab"] = item["lab"].cuda() if item["lab"] else None

        return item

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(torch.nn.Module):

    def __init__(self, size, **kwargs):
        super().__init__()
        self.size = size
        self.Resizer = transforms.Resize(size)

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        if "box" in item:
            scale = [item["img"].size[0]/self.size[0], item["img"].size[1]/self.size[1]]
            for key, value in item["box"].items():
                value[0] = int(value[0] / scale[0])
                value[1] = int(value[1] / scale[0])
                value[2] = int(value[2] / scale[1])
                value[3] = int(value[3] / scale[1])

        item["img"] = self.Resizer.forward(item["img"])
        item["seg"] = self.Resizer.forward(item["seg"]) if item["seg"] else None
        item["cop"] = np.array(item["img"])

        return item

    def __repr__(self):
        return str(self.Resizer)


class ROICrop(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):

        if "roi" in item and item["roi"]:
            x1, x2, y1, y2 = item["roi"]
            top, left, height, width = y1, x1, y2-y1, x2-x1

            item["img"] = F.crop(item["img"], top, left, height, width)
            item["seg"] = F.crop(item["seg"], top, left, height, width) if item["seg"] else None

        return item

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        item["img"] = F.to_tensor(item["img"])
        item["seg"] = F.to_tensor(item["seg"]) if item["seg"] else None

        return item

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(torch.nn.Module):

    def __init__(self, mean, std, inplace=False, **kwargs):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        item["img"] = F.normalize(item["img"], self.mean, self.std, self.inplace)
        return item

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ColorJitter(torch.nn.Module):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, **kwargs):
        super().__init__()
        self.Jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        item["img"] = self.Jitter.forward(item["img"])
        return item

    def __repr__(self):
        str(self.Jitter)


class RandomAffine(torch.nn.Module):

    def __init__(self, degrees, translate=None, scale=None, fill=0, **kwargs):
        super().__init__()
        self.fill = fill
        self.Affine = transforms.RandomAffine(degrees = degrees, translate = translate, scale = scale, fill = fill)

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        fill = self.fill
        if isinstance(item["img"], Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(item["img"])
            else:
                fill = [float(f) for f in fill]

        img_size = F.get_image_size(item["img"])

        ret = self.Affine.get_params(self.Affine.degrees, self.Affine.translate, self.Affine.scale, self.Affine.shear, img_size)

        item["img"] = F.affine(item["img"], *ret, interpolation=self.Affine.interpolation, fill=fill)
        item["seg"] = F.affine(item["seg"], *ret, interpolation=self.Affine.interpolation, fill=fill) if item["seg"] else None
        return item

    def __repr__(self):
        str(self.Affine)


class PaddingResizer(object):

    def __init__(self, size, **kwargs):
        self._size = size

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        def resize(img):
            max_side = self._size
            rows, cols = img.height, img.width
            largest_side = max(rows, cols)
            scale = max_side / largest_side
            target_height = int(round(rows * scale))
            target_width = int(round(cols * scale))
            padding_height = max_side - target_height;
            padding_width = max_side - target_width;

            # resize the image with the computed scale
            new_img = transforms.Resize((target_height, target_width))(img)
            new_img = transforms.Pad((math.ceil(padding_width/2), math.ceil(padding_height/2),
                                      int(padding_width/2), int(padding_height/2)))(new_img)
            return new_img

        item["img"] = resize(item["img"])
        item["seg"] = resize(item["seg"]) if item["seg"] else None
        return item


class RandomResizedCrop(object):

    def __init__(self, size, scale=(0.08, 1.0), **kwargs):
        super().__init__()
        self.Resizer = transforms.RandomResizedCrop(size, scale)

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        i, j, h, w = self.Resizer.get_params(item["img"], self.Resizer.scale, self.Resizer.ratio)
        item["img"] = F.resized_crop(item["img"], i, j, h, w, self.Resizer.size, self.Resizer.interpolation)
        item["seg"] = F.resized_crop(item["seg"], i, j, h, w, self.Resizer.size, self.Resizer.interpolation) if item["seg"] else None
        return item

    def __repr__(self):
        return str(self.Resizer)


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, item, **kwargs):
        return self.forward(item)

    def forward(self, item):
        if torch.rand(1) < self.p:
            item["img"] = F.hflip(item["img"])
            item["seg"] = F.hflip(item["seg"]) if item["seg"] else None
            return item
        return item

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SegmentationPrep(object):

    def __init__(self, size, **kwargs):
        super().__init__()
        self.Resizer = transforms.Resize(size)

    def __call__(self, item):
        return self.forward(item)

    def forward(self, item):
        if "seg" in item and item["seg"] is not None:
            # To grayscale
            item["seg"] = F.rgb_to_grayscale(item["seg"])

            # Binarization
            item["seg"] = item["seg"]*10
            item["seg"] = torch.clamp(item["seg"], 0.5 , 1.0)

            # Resize to heatmap size
            item["seg"] = self.Resizer.forward(item["seg"])

            # import matplotlib.pyplot as plt
            # item["img"] = item["seg"] * self.Resizer.forward(item["img"])
            # seg = item["img"]
            # seg = seg.cpu()
            # plt.figure()
            # plt.imshow(seg.permute(1, 2, 0))
            # plt.waitforbuttonpress()

        return item

    def __repr__(self):
        return str(self.Resizer)


class DataTransforms(object):

    def __init__(self, cfg):
        self._cfg = cfg
        self._transDict = {}
        self._paramDict = {}

        # register class
        self._transDict["ToCUDA"] = ToCUDA
        self._transDict["Resizer"] = Resize
        self._transDict["ROICrop"] = ROICrop
        self._transDict["ToTensor"] = ToTensor
        self._transDict["Normalize"] = Normalize
        self._transDict["ColorJitter"] = ColorJitter
        self._transDict["RandomAffine"] = RandomAffine
        self._transDict["PaddingResizer"] = PaddingResizer
        self._transDict["SegmentationPrep"] = SegmentationPrep
        self._transDict["RandomResizedCrop"] = RandomResizedCrop
        self._transDict["RandomHorizontalFlip"] = RandomHorizontalFlip

    def _get_param(self, name):
        param = {}
        try:
            if name == "PaddingResizer":
                param["size"] = self._cfg.dataset.datatransforms.kwargs.imgtrans_size
            elif name == "Resizer":
                param["size"] = [self._cfg.dataset.datatransforms.kwargs.imgtrans_size,
                                 self._cfg.dataset.datatransforms.kwargs.imgtrans_size]
            elif name == "RandomResizedCrop":
                param["size"] = self._cfg.dataset.datatransforms.kwargs.imgtrans_size
                param["scale"] = self._cfg.dataset.datatransforms.kwargs.imgtrans_scale
            elif name == "Normalize":
                param["mean"] = self._cfg.dataset.datatransforms.kwargs.mormal_mean
                param["std"] = self._cfg.dataset.datatransforms.kwargs.mormal_std
            elif name == "RandomAffine":
                param["degrees"] = self._cfg.dataset.datatransforms.kwargs.affine_degrees
                param["translate"] = self._cfg.dataset.datatransforms.kwargs.affine_translate
                param["scale"] = self._cfg.dataset.datatransforms.kwargs.affine_scale
                param["fill"] = self._cfg.dataset.datatransforms.kwargs.affine_fill
            elif name == "ColorJitter":
                param["brightness"] = self._cfg.dataset.datatransforms.kwargs.jitter_brightness
                param["contrast"] = self._cfg.dataset.datatransforms.kwargs.jitter_contrast
                param["saturation"] = self._cfg.dataset.datatransforms.kwargs.jitter_saturation
                param["hue"] = self._cfg.dataset.datatransforms.kwargs.jitter_hue
            elif name == "SegmentationPrep":
                param["size"] = self._cfg.model.heatmap_size
        except AttributeError as e:
            print(e)
        return param

    def compile(self, mode=None):
        transformSequence = []

        if mode == None:
            mode = self._cfg.mode

        if mode == "train":
            transList = self._cfg.dataset.datatransforms.train
        else:
            transList = self._cfg.dataset.datatransforms.val

        for trans in transList:
            if trans in self._transDict:
                transClass = self._transDict[trans]
                transParam = self._get_param(trans)
                transObj = transClass(**transParam)
                transformSequence.append(transObj)
            else:
                raise ValueError(f"datatransforms {trans} is unknown")

        return transforms.Compose(transformSequence)
