import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):

    def __init__(self, pathImageDirectory, pathDatasetFile, transform):
        self.listImagePaths = []
        self.listChestROIs = []
        self.listImageLabels = []
        self.listSegmentPaths = []
        self.transform = transform

        # ---- Check if pathImageDirectory contains segmentDirectory
        if isinstance(pathImageDirectory, list) and len(pathImageDirectory) > 1:
            imageDirectory = pathImageDirectory[0]
            segmentDirectory = pathImageDirectory[1]
        else:
            imageDirectory = pathImageDirectory
            segmentDirectory = None

        # ---- Open file, get image paths and labels
        fileDescriptor = open(pathDatasetFile, "r")

        # ---- get into the loop
        line = True
        while line:
            line = fileDescriptor.readline()
            # --- if not empty
            if line:
                lineItems = line.split(',')
                imagePath = os.path.join(imageDirectory, lineItems[0])
                imageROI = [int(i) for i in lineItems[1:5]]
                imageLabel = [int(i) for i in lineItems[5:]]

                if segmentDirectory:
                    segmentPath = os.path.join(segmentDirectory, lineItems[0])
                    self.listSegmentPaths.append(segmentPath)

                self.listImagePaths.append(imagePath)
                self.listChestROIs.append(imageROI)
                self.listImageLabels.append(imageLabel)
        fileDescriptor.close()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        imageROI = self.listChestROIs[index]

        if index < len(self.listSegmentPaths):
            segmentPath = self.listSegmentPaths[index][:-4] + "_labels.png"
            segmentData = Image.open(segmentPath).convert('RGB')
        else:
            segmentData = None

        item = {
            "img": imageData,
            "seg": segmentData,
            "lab": imageLabel,
            "roi": imageROI
        }

        if self.transform:
            item = self.transform(item)

        if item["seg"] is None:
            item.pop("seg")

        # npimg = imageData.permute(1,2,0).cpu().detach().numpy()
        # showimg = Image.fromarray(np.uint8(npimg*255)).convert('RGB')
        # showimg.show()

        return item

    def __len__(self):
        return len(self.listImagePaths)


class BBoxDatasetGenerator(Dataset):

    def __init__(self, pathImageDirectory, pathDatasetFile, transform):
        self.dataMap = {}
        self.dataList = []
        self.transform = transform

        # ---- Check if pathImageDirectory contains segmentDirectory
        imageDirectory = pathImageDirectory

        # ---- Open file, get image paths and labels
        fileDescriptor = open(pathDatasetFile, "r")

        # ---- get into the loop
        line = True
        while line:
            line = fileDescriptor.readline()
            # --- if not empty
            if line:
                lineItems = line.split(',')
                imageIdx = lineItems[0]
                imagePath = os.path.join(imageDirectory, imageIdx)
                imageROI = [int(i) for i in lineItems[1:5]]
                imageBBox = [int(i) for i in lineItems[5:9]]
                imageLabel = int(lineItems[9])

                if imageIdx not in self.dataMap:
                    self.dataMap[imageIdx] = {
                        "pth": imagePath,
                        "roi": imageROI,
                        "box": {},
                    }
                self.dataMap[imageIdx]["box"][imageLabel] = imageBBox

        for key, value in self.dataMap.items():
            self.dataList.append(value)

        fileDescriptor.close()

    def __getitem__(self, index):
        data = self.dataList[index]
        imageData = Image.open(data["pth"]).convert('RGB')

        item = {
            "img": imageData,
            "seg": None,
            "roi": data["roi"],
            "box": data["box"],
            "pth": data["pth"],
        }

        if self.transform:
            item = self.transform(item)

        if item["seg"] is None:
            item.pop("seg")

        return item

    def __len__(self):
        return len(self.dataList)


def BBoxCollater(data):
    img = [s['img'] for s in data]
    roi = [s['roi'] for s in data]
    box = [s['box'] for s in data]
    pth = [s['pth'] for s in data]
    cop = [s['cop'] for s in data]

    widths = [int(s.shape[1]) for s in img]
    heights = [int(s.shape[2]) for s in img]
    batch_size = len(img)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()
    batch_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        batch_imgs[i, :, :, :] = img[i].permute(1, 2, 0)

    batch_imgs = batch_imgs.permute(0, 3, 1, 2)

    return {"img": batch_imgs, "roi": roi, "box": box, "cop": cop, "pth": pth}