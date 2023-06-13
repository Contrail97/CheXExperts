import torch
import numpy as np
import torch.nn.functional as F


class BCE(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, pred, gt, **kwargs):
        target = gt.cuda()
        output = torch.cat(pred, 1)
        loss = F.binary_cross_entropy_with_logits(output, target)
        return loss, np.zeros(self.cfg.dataset.num_classes)


class PCAM_BCE(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, pred, gt, **kwargs):
        loss_t = torch.tensor(0., requires_grad=True).cuda()
        loss_idv = np.zeros(self.cfg.dataset.num_classes)
        for index in range(self.cfg.dataset.num_classes):
            target = gt[:, index].view(-1)
            output = pred[index].view(-1)

            # BCE Loss
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).cuda()
                # loss = F.binary_cross_entropy_with_logits(output, target)
            else:
                weight = (target.size()[0] - target.sum()) / target.sum()
                loss = F.binary_cross_entropy_with_logits(
                    output, target, pos_weight=weight)
            loss_t += loss
            loss_idv[index] += loss.item()
        return loss_t, loss_idv


class FocalLoss(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, pred, gt, **kwargs):
        loss_t = torch.tensor(0., requires_grad=True).cuda()
        loss_idv = np.zeros(self.cfg.dataset.num_classes)
        for index in range(self.cfg.dataset.num_classes):
            target = gt[:, index].view(-1)
            output = torch.sigmoid(pred[index].view(-1))

            # Focal Loss
            eps = 1e-7
            loss_1 = -1 * self.cfg.train.focal_alpha * torch.pow((1 - output), self.cfg.train.focal_gamma) * torch.log \
                (output + eps) * target
            loss_0 = -1 * (1 - self.cfg.train.focal_alpha) * torch.pow(output, self.cfg.train.focal_gamma) * torch.log \
                (1 - output + eps) * (
                    1 - target)
            loss_t += torch.mean(loss_0 + loss_1)
            loss_idv[index] += loss_t.item()
        return loss_t, loss_idv


class SoftConvergenceLoss(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, heatmaps, seg, gt, **kwargs):
        loss_t = torch.tensor(0., requires_grad=True).cuda()
        loss_idv = np.zeros(self.cfg.dataset.num_classes)

        if seg is not None:
            for index, hmap in enumerate(heatmaps):
                target = gt[:, index].view(-1)
                if target.sum() != 0:
                    # normalize heatmap
                    prob_hmap = torch.sigmoid(hmap)
                    weight_map = prob_hmap / prob_hmap.sum(dim=[2, 3], keepdim=True)

                    areaHeatmap = torch.sum(weight_map, dim=[2, 3])
                    areaUnion = torch.sum(torch.mul(weight_map, seg), dim=[2, 3])
                    loss_t += torch.sum(areaHeatmap) - torch.sum(areaUnion)
                    loss_idv[index] += loss_t.item()

        return loss_t, loss_idv


class Losses(object):

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.lossDict = {}
        self.lossSequence = []

        # register class
        self.lossDict["BCE"] = BCE
        self.lossDict["PCAM_BCE"] = PCAM_BCE
        self.lossDict["FocalLoss"] = FocalLoss
        self.lossDict["SoftConvergenceLoss"] = SoftConvergenceLoss

    def compile(self):
        if not isinstance(self.cfg.train.criterion, list):
            self.compile_single(self.cfg.train.criterion)
        else:
            for loss in self.cfg.train.criterion:
                self.compile_single(loss)

        return self.lossSequence

    def compile_single(self, loss: str):
        if loss in self.lossDict:
            self.lossSequence.append(self.lossDict[loss](self.cfg))
        else:
            raise ValueError(f"loss function {loss} is unknown")

        return self.lossSequence

    def __call__(self, **kwargs):
        loss_sum_list = None
        loss_idv_list = []

        for loss in self.lossSequence:
            loss_sum, loss_idv = loss(**kwargs)

            loss_idv_list.append(loss_idv)
            if loss_sum_list is None:
                loss_sum_list = loss_sum.unsqueeze(0)
            else:
                loss_sum_list = torch.cat([loss_sum_list, loss_sum.unsqueeze(0)])

        return loss_sum_list, loss_idv_list
