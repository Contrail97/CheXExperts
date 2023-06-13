import torch
from torch import nn
import torch.nn.functional as F
from model.GlobalPool import GlobalPool
from model.AttentionMap import AttentionMap
from model.backbone.DensenetModels import DenseNet121
from model.backbone.DensenetModels import DenseNet169
from model.backbone.DensenetModels import DenseNet201
from model.backbone.DenseUNet import DenseUNet, DenseUNetP, DenseUNetPP

BACKBONES = {'densenet121': DenseNet121,
             'densenet169': DenseNet169,
             'densenet201': DenseNet201}


class ChexClassifier(nn.Module):

    def __init__(self, cfg):
        super(ChexClassifier, self).__init__()
        self.cfg = cfg


        self._init_backbone()
        self._init_bn()
        self._init_conv()
        self._init_classifier()
        self._init_attention_map()

        self.global_pool = GlobalPool(cfg, self.backbone.channel_mapping[self.cfg.model.heatmap_size])

    def _init_backbone(self):
        if self.cfg.model.backbone == 'DenseNet121':
            model = DenseNet121(self.cfg)
        elif self.cfg.model.backbone == 'DenseNet169':
            model = DenseNet169(self.cfg)
        elif self.cfg.model.backbone == 'DenseNet201':
            model = DenseNet201(self.cfg)
        else:
            raise Exception('Unknown backbone : {}'.format(self.cfg.model.backbone))

        self.backbone_num_features = model.num_features
        
        if self.cfg.model.backend == 'Unet':
            self.backbone = DenseUNet(model, self.cfg)
        elif self.cfg.model.backend == 'Unet+':
            self.backbone = DenseUNetP(model, self.cfg)
        elif self.cfg.model.backend == 'Unet++':
            self.backbone = DenseUNetPP(model, self.cfg)
        else:
            raise Exception('Unknown backend : {}'.format(self.cfg.model.backbone))

    def _init_classifier(self):
        for index in range(self.cfg.dataset.num_classes):
            setattr(self, "fc_" + str(index),
                nn.Conv2d(
                    self.backbone.channel_mapping[self.cfg.model.heatmap_size],
                    1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))
            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_conv(self):
        self.conv = nn.Conv2d(
            self.backbone.channel_mapping[self.cfg.model.heatmap_size] * 2,
            self.backbone.channel_mapping[self.cfg.model.heatmap_size],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def _init_bn(self):
        for index in range(self.cfg.dataset.num_classes):
            setattr(self, "bn_" + str(index),
                nn.BatchNorm2d(
                    self.backbone.channel_mapping[self.cfg.model.heatmap_size]))

    def _init_attention_map(self):
            setattr(self, "attention_map",
            AttentionMap(
                self.cfg,
                self.backbone_num_features))

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, img, seg=None):
        # (N, C, H, W)
        feat_map = self.backbone(img)
        if self.cfg.model.attention_map != "None":
            feat_map = self.attention_map(feat_map)

        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()

        for index in range(self.cfg.dataset.num_classes):

            classifier = getattr(self, "fc_" + str(index))

            # (N, 1, H, W)
            logit_map = classifier(feat_map)

            if self.cfg.model.heatmap_size >= 32:
                if seg is not None:
                    logit_map = logit_map * seg

                feat_map = self.conv(torch.cat((feat_map, feat_map * torch.sigmoid(logit_map)), 1))

            # (N, C, 1, 1)
            feat = self.global_pool(feat_map)

            if self.cfg.model.fc_bn:
                bn = getattr(self, "bn_" + str(index))
                feat = bn(feat)
                feat = F.dropout(feat, p=self.cfg.model.fc_drop, training=self.training)

            # (N, num_class, 1, 1)
            logit = classifier(feat)
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)

            logits.append(logit)
            logit_maps.append(logit_map)

        return (logits, logit_maps)
