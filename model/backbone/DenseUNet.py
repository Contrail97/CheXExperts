import torch
from torch import nn

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm :
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        use_batchnorm=True,
        attention_type="scse",
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels)
        self.conv2 = Conv2dReLU(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class DenseUNet(nn.Module):
    def __init__(self, densenet, cfg, deep_supervision=False, **kwargs):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.heatmap_size = cfg.model.heatmap_size
        self.input_size = cfg.dataset.datatransforms.kwargs.imgtrans_size

        self.channel_list = [2 ** x * densenet.num_init_features for x in range(5)]
        self.heatmap_size_list = [self.input_size/2**2, self.input_size/2**3, self.input_size/2**4, self.input_size/2**5]

        # heatmap size mapping to channel size
        self.channel_mapping = {
            self.heatmap_size_list[3]: self.channel_list[4],
            self.heatmap_size_list[2]: self.channel_list[2],
            self.heatmap_size_list[1]: self.channel_list[1],
            self.heatmap_size_list[0]: self.channel_list[0],
        }

        densenetStages = [
            nn.Identity(),
            nn.Sequential(densenet.features.conv0, densenet.features.norm0, densenet.features.relu0),
            nn.Sequential(
                densenet.features.pool0,
                densenet.features.denseblock1,
                densenet.features.transition1,
            ),
            nn.Sequential(densenet.features.denseblock2, densenet.features.transition2),
            nn.Sequential(densenet.features.denseblock3, densenet.features.transition3),
            nn.Sequential(densenet.features.denseblock4, densenet.features.norm5),
        ]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = densenetStages[1]
        self.conv1_0 = densenetStages[2]
        self.conv2_0 = densenetStages[3]
        self.conv3_0 = densenetStages[4]
        self.conv4_0 = densenetStages[5]

        self.conv3_1 = ConvBlock(self.channel_list[3]+self.channel_list[4], self.channel_list[3], self.channel_list[3])
        self.conv2_2 = ConvBlock(self.channel_list[2]+self.channel_list[3], self.channel_list[2], self.channel_list[2])
        self.conv1_3 = ConvBlock(self.channel_list[1]+self.channel_list[2], self.channel_list[1], self.channel_list[1])
        self.conv0_4 = ConvBlock(self.channel_list[0]+self.channel_list[1], self.channel_list[0], self.channel_list[0])


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1)) if self.heatmap_size >= self.heatmap_size_list[2] else None
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1)) if self.heatmap_size >= self.heatmap_size_list[1] else None
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(self.up(x1_3))], 1)) if self.heatmap_size >= self.heatmap_size_list[0] else None

        # heatmap size mapping to DenseUNet layer
        layer_mapping = {
            8: x4_0,
            16: x2_2,
            32: x1_3,
            128: x0_4,
        }

        return layer_mapping[self.heatmap_size]


class DenseUNetP(nn.Module):
    def __init__(self, densenet, cfg, deep_supervision=False, **kwargs):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.heatmap_size = cfg.model.heatmap_size
        self.input_size = cfg.dataset.datatransforms.kwargs.imgtrans_size

        self.channel_list = [2 ** x * densenet.num_init_features for x in range(5)]
        self.heatmap_size_list = [self.input_size/2**2, self.input_size/2**3, self.input_size/2**4, self.input_size/2**5]

        # heatmap size mapping to channel size
        self.channel_mapping = {
            self.heatmap_size_list[3]: self.channel_list[4],
            self.heatmap_size_list[2]: self.channel_list[2],
            self.heatmap_size_list[1]: self.channel_list[1],
            self.heatmap_size_list[0]: self.channel_list[0],
        }

        densenetStages = [
            nn.Identity(),
            nn.Sequential(densenet.features.conv0, densenet.features.norm0, densenet.features.relu0, densenet.features.pool0),
            nn.Sequential(densenet.features.denseblock1,densenet.features.transition1),
            nn.Sequential(densenet.features.denseblock2, densenet.features.transition2),
            nn.Sequential(densenet.features.denseblock3, densenet.features.transition3),
            nn.Sequential(densenet.features.denseblock4, densenet.features.norm5),
        ]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = densenetStages[1]
        self.conv1_0 = densenetStages[2]
        self.conv2_0 = densenetStages[3]
        self.conv3_0 = densenetStages[4]
        self.conv4_0 = densenetStages[5]

        self.conv0_1 = ConvBlock(self.channel_list[0]+self.channel_list[1], self.channel_list[0], self.channel_list[0])
        self.conv1_1 = ConvBlock(self.channel_list[1]+self.channel_list[2], self.channel_list[1], self.channel_list[1])
        self.conv2_1 = ConvBlock(self.channel_list[2]+self.channel_list[3], self.channel_list[2], self.channel_list[2])
        self.conv3_1 = ConvBlock(self.channel_list[3]+self.channel_list[4], self.channel_list[3], self.channel_list[3])

        self.conv0_2 = ConvBlock(self.channel_list[0]+self.channel_list[1], self.channel_list[1], self.channel_list[0])
        self.conv1_2 = ConvBlock(self.channel_list[1]+self.channel_list[2], self.channel_list[2], self.channel_list[1])
        self.conv2_2 = ConvBlock(self.channel_list[2]+self.channel_list[3], self.channel_list[3], self.channel_list[2])
        self.conv3_2 = ConvBlock(self.channel_list[3]+self.channel_list[4], self.channel_list[4], self.channel_list[3])


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))

        x3_2 = self.conv3_2(torch.cat([x3_1, x4_0], 1)) if self.heatmap_size >= self.heatmap_size_list[3] else None
        x2_2 = self.conv2_2(torch.cat([x2_1, self.up(x3_2)], 1)) if self.heatmap_size >= self.heatmap_size_list[2] else None
        x1_2 = self.conv1_2(torch.cat([x1_1, self.up(x2_2)], 1)) if self.heatmap_size >= self.heatmap_size_list[1] else None
        x0_2 = self.conv0_2(torch.cat([x0_1, self.up(x1_2)], 1)) if self.heatmap_size >= self.heatmap_size_list[0] else None

        # heatmap size mapping to DenseUNet layer
        layer_mapping = {
            8: x3_2,
            16: x2_2,
            32: x1_2,
            128: x0_2,
        }

        return layer_mapping[self.heatmap_size]



class DenseUNetPP(nn.Module):
    def __init__(self, densenet, cfg, **kwargs):
        super().__init__()

        self.heatmap_size = cfg.model.heatmap_size
        self.input_size = cfg.dataset.datatransforms.kwargs.imgtrans_size

        self.channel_list = densenet.list_num_features
        self.heatmap_size_list = [self.input_size/2**2, self.input_size/2**3, self.input_size/2**4, self.input_size/2**5]

        # heatmap size mapping to channel size
        self.channel_mapping = {
            self.heatmap_size_list[3]: self.channel_list[4],
            self.heatmap_size_list[2]: self.channel_list[2],
            self.heatmap_size_list[1]: self.channel_list[1],
            self.heatmap_size_list[0]: self.channel_list[0],
        }

        densenetStages = [
            nn.Identity(),
            nn.Sequential(densenet.features.conv0, densenet.features.norm0, densenet.features.relu0, densenet.features.pool0),
            nn.Sequential(densenet.features.denseblock1, densenet.features.transition1),
            nn.Sequential(densenet.features.denseblock2, densenet.features.transition2),
            nn.Sequential(densenet.features.denseblock3, densenet.features.transition3),
            nn.Sequential(densenet.features.denseblock4, densenet.features.norm5),
        ]

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = densenetStages[1]
        self.conv1_0 = densenetStages[2]
        self.conv2_0 = densenetStages[3]
        self.conv3_0 = densenetStages[4]
        self.conv4_0 = densenetStages[5]

        self.conv0_1 = ConvBlock(self.channel_list[0]+self.channel_list[1], self.channel_list[0], self.channel_list[0])
        self.conv1_1 = ConvBlock(self.channel_list[1]+self.channel_list[2], self.channel_list[1], self.channel_list[1])
        self.conv2_1 = ConvBlock(self.channel_list[2]+self.channel_list[3], self.channel_list[2], self.channel_list[2])
        self.conv3_1 = ConvBlock(self.channel_list[3]+self.channel_list[4], self.channel_list[3], self.channel_list[3])

        self.conv0_2 = ConvBlock(self.channel_list[0]*2+self.channel_list[1], self.channel_list[0], self.channel_list[0])
        self.conv1_2 = ConvBlock(self.channel_list[1]*2+self.channel_list[2], self.channel_list[1], self.channel_list[1])
        self.conv2_2 = ConvBlock(self.channel_list[2]*2+self.channel_list[3], self.channel_list[2], self.channel_list[2])

        self.conv0_3 = ConvBlock(self.channel_list[0]*3+self.channel_list[1], self.channel_list[0], self.channel_list[0])
        self.conv1_3 = ConvBlock(self.channel_list[1]*3+self.channel_list[2], self.channel_list[1], self.channel_list[1])

        self.conv0_4 = ConvBlock(self.channel_list[0]*4+self.channel_list[1], self.channel_list[0], self.channel_list[0])

    def get_final_conv(self):
        conv_mapping = {
            self.input_size/2**5: self.conv4_0,
            self.input_size/2**4: self.conv2_2,
            self.input_size/2**3: self.conv1_3,
            self.input_size/2**2: self.conv0_4,
        }
        return conv_mapping[self.heatmap_size]

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1)) if self.heatmap_size >= self.heatmap_size_list[0] else None

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1)) if self.heatmap_size >= self.heatmap_size_list[1] else None
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1)) if self.heatmap_size >= self.heatmap_size_list[0] else None

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1)) if self.heatmap_size >= self.heatmap_size_list[2] else None
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1)) if self.heatmap_size >= self.heatmap_size_list[1] else None
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)) if self.heatmap_size >= self.heatmap_size_list[0] else None

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1)) if self.heatmap_size >= self.heatmap_size_list[2] else None
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1)) if self.heatmap_size >= self.heatmap_size_list[1] else None
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1)) if self.heatmap_size >= self.heatmap_size_list[0] else None

        if self.heatmap_size == self.input_size / 2 ** 5:
            return x4_0
        if self.heatmap_size == self.input_size / 2 ** 4:
            return x2_2
        if self.heatmap_size == self.input_size / 2 ** 3:
            return x1_3
        if self.heatmap_size == self.input_size / 2 ** 2:
            return x0_4
        else:
            return None
