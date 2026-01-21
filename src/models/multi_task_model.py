# from models.block import DownBlock, UpBlock, ResidualConvBlock, ConvBlock
import torch
import torch.nn as nn
import torchvision.models as models


class decode(torch.nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel):
        super(decode, self).__init__()
        self.transpose = torch.nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2)
        self.conv1 = torch.nn.Conv2d(in_channel + middle_channel, out_channel,  kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.indenti1 = torch.nn.Identity()

        self.conv2 = torch.nn.Conv2d(out_channel, out_channel,  kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.indenti2 = torch.nn.Identity()


    def forward(self, x1, x2):
        x1 = self.transpose(x1)
        if x1.size() != x2.size():
            x1 = nn.functional.interpolate(x1, size=x2.shape[2:],
                                            mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.indenti1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.indenti2(x)

        return x
class MultiTaskModelResNet(nn.Module):
    def __init__(self, n_classes, n_segment, in_channels, pretrained=True):
        super(MultiTaskModelResNet, self).__init__()

        # load pretrained resnet50
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)

        # encoder
        self.block_input = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool

        self.encode1 = resnet.layer1
        self.encode2 = resnet.layer2
        self.encode3 = resnet.layer3
        self.encode4 = resnet.layer4

        # decoder

        self.decode1 = decode(2048, 1024, 512)
        self.decode2 = decode(512, 512, 256)
        self.decode3 = decode(256, 256, 128)
        self.decode4 = decode(128, 64, 64)

        # output segment
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_segment, kernel_size=1)  # Output layer
        )
        # output classify
        self.classify = nn.Sequential(
            resnet.avgpool,
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048,
                        out_features=n_classes,
                        bias=True)
        )

    def forward(self, inputs):
        x1 = self.block_input(inputs)
        x1_maxpool = self.maxpool(x1)
        x2 = self.encode1(x1_maxpool)
        x3 = self.encode2(x2)
        x4 = self.encode3(x3)
        x5 = self.encode4(x4)

        x6 = self.decode1(x5, x4)
        x7 = self.decode2(x6, x3)
        x8 = self.decode3(x7, x2)
        x9 = self.decode4(x8, x1)



        out_classify = self.classify(x5)

        out_segment = self.segmentation_head(x9)

        return out_classify, out_segment



# class MultiTaskModelEfficient(nn.Module):
#     def __init__(self, n_classes, n_segment, in_channels):
#         super(MultiTaskModelEfficient, self).__init__()

#         # load pretrained EfficientNet-B4
#         efficient = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
#         features = efficient.features

#         # encoder
#         self.encoder1 = nn.Sequential(
#             features[0],
#             features[1]
#         )
#         self.encoder2 = nn.Sequential(
#             features[2]
#         )
#         self.encoder3 = nn.Sequential(
#             features[3],
#             features[4]
#         )
#         self.encoder4 = nn.Sequential(
#             features[5],
#             features[6]
#         )
#         self.encoder5 = nn.Sequential(
#             features[7],
#             features[8]
#         )

#         self.brigde = nn.Sequential(
#             nn.BatchNorm2d(1792, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True),
#             nn.AdaptiveAvgPool2d(output_size=1),
#             nn.Dropout(0.1),
#             nn.SiLU(),
#         )

#         # decoder

#         self.decode1 = decode(1792, 272, 256) # In: 1792, Skip: 160
#         self.decode2 = decode(256, 112, 128)   # In: 256, Skip: 56
#         self.decode3 = decode(128, 32, 64)    # In: 128, Skip: 32
#         self.decode4 = decode(64, 24, 32)     # In: 64, Skip: 24

#         # output segment
#         self.segmentation_head = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, n_segment, kernel_size=1)  # Output layer
#         )

#         # output classify
#         self.classify = nn.Sequential(
#             nn.AdaptiveAvgPool2d(output_size=1),
#             nn.Flatten(),
#             nn.Dropout(p=0.4),
#             nn.Linear(in_features=1792,
#                       out_features=n_classes,
#                       bias=True)
#         )

#     def forward(self, inputs):
#         x1 = self.encoder1(inputs)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#         x4 = self.encoder4(x3)
#         x5 = self.encoder5(x4)

#         x_brigde = self.brigde(x5)

#         x6 = self.decode1(x_brigde, x4)
#         x7 = self.decode2(x6, x3)
#         x8 = self.decode3(x7, x2)
#         x9 = self.decode4(x8, x1)



#         out_classify = self.classify(x5)


#         out_segment = self.segmentation_head(x9)

#         return out_classify, out_segment
