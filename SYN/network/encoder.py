from __future__ import print_function
import ipdb
import torch
import torch.nn as nn
import torch.utils.data


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

#S3.1 Deep Feature Extraction,maybe use resnet50
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        # ipdb.set_trace()
        self.inplanes = 32
        # Helge: set it to 19 from 18
        self.firstconv = nn.Sequential(convbn(19, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 2, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.lastconv = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(64, 64, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))


    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # ipdb.set_trace()
        output = self.firstconv(x)
        # print('output.shape=', output.shape)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output= self.layer4(output)
        output = self.lastconv(output)

        return output

# ReLU-Pooling：In total, five ConvolutionReLU-Pooling structures are used to 
# get the output with a dimension of F = 256. 
class fcn(nn.Module):  
    def __init__(self):
        super(fcn, self).__init__()
        self.conv1 = nn.Conv2d(64, 96, 3, padding=1)#3x3 conv layer
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)#max pooling layer:1st 2 is kernel, while 2nd 2 is the stride size

        self.conv2 = nn.Conv2d(96, 128, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(128, 152, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4 = nn.Conv2d(152, 196, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5 = nn.Conv2d(196, 256, 3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))

        return x



class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        # vggnet = models.__dict__['vgg13_bn'](pretrained=True)
        # resnet34 = models.__dict__['resnet34'](pretrained=True)
        # Remove last max pooling layer of vggnet
        # print(resnet18)
        # print(list(vggnet.features.children())[:-2])
        self.encoder = feature_extraction()

    def forward(self, clips):
        # Permute to run encoder on batch of each frame
        # NOTE: This requires clips to have the same number of frames!!
        # print('clips01=',clips.shape)
        frame_ordered_clips = clips.permute(1, 0, 2, 3, 4)
        # print('clips02=',frame_ordered_clips.shape)
        clips_feature_maps = [self.encoder(frame) for frame in frame_ordered_clips]

        return torch.stack(clips_feature_maps, dim=0).permute(1, 0, 2, 3, 4)

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        # vggnet = models.__dict__['vgg13_bn'](pretrained=True)
        # resnet34 = models.__dict__['resnet34'](pretrained=True)
        # Remove last max pooling layer of vggnet
        # print(resnet18)
        # print(list(vggnet.features.children())[:-2])
        self.encoder = fcn()

    def forward(self, clips):
        # Permute to run encoder on batch of each frame
        # NOTE: This requires clips to have the same number of frames!!
        frame_ordered_clips = clips.permute(1, 0, 2, 3, 4)
        clips_feature_maps = [self.encoder(frame) for frame in frame_ordered_clips]

        return torch.stack(clips_feature_maps, dim=0).permute(1, 0, 2, 3, 4)

