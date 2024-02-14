import os
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
import config
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
from attention import EnhancedCoordAttention  #CEAM_Block #attention_layer EnhancedCoordAttention
from modelBiFPN import BiFPN, Conv2dStaticSamePadding

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, stride=stride, kernel_size=3, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=21, scale=1, groups=1, width_per_group=64):
        self.inplanes = 64      
# self.inplanes是block的输入通道数，planes是做3x3卷积的空间的通道数，expansion是残差结构中输出维度是输入维度的多少倍，
# 同一个stage内，第一个block，inplanes=planes， 输出为planes*block.expansion
# 第二个block开始，该block的inplanes等于上一层的输出的通道数planes*block.expansion
# （类似于卷积后的结果进入下一个卷积时，前一个卷积得到的output的输出为下一个卷积的input）
        super(ResNet, self).__init__()

        self.groups = groups
        self.width_per_group = width_per_group

        #处理输入的C1模块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 搭建自上而下的C2、C3、C4、C5
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        #统一通道数，原来的通道数降为同一个值或者升为同一个值，才能进入BiFPN特征融合
        self.d1_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(64, 256, 1),
            nn.BatchNorm2d(256, momentum=0.01, eps=1e-3),
        )
        self.d2_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(256, 256, 1),
            nn.BatchNorm2d(256, momentum=0.01, eps=1e-3),
        )
        self.d3_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(512, 256, 1),
            nn.BatchNorm2d(256, momentum=0.01, eps=1e-3),
        )
        self.d4_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(1024, 256, 1),
            nn.BatchNorm2d(256, momentum=0.01, eps=1e-3),
        )
        self.d5_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(2048, 256, 1),
            nn.BatchNorm2d(256, momentum=0.01, eps=1e-3),
        )

        
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # 对C5减少通道数得到P5  #Reduce channels
        self.toplayer_bn = nn.BatchNorm2d(256)
        self.toplayer_relu = nn.ReLU(inplace=True)

        # Smooth layers #3x3卷积融合特征
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(256)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(256)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(256)
        self.smooth3_relu = nn.ReLU(inplace=True)
        
        '''
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_bn = nn.BatchNorm2d(256)
        self.smooth4_relu = nn.ReLU(inplace=True)
        '''

        # Lateral layers   #横向连接，保证通道数相同
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(256)
        self.latlayer1_relu = nn.ReLU(inplace=True)

        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(256)
        self.latlayer2_relu = nn.ReLU(inplace=True)

        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(256)
        self.latlayer3_relu = nn.ReLU(inplace=True)
        
        '''
        self.latlayer4 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4_bn = nn.BatchNorm2d(256)
        self.latlayer4_relu = nn.ReLU(inplace=True)      
        '''
        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attention_p1top2 = EnhancedCoordAttention(256,256)
        self.attention_p2top3 = EnhancedCoordAttention(256,256)  #CEAM_Block(256)  attention_layer() EnhancedCoordAttention(256,256)
        self.attention_p3top4 = EnhancedCoordAttention(256,256)  #CEAM_Block(256)  attention_layer() EnhancedCoordAttention(256,256)
        self.attention_p4top5 = EnhancedCoordAttention(256,256)  #CEAM_Block(256)  attention_layer() EnhancedCoordAttention(256,256)

        #BiFPN first_time=False BIFPN只重复了一次
        #self.bifpn = BiFPN(256, [64, 256, 512], first_time=False, epsilon=1e-4, onnx_export=False, attention=True, use_p8=False)
        self.bifpn2 = BiFPN(256, [64, 256, 512], first_time=False, epsilon=1e-4, onnx_export=False, attention=True, use_p8=False)
        # 变为原来的通道数,进行特征融合、特征计算
        self.d1_up_channel = nn.Sequential(
            Conv2dStaticSamePadding(256, 64, 1),
            nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),
        )
        self.d2_up_channel = nn.Sequential(
            Conv2dStaticSamePadding(256, 256, 1),
            nn.BatchNorm2d(256, momentum=0.01, eps=1e-3),
        )
        self.d3_up_channel = nn.Sequential(
            Conv2dStaticSamePadding(256, 512, 1),
            nn.BatchNorm2d(512, momentum=0.01, eps=1e-3),
        )
        self.d4_up_channel = nn.Sequential(
            Conv2dStaticSamePadding(256, 1024, 1),
            nn.BatchNorm2d(1024, momentum=0.01, eps=1e-3),
        )
        self.d5_up_channel = nn.Sequential(
            Conv2dStaticSamePadding(256, 2048, 1),
            nn.BatchNorm2d(2048, momentum=0.01, eps=1e-3),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 构建C2到C5需要注意stride值为1和2的情况
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups, width_per_group=self.width_per_group))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    # 上采样放大图像，分辨率变高
    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        #return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')
        #nn.functional.interpolate  UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
        #warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")

    # 自上而下的上采样模块
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        #return F.upsample(x, size=(H, W), mode='bilinear') + y
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # h = x
        # h = self.conv1(h)
        # h = self.bn1(h)
        # h = self.relu1(h)
        # h = self.maxpool(h)
        # d1 = h
        # h = self.layer1(h)
        # d2 = h
        # h = self.layer2(h)
        # d3 = h
        # h = self.layer3(h)
        # d4 = h
        # h = self.layer4(h)
        # d5 = h

        # 自下而上  # 上边的多行代码可以换为以下少量代码
        d1 = self.maxpool(self.relu1(self.bn1(self.conv1(x))))
        d2 = self.layer1(d1)
        d3 = self.layer2(d2)
        d4 = self.layer3(d3)
        d5 = self.layer4(d4)
        # print(d1.size())  # Resnet50出来的5层输出如下   # torch.Size([64, 64, 56, 56])      
        # print(d2.size())                                # torch.Size([64, 256, 56, 56])
        # print(d3.size())                                # torch.Size([64, 512, 28, 28])
        # print(d4.size())                                # torch.Size([64, 1024, 14, 14])
        # print(d5.size())                                # torch.Size([64, 2048, 7, 7])

        e1 = self.d1_down_channel(d1)
        e2 = self.d2_down_channel(d2)
        e3 = self.d3_down_channel(d3)
        e4 = self.d4_down_channel(d4)
        e5 = self.d5_down_channel(d5)
        # print(e1.size())
        # print(e2.size())
        # print(e3.size())
        # print(e4.size())
        # print(e5.size())

        inputs = [e1, e2, e3, e4, e5]
        # print(inputs) #输出结果是tensor:[tensor([[[[ 2.2302e-01, -8.2730e-01, -8.9046e-01,  ..., -1.3193e+00,
        #g1, g2, g3, g4, g5 = self.bifpn(inputs)
        #inputs = [g1, g2, g3, g4, g5]
        f1, f2, f3, f4, f5 = self.bifpn2(inputs)

        # print(f1.size())
        # print(f2.size())
        # print(f3.size())
        # print(f4.size())
        # print(f5.size())

        f5 = self.d5_up_channel(f5)
        f4 = self.d4_up_channel(f4)
        f3 = self.d3_up_channel(f3)
        f2 = self.d2_up_channel(f2)
        #f1 = self.d1_up_channel(f1)
        # print(f1.size()) 
        # print(f2.size())
        # print(f3.size())
        # print(f4.size())
        # print(f5.size())

        
        # Top-down  #自上而下
        p5 = self.toplayer(f5)
        p5 = self.toplayer_relu(self.toplayer_bn(p5))

        f4 = self.latlayer1(f4)
        f4 = self.latlayer1_relu(self.latlayer1_bn(f4))
        p4 = self._upsample_add(p5, f4)
        p4 = self.smooth1(p4)
        p4 = self.smooth1_relu(self.smooth1_bn(p4))

        f3 = self.latlayer2(f3)
        f3 = self.latlayer2_relu(self.latlayer2_bn(f3))
        p3 = self._upsample_add(p4, f3)
        p3 = self.smooth2(p3)
        p3 = self.smooth2_relu(self.smooth2_bn(p3))

        f2 = self.latlayer3(f2)
        f2 = self.latlayer3_relu(self.latlayer3_bn(f2))
        p2 = self._upsample_add(p3, f2)
        p2 = self.smooth3(p2)
        p2 = self.smooth3_relu(self.smooth3_bn(p2))
                
        p1 = f1
        
        '''
        f1 = self.latlayer4(f1)
        f1 = self.latlayer4_relu(self.latlayer4_bn(f1))
        p1 = self._upsample_add(p2, f1)
        p1 = self.smooth4(p1)
        p1 = self.smooth4_relu(self.smooth4_bn(p1))
        '''
       
        #p2 = f2
        #p3 = self._upsample(f3, p2)
        #p4 = self._upsample(f4, p2)
        #p5 = self._upsample(f5, p2)
        
        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)
        
        # print(p1.size())   
        # p2 p3 p4 p5通道数、尺寸长宽均相同 batchsize=32 通道数256 # torch.Size([32, 256, 56, 56])         
        # print(p2.size())
        # print(p3.size())
        # print(p4.size())
        # print(p5.size())
        
        #p1_1 = p1
        p2_1 = self.attention_p1top2(p1, p2)         # p2_1 p3_1 p4_1 p5_1通道数、尺寸长宽均相同 # torch.Size([32, 256, 56, 56])
        p3_1 = self.attention_p2top3(p2, p3) 
        #p3_1 = p3
        p4_1 = self.attention_p3top4(p3, p4)
        p5_1 = self.attention_p4top5(p4, p5)
        
        # print(p2_1.size())
        # print(p3_1.size())
        # print(p4_1.size())
        # print(p5_1.size())

        out = torch.cat((p2_1, p3_1, p4_1, p5_1), 1)
        #out = torch.cat((p1_1, p2_1, p3_1, p4_1, p5_1), 1)
        #out = torch.cat((p2, p3, p4, p5), 1)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return out

        # c1 = h
        # h = self.layer1(h)
        # c2 = h
        # h = self.layer2(h)
        # c3 = h
        # h = self.layer3(h)
        # c4 = h
        # h = self.layer4(h)
        # c5 = h
        # Top-down
        #p5 = self.toplayer(c5)
        #p5 = self.toplayer_relu(self.toplayer_bn(p5))

        #c4 = self.latlayer1(c4)
        #c4 = self.latlayer1_relu(self.latlayer1_bn(c4))
        #p4 = self._upsample_add(p5, c4)
        #p4 = self.smooth1(p4)
        #p4 = self.smooth1_relu(self.smooth1_bn(p4))

        #c3 = self.latlayer2(c3)
        #c3 = self.latlayer2_relu(self.latlayer2_bn(c3))
        #p3 = self._upsample_add(p4, c3)
        #p3 = self.smooth2(p3)
        #p3 = self.smooth2_relu(self.smooth2_bn(p3))

        #c2 = self.latlayer3(c2)
        #c2 = self.latlayer3_relu(self.latlayer3_bn(c2))
        #p2 = self._upsample_add(p3, c2)
        #p2 = self.smooth3(p2)
        #p2 = self.smooth3_relu(self.smooth3_bn(p2))

        #p3 = self._upsample(p3, p2)
        #p4 = self._upsample(p4, p2)
        #p5 = self._upsample(p5, p2)
'''
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                if "fc" not in key and "features.13" not in key:
                    state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model
'''

def resnext50_32x4d(pretrained=False, **kwargs):
    # resnext50:https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=groups, width_per_group=width_per_group, **kwargs)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnext50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                if "fc" not in key and "features.13" not in key:
                    state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model

class CFEANet(nn.Module):
    def __init__(self):
        super(CFEANet, self).__init__()

        self.backbone = self._get_backbone()
        self.fc = nn.Linear(1024, config.NUM_CLASSES)
        # self.softmax = nn.LogSoftmax(dim=1)
    def _get_backbone(self):
        backbone = resnext50_32x4d(pretrained=True, num_classes=21)

        for param in backbone.layer1.parameters():
            param.requires_grad = False
        for param in backbone.layer2.parameters():
            param.requires_grad = False
        for param in backbone.layer3.parameters():
            param.requires_grad = False
        for param in backbone.layer4.parameters():
            param.requires_grad = False
        return backbone

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(-1,1024)
        out = self.fc(out)
        return out