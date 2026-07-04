# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG backbone
class Base_VGG(nn.Module):
    def __init__(self, name: str, last_pool=False, num_channels=256,
                 dcnv2_enabled=False, dcnv2_layers=0, **kwargs):
        super().__init__()
        print("### VGG16: last_pool=", last_pool,
              "dcnv2_enabled=", dcnv2_enabled,
              "dcnv2_layers=", dcnv2_layers)
        # loading backbone features
        from .backbones import vgg as models
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)

        features = list(backbone.features.children())

        # setting base module.
        if name == 'vgg16_bn':
            self.body1 = nn.Sequential(*features[:13])
            self.body2 = nn.Sequential(*features[13:23])
            self.body3 = nn.Sequential(*features[23:33])
            if last_pool:
                self.body4 = nn.Sequential(*features[33:44])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[33:43])  # 16x down-sample
        else:
            self.body1 = nn.Sequential(*features[:9])
            self.body2 = nn.Sequential(*features[9:16])
            self.body3 = nn.Sequential(*features[16:23])
            if last_pool:
                self.body4 = nn.Sequential(*features[23:31])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[23:30])  # 16x down-sample

        # -------- DCNv2 injection (direction D) --------
        if dcnv2_enabled and dcnv2_layers > 0:
            self._inject_dcnv2(dcnv2_layers)

        self.num_channels = num_channels
        self.last_pool = last_pool

    def _inject_dcnv2(self, dcnv2_layers):
        """Replace the last N Conv2d layers in body4 with ModulatedDeformConv2d.

        Uses explicit index assignment (body4[idx] = new_mod) which goes through
        nn.Sequential.__setitem__ → setattr — the container is truly updated.
        """
        from .dcnv2 import ModulatedDeformConv2d

        # Find all Conv2d indices in body4
        conv_indices = [i for i, m in enumerate(self.body4) if isinstance(m, nn.Conv2d)]
        target_indices = conv_indices[-dcnv2_layers:]  # last N

        for idx in target_indices:
            old_conv = self.body4[idx]
            new_mod = ModulatedDeformConv2d(
                old_conv.in_channels, old_conv.out_channels,
                old_conv.kernel_size[0], old_conv.stride[0], old_conv.padding[0],
                bias=(old_conv.bias is not None)
            )
            # Transfer pretrained weights
            new_mod.deform_weight.data.copy_(old_conv.weight.data)
            if old_conv.bias is not None:
                new_mod.bias.data.copy_(old_conv.bias.data)
            # Replace in container (nn.Sequential.__setitem__ → setattr)
            self.body4[idx] = new_mod
            print(f"  [DCNv2] body4[{idx}] Conv2d → ModulatedDeformConv2d "
                  f"({old_conv.in_channels}→{old_conv.out_channels})")
        
    def get_outplanes(self):
        outplanes = []
        for i in range(4):
            last_dims = 0
            for param_tensor in self.__getattr__('body'+str(i+1)).state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(self.__getattr__('body'+str(i+1)).state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes   # get the last layer params of all modules, and trans to the size.

    def forward(self, tensor_list):
        out = []
        xs = tensor_list
        for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
            xs = layer(xs)
            out.append(xs)
        return out

# ResNet backbone
class Base_ResNet(nn.Module):
    def __init__(self, name: str, last_pool=False , num_channels=256, **kwargs):
        super().__init__()
        print("### ResNet: last_pool=", last_pool)
        # loading backbone features
        from .backbones import resnet as models
        if name == 'resnet18':
            self.backbone = models.resnet18_ibn_a(pretrained=True)
        elif name == 'resnet34':
            self.backbone = models.resnet34_ibn_a(pretrained=True)
        elif name == 'resnet50':
            self.backbone = models.resnet50_ibn_a(pretrained=True)
        elif name == 'resnet101':
            self.backbone = models.resnet101_ibn_a(pretrained=True)
        elif name == 'resnet152':
            self.backbone = models.resnet152_ibn_a(pretrained=True)     

        self.num_channels = num_channels
        self.last_pool = last_pool

    def get_outplanes(self):
        outplanes = []
        for Layer in [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]:
            last_dims = 0
            for param_tensor in Layer.state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(Layer.state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes   # get the last layer params of all modules, and trans to the size.

    def forward(self, tensor_list):
        out = []
        xs = tensor_list
        out = self.backbone(xs)
        return out