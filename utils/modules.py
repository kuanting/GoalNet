import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

'''
2D Deformable Convolution v2 (Dai, 2018).
'''
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int
        
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=self.padding, bias=True)
        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=self.padding, bias=True)
        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=bias)
        
        # Init offset and modulator weights
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
    
    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        return torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias, padding=self.padding, mask=modulator, stride=self.stride)

'''
1. GRN from ConvNext-V2 (2023).
2. Add (N, C, H, W) code, original only has (N, H, W, C)
'''
class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer
    """
    def __init__(self, dim, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        if self.data_format == "channels_last":
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        elif self.data_format == "channels_first":
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        if self.data_format == "channels_last":
            Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        elif self.data_format == "channels_first":
            Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
            Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

'''
LayerNorm from ConvNext (2022).
'''
class LayerNorm2d(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    channels_first : [N,C,H,W]
    channels_last : [N,H,W,C]
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Perform beter than original ConvNeXt implement (e.g lower VRAM usage)
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x

'''
From DeepLabv2 (2016)
'''
class ASPP(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling)
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # global average pooling
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=18, dilation=18)
 
        self.conv_5trans = nn.Conv2d(out_channel * 5, out_channel, kernel_size=1, stride=1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')  # bilinear interpolation for upsampling
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        
        return self.conv_5trans(torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1))

'''
SCSE (Concurrent Spatial and Channel ‘Squeeze & Excitation’) attention (AG Roy, 2018)
'''
class SCSE_Attention(nn.Module):
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

"""
From Attention UNet (Ozan Oktay, 2018)
"""
class Attention_gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g & F_l recommanded as same value, otherwise may not get good result
        :param F_g: int, up-conv channels
        :param F_l: int, channels from encoder concatenate
        :param F_int: int, Usually use (F_g+F_l) // 2
        :return: Tensor, F_l channels
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x * psi

'''
1. FractalNet (2016)
2. Also see DropPath in timm module
'''
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def drop_path(self, x, training: bool = False):
        if self.drop_prob == 0. or not training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        
        return x * random_tensor
    
    def forward(self, x):
        return self.drop_path(x, self.training)
