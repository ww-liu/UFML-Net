import math

import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.dswin1d import DSwinBlock

# from models.attention import SEBasicBlock, Fusion_block
# from attention import SEBasicBlock, Fusion_block
from math import log2

class FDConv1D(nn.Module):
    """
    A corrected 1D version of Frequency Dynamic Convolution (FDConv).
    This module replaces the static multi-scale convolutions in BaseScale.
    """
    def __init__(self, in_channels, out_channels, kernel_size=15, groups=4, stride=1, padding=None):
        super(FDConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

        # Fourier Disjoint Weight (FDW) - Simplified
        # Create 'groups' kernels by learning a base weight
        # The weight tensor shape is [groups, out_channels, in_channels, kernel_size]
        self.weight = nn.Parameter(torch.randn(groups, out_channels, in_channels, kernel_size))
        # Note: Bias is typically shared across groups or omitted in dynamic convs.
        # We'll add a single bias for the final output sum.
        self.bias = nn.Parameter(torch.randn(out_channels)) if True else None

        # Kernel Spatial Modulation (KSM)
        # Learn a modulation map for each group, based on the *input* feature map
        # Input: [B, in_channels, L]
        # Output: [B, groups, L] - one modulation map per group
        self.ksm = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Global average pooling along the length dimension
            nn.Conv1d(in_channels, groups, kernel_size=1, bias=False), # Map in_channels -> groups
            nn.BatchNorm1d(groups),
            nn.Sigmoid()
        )

        # Frequency Band Modulation (FBM) - Corrected
        # This should take the *output* of the grouped convolutions and apply modulation.
        # However, the original FBM logic applies modulation *before* summation.
        # Let's correct this: FBM gets the input 'x' and generates weights for the groups.
        # We need a way to generate per-group, per-sample, per-location weights from input x.
        # A simpler approach: Use KSM output (which is already per-group) and refine it globally.
        # Or, generate weights for the *outputs* of the groups *before* summation.
        # Let's try generating weights based on the *input* again, but reshape correctly later.
        # FBM takes input x [B, Cin, L] and produces group weights [B, groups, L]
        # We reuse KSM concept but maybe make it more independent or just apply global modulation per group.
        # For simplicity, let's define FBM to generate global weights per group based on pooled input.
        # This means FBM generates [B, groups, 1]
        # Then KSM generates [B, groups, L]
        # Combined: [B, groups, L] -> applied to [groups, B, Cout, L] via broadcasting/looping

        # FBM: Generate global frequency band weights based on input x
        # Input: [B, in_channels, L]
        # Output: [B, groups, 1]
        self.fbm = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # [B, Cin, L] -> [B, Cin, 1]
            nn.Conv1d(in_channels, groups, kernel_size=1, bias=True), # [B, Cin, 1] -> [B, groups, 1]
            nn.Sigmoid()
        )


    def forward(self, x):
        B, C_in, L = x.shape
        assert C_in == self.in_channels, f"Input channels ({C_in}) do not match expected channels ({self.in_channels})"

        # FBM: Generate global modulation weights per group [B, groups, 1]
        fbm_weights_global = self.fbm(x) # [B, groups, 1]

        # KSM: Generate spatial modulation weights per group [B, groups, L]
        ksm_weights_spatial = self.ksm(x) # [B, groups, L]

        # Combine FBM (global) and KSM (spatial) modulation
        # Broadcasting: [B, groups, 1] * [B, groups, L] -> [B, groups, L]
        combined_mod_weights = fbm_weights_global * ksm_weights_spatial # [B, groups, L]

        outputs = []
        for i in range(self.groups):
            # Get the i-th kernel: [out_channels, in_channels, kernel_size]
            w_i = self.weight[i]

            # Get the combined modulation for the i-th group: [B, 1, L]
            mod_i = combined_mod_weights[:, i:i+1, :] # [B, 1, L]

            # Apply spatial modulation to input feature map
            x_mod = x * mod_i # [B, C_in, L] * [B, 1, L] -> [B, C_in, L]

            # Perform convolution with the modulated input
            out_i = F.conv1d(x_mod, w_i, bias=None, stride=self.stride, padding=self.padding) # [B, out_channels, L']
            outputs.append(out_i)

        # Sum outputs from all groups
        y = sum(outputs) # [B, out_channels, L']

        # Add the single learned bias term
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1) # Broadcast bias across batch and length dims

        return y

# ==================== END OF 1D-FDConv ====================

#基础卷积块BasicBlock
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=2, downsample=None):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes * self.expansion, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
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

#注意力机制模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        # if sigmoid:
        #     self.sigmoid = nn.Sigmoid()
        # else:
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECALayerBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(ECALayerBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes, stride)
        self.group_conv3x3 = nn.Conv1d(inplanes, inplanes, kernel_size=7, stride=1,
                                       padding=7 // 2, groups=inplanes // 16, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(inplanes)
        self.se = SELayer(inplanes, reduction)
        self.ECA = ECALayer(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.group_conv3x3(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ECA(out)

        out = residual + out
        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes, stride)
        self.group_conv3x3 = nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1,
                                       padding=1, groups=inplanes // 16, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(inplanes, reduction)

    def forward(self, x):
        residual = x
        out = self.group_conv3x3(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.se(out)
        out = residual + out
        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class Scale_Fusion(nn.Module):
    def __init__(self, x1_channels, x2_channels, ):
        super(Scale_Fusion, self).__init__()

        self.up = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
        )

        self.down = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        # --- 修改 layer1 和 layer2 ---
        self.layer1 = nn.Sequential(
            GatedCNNBlock(x1_channels, expension_ratio=8 / 3, kernel_size=7, conv_ratio=1.0),
            DSwinBlock(x1_channels, window_size=16, num_heads=4),  # ← 新加这一行
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            GatedCNNBlock(x1_channels, expension_ratio=8 / 3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            GatedCNNBlock(x1_channels, expension_ratio=8 / 3, kernel_size=7, conv_ratio=1.0),
        )

        self.layer2 = nn.Sequential(
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
        )
        # --- END 修改 ---

    def forward(self, x1, x2):
        up1 = self.up(x1)
        down1 = self.down(x2)
        feat1 = down1 + x1
        feat2 = up1 + x2
        x1 = self.layer1(feat1)
        x2 = self.layer2(feat2)
        return x1, x2

class Scale_split(nn.Module):
    def __init__(self, x1_channels, x2_channels, ):
        super(Scale_split, self).__init__()

        self.up = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.down = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(size=127, mode='nearest'),
        )

        # --- 修改 layer1 和 layer2 ---
        self.layer1 = nn.Sequential(
            GatedCNNBlock(x1_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            GatedCNNBlock(x1_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            GatedCNNBlock(x1_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
        )
        self.layer2 = nn.Sequential(
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
        )
        # --- END 修改 ---

    def forward(self, x1, x2):
        up1 = self.up(x1)
        down1 = self.down(x2)
        feat1 = down1 + x1
        feat2 = up1 + x2
        x1 = self.layer1(feat1)
        x2 = self.layer2(feat2)
        return x1, x2


class Scale_Fusion2(nn.Module):
    def __init__(self, x1_channels, x2_channels, x3_channels):
        super(Scale_Fusion2, self).__init__()

        self.up1_2 = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
        )

        self.up1_3 = nn.Sequential(
            nn.Conv1d(x1_channels, x3_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x3_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.up2_3 = nn.Sequential(
            nn.Conv1d(x2_channels, x3_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x3_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.down2_1 = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        self.down3_1 = nn.Sequential(
            nn.Conv1d(x3_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=4.0, mode='nearest'),
        )

        self.down3_2 = nn.Sequential(
            nn.Conv1d(x3_channels, x2_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        # --- 修改 layer1, layer2, layer3 ---
        self.layer1 = nn.Sequential(
            GatedCNNBlock(x1_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            GatedCNNBlock(x1_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            GatedCNNBlock(x1_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
        )
        self.layer2 = nn.Sequential(
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            GatedCNNBlock(x2_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
        )
        self.layer3 = nn.Sequential(
            GatedCNNBlock(x3_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x3_channels),
            ECALayerBlock(x3_channels),
            GatedCNNBlock(x3_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
            SEBasicBlock(x3_channels),
            ECALayerBlock(x3_channels),
            GatedCNNBlock(x3_channels, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0),
        )
        # --- END 修改 ---

    def forward(self, x1, x2, x3):
        # up1_2 = self.up1_2(x1)
        # up1_3 = self.up1_3(x1)
        # up2_3 = self.up2_3(x2)
        # down2_1 = self.down2_1(x2)
        # down3_1 = self.down3_1(x3)
        # down3_2 = self.down3_2(x3)
        # x1 = down2_1 + down3_1 + x1
        # x2 = up1_2 + down3_2 + x2
        # x3 = up1_3 + up2_3 + x3
        # x1 = self.layer1(x1)
        # x2 = self.layer2(x2)
        # x3 = self.layer3(x3)

        up1_2 = self.up1_2(x1)
        up1_3 = self.up1_3(x1)
        up2_3 = self.up2_3(x2)
        # down2_1 = self.down2_1(x2)
        # down3_1 = self.down3_1(x3)
        # down3_2 = self.down3_2(x3)
        x1 = x1
        x2 = up1_2 + x2
        x3 = up1_3 + up2_3 + x3
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)

        return x1, x2, x3


# --- 新增 GatedCNNBlock ---
class GatedCNNBlock(nn.Module):
    def __init__(self, dim, expension_ratio=8/3, kernel_size=7, conv_ratio=1.0, norm_layer=nn.BatchNorm1d, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expension_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2) # 扩展通道
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        # 分割索引: (g, i, c)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        # 使用 1D 深度可分离卷积
        self.conv = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        shortcut = x
        # 假设输入形状为 [B, C, L]
        x = self.norm(x.permute(0, 2, 1)) # 转换为 [B, L, C] 以便进行线性层操作
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1) # 分割通道
        # 对 c 部分进行深度可分离卷积
        c = c.permute(0, 2, 1) # 转换回 [B, C, L]
        c = self.conv(c)
        c = c.permute(0, 2, 1) # 再次转换为 [B, L, C]
        # 合并 g, i, c 并通过第二个线性层
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = x.permute(0, 2, 1) # 转换回 [B, C, L]
        return x + shortcut
# --- END 新增 ---


# if __name__ == '__main__':
#     a = torch.randn(8, 12, 1000)
#     out = model(a)