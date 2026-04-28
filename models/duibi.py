
import math
import torch
from torch import nn
import torch.nn.functional as F
from models.attention import BasicBlock, SEBasicBlock, ECALayerBlock, LayerNorm, Scale_Fusion, Scale_Fusion2, FDConv1D


class ShapeEnhanceBlock(nn.Module):
    """Enhance ECG morphology by learning a small residual from first-order slope."""

    def __init__(self, channels, gamma_init=0.01):
        super(ShapeEnhanceBlock, self).__init__()
        self.slope_filter = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        slope = x[:, :, 1:] - x[:, :, :-1]
        slope = F.pad(slope, (1, 0))
        return x + self.gamma * self.slope_filter(slope)


class LeadCalibrationBlock(nn.Module):
    """Lead-aware calibration without aggressive cross-lead mixing."""

    def __init__(self, channels, gamma_init=0.01):
        super(LeadCalibrationBlock, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        avg_desc = F.adaptive_avg_pool1d(x, 1)
        max_desc = F.adaptive_max_pool1d(x, 1)
        gate = self.gate(torch.cat([avg_desc, max_desc], dim=1))
        return x * (1.0 + self.gamma * (2.0 * gate - 1.0))


class LeadRelationGraphBlock(nn.Module):
    """Model pairwise lead relations and mix lead signals with a small residual."""

    def __init__(self, num_leads, hidden_dim=16, gamma_init=0.003):
        super(LeadRelationGraphBlock, self).__init__()
        self.query = nn.Linear(3, hidden_dim, bias=False)
        self.key = nn.Linear(3, hidden_dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.scale = hidden_dim ** -0.5
        self.num_leads = num_leads

    def forward(self, x):
        avg_desc = x.mean(dim=-1)
        max_desc = x.max(dim=-1).values
        std_desc = x.std(dim=-1, unbiased=False)
        desc = torch.stack([avg_desc, max_desc, std_desc], dim=-1)  # [B, leads, 3]
        q = self.query(desc)
        k = self.key(desc)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        mixed = torch.bmm(attn, x)
        return x + self.gamma * (mixed - x)


class FrequencyBandGate(nn.Module):
    """Use coarse frequency-band descriptors to recalibrate temporal features."""

    def __init__(self, channels, bands=4, reduction=8, gamma_init=0.02):
        super(FrequencyBandGate, self).__init__()
        hidden = max(channels // reduction, 8)
        self.bands = bands
        self.fc = nn.Sequential(
            nn.Linear(channels * (bands + 1), hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        b, c, _ = x.shape
        avg_desc = F.adaptive_avg_pool1d(x, 1).view(b, c)
        freq_mag = torch.fft.rfft(x.float(), dim=-1).abs()
        freq_desc = F.adaptive_avg_pool1d(freq_mag, self.bands).to(dtype=x.dtype).reshape(b, c * self.bands)
        gate = self.fc(torch.cat([avg_desc, freq_desc], dim=1)).view(b, c, 1)
        return x * (1.0 + self.gamma * (2.0 * gate - 1.0))


class FrequencyAuxBranch(nn.Module):
    """A lightweight frequency-domain classifier used as a residual cue."""

    def __init__(self, input_channels, num_classes, hidden_channels=32, freq_bins=128, dropout=0.1):
        super(FrequencyAuxBranch, self).__init__()
        self.freq_bins = freq_bins
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x):
        freq = torch.fft.rfft(x.float(), dim=-1).abs()
        freq = torch.log1p(freq)
        freq = F.adaptive_avg_pool1d(freq, self.freq_bins).to(dtype=x.dtype)
        feat = self.encoder(freq).squeeze(-1)
        return self.fc(self.dropout(feat))


class TFBranch(nn.Module):
    """A lightweight time-frequency branch based on STFT magnitude maps."""

    def __init__(self, input_channels, num_classes, n_fft=64, hop_length=16, win_length=64, hidden_channels=32):
        super(TFBranch, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x):
        b, c, t = x.shape
        x_flat = x.reshape(b * c, t).float()
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        tf_map = torch.log1p(stft.abs())
        tf_map = tf_map.reshape(b, c, tf_map.size(-2), tf_map.size(-1)).to(dtype=x.dtype)
        feat = self.encoder(tf_map).flatten(1)
        return self.fc(self.dropout(feat))


class CategoryAwareFusionHead(nn.Module):
    """Per-class dynamic fusion head for branch1/branch2 logits."""

    def __init__(self, num_classes):
        super(CategoryAwareFusionHead, self).__init__()
        self.branch_gap_scale = nn.Parameter(torch.zeros(num_classes))
        self.uncert_gap_scale = nn.Parameter(torch.zeros(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, out1, out2):
        prob1 = torch.sigmoid(out1)
        prob2 = torch.sigmoid(out2)
        entropy1 = -(prob1 * torch.log(prob1.clamp_min(1e-6)) +
                     (1.0 - prob1) * torch.log((1.0 - prob1).clamp_min(1e-6))) / math.log(2.0)
        entropy2 = -(prob2 * torch.log(prob2.clamp_min(1e-6)) +
                     (1.0 - prob2) * torch.log((1.0 - prob2).clamp_min(1e-6))) / math.log(2.0)
        gate_logit = (
            self.bias +
            self.branch_gap_scale * (out2 - out1) +
            self.uncert_gap_scale * (entropy1 - entropy2)
        )
        gate = torch.sigmoid(gate_logit)
        return gate * out2 + (1.0 - gate) * out1


class BaseScale(nn.Module):
    def __init__(self, input_channels, mid_channels, stride=1):
        super(BaseScale, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size=3, stride=stride, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    #     filter_sizes = [3, 13, 23, 33]
    #     self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[0],
    #                            stride=stride, bias=False, padding=(filter_sizes[0] // 2))
    #     self.conv2 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[1],
    #                            stride=stride, bias=False, padding=(filter_sizes[1] // 2))
    #     self.conv3 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[2],
    #                            stride=stride, bias=False, padding=(filter_sizes[2] // 2))
    #     self.conv4 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[3],
    #                            stride=stride, bias=False, padding=(filter_sizes[3] // 2))
    #     self.bn = nn.BatchNorm1d(mid_channels)
    #     self.relu = nn.ReLU()
    #     self.dro = nn.Dropout(0.2)
    #     self.max = nn.MaxPool1d(2)
    #
    # def forward(self, x):
    #     x1 = self.conv1(x)
    #     x2 = self.conv2(x)
    #     x3 = self.conv3(x)
    #     x4 = self.conv4(x)
    #     x_concat = torch.mean(torch.stack([x1, x2, x3, x4], 2), 2)
    #     x_concat = self.dro(self.max(self.relu(self.bn(x_concat))))
    #     x5 = self.conv_block(x)
    #     x = x5 + x_concat
    #     return x

        # --- 替换原有的4个固定卷积核 ---
        # self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[0], ...)
        # ... 省略 ...
        # --- 使用 FDConv1D ---
        self.fdconv = FDConv1D(
            in_channels=input_channels,
            out_channels=mid_channels,
            kernel_size=15,  # 可以调整，例如 13 或 17
            groups=4,  # 4个频带组，对应原来的4个kernel
            stride=stride,
            padding=15 // 2
        )

        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU()
        self.dro = nn.Dropout(0.2)
        self.max = nn.MaxPool1d(2)

    def forward(self, x):
        # --- 原始代码 ---
        # x1 = self.conv1(x)
        # x2 = self.conv2(x)
        # x3 = self.conv3(x)
        # x4 = self.conv4(x)
        # x_concat = torch.mean(torch.stack([x1, x2, x3, x4], 2), 2)
        # --- 替换为 FDConv ---
        x_concat = self.fdconv(x)  # [B, mid_channels, L]

        x_concat = self.dro(self.max(self.relu(self.bn(x_concat))))
        x5 = self.conv_block(x)
        x = x5 + x_concat
        return x


class BaseBlock(nn.Module):
    def __init__(self, input_channels=12, mid_channels=16, stride=1):
        super(BaseBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size=11, stride=1, bias=False, padding=11 // 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=7, stride=1, bias=False, padding=7 // 2),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels * 2, mid_channels * 4, kernel_size=5, stride=1, bias=False, padding=5 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(mid_channels * 4, mid_channels * 4, kernel_size=3, stride=1, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(mid_channels * 4, mid_channels * 4, kernel_size=3, stride=1, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.BaseScale1 = BaseScale(mid_channels * 4, mid_channels * 4, stride=1)
        self.BaseScale2 = BaseScale(mid_channels * 4, mid_channels * 8, stride=2)
        self.layer1 = nn.Sequential(
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            ECALayerBlock(mid_channels * 4),
            SEBasicBlock(mid_channels * 4),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x1 = self.BaseScale1(x)
        x2 = self.BaseScale2(x)
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        return x1, x2


class Fusion_block(nn.Module):
    def __init__(self, input_channels, kernel_size=7, c=1):
        super(Fusion_block, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.norm1 = LayerNorm(input_channels * 2, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(input_channels * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(input_channels * 3, eps=1e-6, data_format="channels_first")
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels * 2, input_channels, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(input_channels),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(1),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(input_channels, input_channels // 8, 1, bias=False),
            nn.Conv1d(input_channels // 8, input_channels, 1, bias=False),
            nn.BatchNorm1d(input_channels),
        )
        self.conv1 = nn.Conv1d(input_channels, input_channels, kernel_size=1, )
        self.gelu = nn.GELU()
        self.fc = nn.Linear(372 // c, 250 // c)

        self.residual = BasicBlock(input_channels * 3, input_channels)
        self.residual2 = BasicBlock(input_channels * 2, input_channels)
        self.drop_path = nn.Identity()
        downsample = nn.Sequential(
            nn.Conv1d(input_channels * 2, input_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(input_channels)
        )
        self.layer1 = nn.Sequential(
            BasicBlock(input_channels * 2, input_channels, expansion=1, downsample=downsample),
            ECALayerBlock(input_channels),
            SEBasicBlock(input_channels),
            BasicBlock(input_channels, input_channels, expansion=1),
        )

    def forward(self, l):
        max_result = self.maxpool(l)
        avg_result = self.avgpool(l)
        max_out = self.conv_block3(max_result)
        avg_out = self.conv_block3(avg_result)
        l1 = self.sigmoid(max_out + avg_out) * l

        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l2 = self.conv_block2(result)
        l2 = self.sigmoid(l2) * l

        fuse = torch.cat([l1, l2], 1)
        fuse = self.norm1(fuse)
        fuse = self.layer1(fuse)
        fuse = self.drop_path(fuse)
        return fuse


class Dui1(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, mid_channels=16, mu2=1, sigma2=0.1,
                 use_shape_enhance=True, shape_gamma_init=0.01,
                 use_lead_calibration=True, lead_gamma_init=0.01,
                 use_lead_relation_graph=True, lead_relation_graph_dim=16, lead_relation_graph_gamma_init=0.003,
                 use_frequency_gate=True, frequency_gamma_init=0.02,
                 use_frequency_aux=True, frequency_aux_gamma_init=0.03,
                 frequency_aux_uncertainty_power=1.5,
                 frequency_aux_disagreement_power=1.0,
                 use_tf_branch=True, tf_n_fft=64, tf_hop_length=16, tf_win_length=64,
                 use_category_fusion=True):
        super(Dui1, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shape_enhance = ShapeEnhanceBlock(
            input_channels,
            gamma_init=shape_gamma_init
        ) if use_shape_enhance else nn.Identity()
        self.lead_calibration = LeadCalibrationBlock(
            input_channels,
            gamma_init=lead_gamma_init
        ) if use_lead_calibration else nn.Identity()
        self.lead_relation_graph = LeadRelationGraphBlock(
            input_channels,
            hidden_dim=lead_relation_graph_dim,
            gamma_init=lead_relation_graph_gamma_init
        ) if use_lead_relation_graph else nn.Identity()
        self.frequency_aux = FrequencyAuxBranch(
            input_channels,
            num_classes
        ) if use_frequency_aux else None
        self.frequency_aux_gamma = nn.Parameter(
            torch.tensor(float(frequency_aux_gamma_init))
        ) if use_frequency_aux else None
        self.frequency_aux_uncertainty_power = float(frequency_aux_uncertainty_power)
        self.frequency_aux_disagreement_power = float(frequency_aux_disagreement_power)
        self.tf_branch = TFBranch(
            input_channels,
            num_classes,
            n_fft=tf_n_fft,
            hop_length=tf_hop_length,
            win_length=tf_win_length
        ) if use_tf_branch else None
        self.category_fusion_head = CategoryAwareFusionHead(num_classes) if use_category_fusion else None
        self.Fusion_block1 = Fusion_block(64, c=1)
        self.Fusion_block2 = Fusion_block(128, c=1)
        self.Fusion_block3 = Fusion_block(512, c=1)
        self.conv_norm1 = nn.LayerNorm(64, eps=1e-6)
        self.conv_norm2 = nn.LayerNorm(128, eps=1e-6)
        self.conv_norm3 = nn.LayerNorm(128 * 4, eps=1e-6)
        self.BaseBlock = BaseBlock(input_channels=input_channels)
        self.layer2 = nn.Sequential(
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            ECALayerBlock(mid_channels * 4),
            SEBasicBlock(mid_channels * 4),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            ECALayerBlock(mid_channels * 4),
            SEBasicBlock(mid_channels * 4),
        )
        self.freq_gate1 = FrequencyBandGate(
            mid_channels * 4,
            gamma_init=frequency_gamma_init
        ) if use_frequency_gate else nn.Identity()
        self.freq_gate2 = FrequencyBandGate(
            mid_channels * 8,
            gamma_init=frequency_gamma_init
        ) if use_frequency_gate else nn.Identity()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.down1 = nn.Conv1d(128, 128, kernel_size=1, stride=1, bias=False)
        self.down2 = nn.Conv1d(256, 128, kernel_size=1, stride=1, bias=False)
        self.scale_f = Scale_Fusion(128, 256)
        self.scale_f2 = Scale_Fusion2(128, 256, 512)
        self.up1_2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, bias=False, padding=3 // 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(64, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状: (batch_size, 1000, 12)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, 12, 1000)
        x = self.shape_enhance(x)
        x = self.lead_relation_graph(x)
        x = self.lead_calibration(x)
        freq_out = self.frequency_aux(x) if self.frequency_aux is not None else None
        tf_logits = self.tf_branch(x) if self.tf_branch is not None else None
        x1, x2 = self.BaseBlock(x)
        x1 = self.Fusion_block1(x1)
        x2 = self.Fusion_block2(x2)
        x1 = self.freq_gate1(x1)
        x2 = self.freq_gate2(x2)
        feat1 = x1.view(x1.size(0), -1)
        feat2 = x2.view(x2.size(0), -1)
        x1 = self.avg_pool(x1).squeeze(-1)
        x2 = self.avg_pool(x2).squeeze(-1)
        x1 = self.conv_norm1(x1)
        x2 = self.conv_norm2(x2)
        out1 = self.fc1(x1)
        out2 = self.fc2(x2)
        if freq_out is not None:
            prob2 = torch.sigmoid(out2)
            prob1 = torch.sigmoid(out1)
            entropy2 = -(prob2 * torch.log(prob2.clamp_min(1e-6)) +
                         (1.0 - prob2) * torch.log((1.0 - prob2).clamp_min(1e-6)))
            uncertainty_gate = (entropy2 / math.log(2.0)).clamp(0.0, 1.0).detach()
            uncertainty_gate = uncertainty_gate.pow(self.frequency_aux_uncertainty_power)
            disagreement_gate = torch.abs(prob2 - prob1).detach().pow(self.frequency_aux_disagreement_power)
            out2 = out2 + self.frequency_aux_gamma * uncertainty_gate * disagreement_gate * freq_out
        category_fusion_logits = self.category_fusion_head(out1, out2) if self.category_fusion_head is not None else None
        return feat1, out1, feat2, out2, tf_logits, category_fusion_logits


if __name__ == '__main__':
    model = Dui1()
    input1 = torch.randn(8, 12, 1000)
    out = model(input1)
