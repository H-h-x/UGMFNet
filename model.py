import torch.nn.functional as F
import torch
import torch.nn as nn

# 第一步：优化Xvolution核心（融合Depthwise Separable Convolution）
class Xvolution(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=None, groups=1, bias=False):
        super(Xvolution, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups  # 用于Depthwise卷积的分组数

        # 计算padding（默认same padding，保证输入输出尺寸一致）
        if padding is None:
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = padding

        # 分支1：Depthwise卷积（深度可分离的核心，减少计算量）
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # Depthwise输出通道数=输入通道数
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=in_channels,  # 分组数=输入通道数，实现Depthwise
            bias=bias
        )

        # 分支2：轻量自注意力近似（保留Xvolution的全局建模能力）
        self.reduce = nn.Conv2d(in_channels, in_channels // groups, 1, bias=bias)  # 降维
        self.span = nn.Conv2d(in_channels // groups, kernel_size * kernel_size, 1, bias=bias)  # 生成注意力核
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=self.padding)

        # 分支融合+通道调整（Pointwise卷积，深度可分离的逐点融合）
        self.pw_fusion = nn.Conv2d(
            in_channels=in_channels,  # 双分支输出均为in_channels，直接叠加后输入
            out_channels=in_channels,  # 保持通道数一致
            kernel_size=1,  # Pointwise卷积
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 分支1：Depthwise卷积（局部特征提取，轻量化）
        dw_out = self.dw_conv(x)  # [B, C, H', W']，H'=H/stride

        # 分支2：轻量自注意力（全局依赖捕捉）
        kernel = self.span(self.reduce(x)).reshape(B, 1, self.kernel_size**2, H, W)  # [B, 1, k², H, W]
        kernel = kernel.permute(0, 3, 4, 1, 2)  # [B, H, W, 1, k²]
        x_unfold = self.unfold(x).reshape(B, C, self.kernel_size**2, H//self.stride, W//self.stride)  # [B, C, k², H', W']
        x_unfold = x_unfold.permute(0, 3, 4, 1, 2)  # [B, H', W', C, k²]
        attn_out = (x_unfold * kernel).sum(dim=-1).permute(0, 3, 1, 2)  # [B, C, H', W']

        # 双分支融合：叠加后通过Pointwise卷积统一特征
        fused_out = self.pw_fusion(dw_out + attn_out)  # [B, C, H', W']

        return fused_out


# 第二步：优化LightXvolution（进一步轻量化，可选通道升维）
class LightXvolution(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=None, groups=1, bias=False, expand_ratio=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.expand_channels = int(in_channels * expand_ratio)  # 可选通道扩展（默认不扩展）

        # 核心：轻量化Xvolution（已融合Depthwise+自注意力）
        self.core_xvolution = Xvolution(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

        # 额外Pointwise卷积（灵活调整通道数，提升特征表达）
        self.pw_expand = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.expand_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        # 批归一化+激活（增强稳定性和非线性）
        self.bn = nn.BatchNorm2d(self.expand_channels)
        self.relu = nn.ReLU6(inplace=True)  # ReLU6更适合轻量化网络

    def forward(self, x):
        # 流程：核心Xvolution → 通道调整 → BN+激活
        x = self.core_xvolution(x)
        x = self.pw_expand(x)
        x = self.relu(self.bn(x))
        return x

# 第三步：Spectral_Weight（LightXvolution替换，保持原有功能）
class Spectral_Weight(nn.Module):
    """光谱权重生成模块（LightXvolution+Depthwise Separable，轻量化+强建模）"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=False):
        super(Spectral_Weight, self).__init__()
        self.f_inv_11 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=groups, bias=bias)
        self.f_inv_12 = LightXvolution(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            expand_ratio=1.0
        )
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.relu(self.bn_h(self.f_inv_11(self.f_inv_12(X_h))))
        return X_h

# HSI特征提取模块（保持原有输出，用于后续对比计算）
class Feature_HSI(nn.Module):
    """HSI（高光谱图像）特征提取模块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Feature_HSI, self).__init__()
        self.stride = stride
        self.F_1 = nn.Conv2d(in_channels, 16, kernel_size, stride, padding, dilation, groups, bias)
        self.F_2 = nn.Conv2d(16, 32, kernel_size, stride, padding, dilation, groups, bias)
        self.F_3 = nn.Conv2d(32, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.F_3(self.F_2(self.F_1(X_h)))
        X_h = self.relu(self.bn_h(X_h))
        return X_h

# Spatial_Weight模块（保持原有功能，输出LIDAR特征用于对比）
class Spatial_Weight(nn.Module):
    """空间权重生成模块（需要kernel_size参数）"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spatial_Weight, self).__init__()
        self.Conv_weight = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.relu(self.bn_h(self.Conv_weight(X_h)))
        return X_h


# 核心修改：融合“预测结果置信度”和“特征层面置信度”
class Spectral_Spatial_Fusion(nn.Module):
    """融合模块：双重置信度（预测结果+特征层面）加权融合"""
    def __init__(self, weight_channels, n_classes, hidden=[128, 64], dropout=0.5):
        super(Spectral_Spatial_Fusion, self).__init__()
        self.weight_channels = weight_channels
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 原有分类器（用于生成logits，计算预测结果置信度）
        self.clf_spectral = nn.ModuleList()
        spectral_last_size = weight_channels
        for hid in hidden:
            self.clf_spectral.append(nn.Linear(spectral_last_size, hid))
            self.clf_spectral.append(nn.ReLU())
            self.clf_spectral.append(nn.Dropout(dropout))
            spectral_last_size = hid
        self.clf_spectral.append(nn.Linear(spectral_last_size, n_classes))

        self.clf_spatial = nn.ModuleList()
        spatial_last_size = weight_channels
        for hid in hidden:
            self.clf_spatial.append(nn.Linear(spatial_last_size, hid))
            self.clf_spatial.append(nn.ReLU())
            self.clf_spatial.append(nn.Dropout(dropout))
            spatial_last_size = hid
        self.clf_spatial.append(nn.Linear(spatial_last_size, n_classes))

        # 新增：可学习的模态原型特征（用于特征层面置信度）
        self.hsi_prototype = nn.Parameter(torch.randn(1, weight_channels))  # HSI模态原型 [1, C]
        self.lidar_prototype = nn.Parameter(torch.randn(1, weight_channels))  # LIDAR模态原型 [1, C]

        # 新增：特征层面相似度权重（平衡模态内和模态间）
        self.intra_weight = nn.Parameter(torch.tensor(0.5))  # 模态内相似度权重
        self.inter_weight = nn.Parameter(torch.tensor(0.5))  # 模态间相似度权重

        # 新增：双重置信度融合权重（可学习，平衡预测结果和特征层面）
        self.pred_weight = nn.Parameter(torch.tensor(0.5))  # 预测结果置信度权重
        self.feat_weight = nn.Parameter(torch.tensor(0.5))  # 特征层面置信度权重

    def forward(self, spectral_weight, spatial_weight, hsi_feat, lidar_feat):
        """
        输入：
            spectral_weight: 光谱权重 [B, C, H, W]
            spatial_weight: 空间权重 [B, C, H, W]
            hsi_feat: HSI基础特征 [B, C, H, W]（用于特征置信度）
            lidar_feat: LIDAR特征 [B, C, H, W]（用于特征置信度）
        """
        B, C, H, W = spectral_weight.shape

        # -------------------------- 第一部分：原有预测结果置信度（基于logits能量分数）--------------------------
        # GAP压缩权重特征，生成logits
        spectral_gap = self.gap(spectral_weight).squeeze(-1).squeeze(-1)
        spatial_gap = self.gap(spatial_weight).squeeze(-1).squeeze(-1)

        spectral_logits = spectral_gap
        for layer in self.clf_spectral:
            spectral_logits = layer(spectral_logits)

        spatial_logits = spatial_gap
        for layer in self.clf_spatial:
            spatial_logits = layer(spatial_logits)

        # 计算能量分数，转换为置信度（保留原逻辑）
        spectral_energy = -torch.logsumexp(spectral_logits, dim=1)  # [B,]
        spatial_energy = -torch.logsumexp(spatial_logits, dim=1)  # [B,]

        # 归一化到0-1区间（原缩放因子-0.1，再通过sigmoid稳定）
        pred_spectral_conf = torch.sigmoid(-0.1 * spectral_energy)  # [B,]
        pred_spatial_conf = torch.sigmoid(-0.1 * spatial_energy)  # [B,]

        # -------------------------- 第二部分：新增特征层面置信度（对比学习）--------------------------
        # 1. 模态内相似度（特征与原型）
        hsi_global = self.gap(hsi_feat).squeeze(-1).squeeze(-1)  # [B, C]
        lidar_global = self.gap(lidar_feat).squeeze(-1).squeeze(-1)  # [B, C]

        hsi_intra_sim = F.cosine_similarity(hsi_global, self.hsi_prototype.expand(B, C), dim=1)  # [B,]
        lidar_intra_sim = F.cosine_similarity(lidar_global, self.lidar_prototype.expand(B, C), dim=1)  # [B,]

        # 2. 模态间相似度（HSI与LIDAR特征）
        inter_sim_global = F.cosine_similarity(hsi_global, lidar_global, dim=1)  # [B,]
        inter_sim_local = F.cosine_similarity(hsi_feat, lidar_feat, dim=1).mean(dim=[1,2])  # [B,]
        inter_sim = (inter_sim_global + inter_sim_local) / 2  # [B,]

        # 3. 特征置信度归一化
        hsi_intra_sim = (hsi_intra_sim + 1) / 2  # [-1,1] → [0,1]
        lidar_intra_sim = (lidar_intra_sim + 1) / 2
        inter_sim = (inter_sim + 1) / 2

        feat_spectral_conf = (self.intra_weight * hsi_intra_sim) + (self.inter_weight * inter_sim)  # [B,]
        feat_spatial_conf = (self.intra_weight * lidar_intra_sim) + (self.inter_weight * inter_sim)  # [B,]

        feat_spectral_conf = torch.sigmoid(feat_spectral_conf)  # [B,]
        feat_spatial_conf = torch.sigmoid(feat_spatial_conf)  # [B,]

        # -------------------------- 第三部分：双重置信度融合（总置信度）--------------------------
        # 可学习权重加权和（保证权重和为1，避免偏置）
        total_weight = self.pred_weight + self.feat_weight + 1e-8  # 防止除以0
        total_spectral_conf = (self.pred_weight * pred_spectral_conf + self.feat_weight * feat_spectral_conf) / total_weight
        total_spatial_conf = (self.pred_weight * pred_spatial_conf + self.feat_weight * feat_spatial_conf) / total_weight

        # 扩展为空间维度置信度图（用于权重加权）
        total_spectral_conf = total_spectral_conf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)  # [B,1,H,W]
        total_spatial_conf = total_spatial_conf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)  # [B,1,H,W]

        # -------------------------- 第四部分：权重融合--------------------------
        fused_weight = spectral_weight * total_spectral_conf + spatial_weight * total_spatial_conf

        # 返回融合权重+三种置信度（可选，方便调试）
        return fused_weight, total_spectral_conf.squeeze(-1).squeeze(-1), total_spatial_conf.squeeze(-1).squeeze(-1)


# 主模型（无需额外修改，仅传递特征用于置信度计算）
class octfusion_multi_adder_1(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, weight_channels=64,
                 fusion_hidden=[128, 64], dropout=0.5):
        super(octfusion_multi_adder_1, self).__init__()
        self.weight_channels = weight_channels

        # 1. HSI基础特征提取
        self.Feature_HSI = Feature_HSI(in_channels_1, weight_channels, kernel_size=1, stride=1, padding=0)

        # 2. 权重生成
        self.Spectral_Weight_HSI = Spectral_Weight(
            in_channels=in_channels_1,
            out_channels=weight_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.Spatial_Weight_LIDAR = Spatial_Weight(
            in_channels=in_channels_2,
            out_channels=weight_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 3. 双重置信度融合模块（修改后）
        self.Weight_Fusion = Spectral_Spatial_Fusion(
            weight_channels=weight_channels,
            n_classes=out_channels,
            hidden=fusion_hidden,
            dropout=dropout
        )

        # 4. 分类头
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(weight_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        B = x1.shape[0]

        # HSI基础特征（用于特征层面置信度）
        hsi_base_feat = self.Feature_HSI(x1)  # [B, C, H, W]

        # 生成权重 + LIDAR特征（用于特征层面置信度）
        spectral_weight = self.Spectral_Weight_HSI(x1)  # [B, C, H, W]
        lidar_feat = self.Spatial_Weight_LIDAR(x2)  # [B, C, H, W]
        spatial_weight = lidar_feat  # LIDAR特征直接作为空间权重

        # 双重置信度融合
        fused_weight, spectral_conf, spatial_conf = self.Weight_Fusion(
            spectral_weight, spatial_weight, hsi_feat=hsi_base_feat, lidar_feat=lidar_feat
        )

        # 特征加权+分类
        final_feat = hsi_base_feat * fused_weight
        out = self.GAP(final_feat).view(B, -1)
        out = self.dropout(out)
        out = self.fc(out)
        out = F.softmax(out, dim=1)

        return out  # 训练时返回分类输出，若需调试可返回 out, spectral_conf, spatial_conf