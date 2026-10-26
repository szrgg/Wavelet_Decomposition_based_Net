
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Wavelet_Decomposition_Net"]


class ConvBlock(nn.Module):
    """Two 3x3 convolution, batch-normalization and ReLU layers."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    """Upsample by two, then apply convolution, normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class DSAG(nn.Module):

    def __init__(self, decoder_channels, encoder_channels, hidden_channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, hidden_channels // reduction)

        self.decoder_projection = nn.Sequential(
            nn.Conv2d(decoder_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.encoder_projection = nn.Sequential(
            nn.Conv2d(encoder_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.activation = nn.ReLU(inplace=True)
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.dynamic_channel_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, hidden_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_weight = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, decoder_feature, encoder_feature):
        merged = self.activation(
            self.decoder_projection(decoder_feature)
            + self.encoder_projection(encoder_feature)
        )
        separated = self.separable_conv(merged)
        separated = separated * self.dynamic_channel_weight(separated)
        return encoder_feature * self.spatial_weight(separated)


class AdaptiveFusion(nn.Module):


    def __init__(self, decoder_channels, hidden_channels=16):
        super().__init__()
        self.alpha = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(decoder_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, decoder_feature, raw_feature, attended_feature):
        alpha = self.alpha(decoder_feature)
        return alpha * raw_feature + (1.0 - alpha) * attended_feature


class ChannelOrientationModulationBlock(nn.Module):
    """Fixed, non-trainable channel weights for each wavelet orientation."""

    def __init__(self, channels, direction):
        super().__init__()
        if direction == "horizontal":
            initial_weight = torch.linspace(0.0, 1.0, channels)
        elif direction == "vertical":
            initial_weight = torch.linspace(1.0, 0.0, channels)
        else:
            initial_weight = torch.full((channels,), 0.7)

        self.register_buffer(
            "direction_weight", initial_weight.view(1, channels, 1, 1)
        )

    def forward(self, x):
        return x * self.direction_weight


class SpatialAttentionBlock(nn.Module):


    def __init__(self):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.max(x, dim=1, keepdim=True).values
        return x * self.spatial_attention(torch.cat((average, maximum), dim=1))


class OrientationAwareFeatureExtraction(nn.Module):


    def __init__(self, channels):
        super().__init__()
        self.hl_comb = ChannelOrientationModulationBlock(
            channels, direction="vertical"
        )
        self.lh_comb = ChannelOrientationModulationBlock(
            channels, direction="horizontal"
        )
        self.hh_comb = ChannelOrientationModulationBlock(
            channels, direction="diagonal"
        )
        self.hl_sab = SpatialAttentionBlock()
        self.lh_sab = SpatialAttentionBlock()
        self.hh_sab = SpatialAttentionBlock()

    def forward(self, ll, hl, lh, hh):
        return (
            ll,
            self.hl_sab(self.hl_comb(hl)),
            self.lh_sab(self.lh_comb(lh)),
            self.hh_sab(self.hh_comb(hh)),
        )


class HighFrequencyAttention(nn.Module):


    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(3 * channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, hl, lh, hh):
        weights = self.attention(torch.cat((hl, lh, hh), dim=1))
        return (
            hl * weights[:, 0:1]
            + lh * weights[:, 1:2]
            + hh * weights[:, 2:3]
        )


class HaarWaveletDecomposition(nn.Module):


    def __init__(self):
        super().__init__()
        filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("filters", filters.unsqueeze(1))

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        filters = self.filters.repeat(channels, 1, 1, 1)
        bands = F.conv2d(x, filters, stride=2, groups=channels)
        bands = bands.reshape(
            batch_size,
            channels,
            4,
            bands.shape[-2],
            bands.shape[-1],
        )
        return bands[:, :, 0], bands[:, :, 1], bands[:, :, 2], bands[:, :, 3]


class FrequencyAwarePyramidFusion(nn.Module):


    def __init__(self, channels, levels=3):
        super().__init__()
        self.pyramid_convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=2**level,
                        dilation=2**level,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for level in range(levels)
            ]
        )
        self.high_frequency_attention = HighFrequencyAttention(channels)
        self.pyramid_weights = nn.Sequential(
            nn.Conv2d(channels * (levels + 1), channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, levels + 1, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, ll, hl, lh, hh):
        target_size = ll.shape[-2:]
        hl = F.adaptive_avg_pool2d(hl, target_size)
        lh = F.adaptive_avg_pool2d(lh, target_size)
        hh = F.adaptive_avg_pool2d(hh, target_size)
        high_frequency = self.high_frequency_attention(hl, lh, hh)

        # The three dilated 3x3 branches operate on LL in parallel, as shown
        # in the WFDM diagram. LL itself forms the fourth fusion branch.
        pyramid_features = [ll] + [
            convolution(ll) for convolution in self.pyramid_convolutions
        ]

        weights = self.pyramid_weights(torch.cat(pyramid_features, dim=1))
        low_frequency = sum(
            weight * feature
            for weight, feature in zip(weights.split(1, dim=1), pyramid_features)
        )
        return low_frequency + high_frequency


class WFDM(nn.Module):


    def __init__(self, channels):
        super().__init__()
        self.wavelet = HaarWaveletDecomposition()
        self.orientation_features = OrientationAwareFeatureExtraction(channels)
        self.fusion = FrequencyAwarePyramidFusion(channels)

    def forward(self, x):
        ll, hl, lh, hh = self.wavelet(x)
        ll, hl, lh, hh = self.orientation_features(ll, hl, lh, hh)
        return self.fusion(ll, hl, lh, hh)


class Wavelet_Decomposition_Net(nn.Module):


    def __init__(self, img_ch=3, output_ch=1):
        super().__init__()

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = ConvBlock(img_ch, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.bottleneck = ConvBlock(512, 1024)

        self.wfdm3 = WFDM(256)
        self.wfdm4 = WFDM(512)

        self.up5 = UpConv(1024, 512)
        self.dsag5 = DSAG(512, 512, 256)
        self.decoder5 = ConvBlock(1024, 512)

        self.up4 = UpConv(512, 256)
        self.dsag4 = DSAG(256, 256, 128)
        self.decoder4 = ConvBlock(512, 256)

        self.up3 = UpConv(256, 128)
        self.dsag3 = DSAG(128, 128, 64)
        self.decoder3 = ConvBlock(256, 128)

        self.up2 = UpConv(128, 64)
        self.dsag2 = DSAG(64, 64, 32)
        self.adaptive_fusion = AdaptiveFusion(decoder_channels=64)
        self.decoder2 = ConvBlock(128, 64)

        self.output = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.max_pool1(x1))
        x3 = self.encoder3(self.max_pool2(x2))
        x4 = self.encoder4(self.wfdm3(x3))
        x5 = self.bottleneck(self.wfdm4(x4))

        d5 = self.up5(x5)
        x4_attended = self.dsag5(d5, x4)
        d5 = self.decoder5(torch.cat((x4_attended, d5), dim=1))

        d4 = self.up4(d5)
        x3_attended = self.dsag4(d4, x3)
        d4 = self.decoder4(torch.cat((x3_attended, d4), dim=1))

        d3 = self.up3(d4)
        x2_attended = self.dsag3(d3, x2)
        d3 = self.decoder3(torch.cat((x2_attended, d3), dim=1))

        d2 = self.up2(d3)
        x1_attended = self.dsag2(d2, x1)
        x1_fused = self.adaptive_fusion(d2, x1, x1_attended)
        d2 = self.decoder2(torch.cat((x1_fused, d2), dim=1))

        return self.output(d2)
