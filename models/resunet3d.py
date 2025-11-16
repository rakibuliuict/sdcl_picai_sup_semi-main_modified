import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResUNet3D(nn.Module):
    """
    Simple residual U-Net in 3D.
    """
    def __init__(self, in_channels=3, num_classes=2, base_filters=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            ResBlock3d(in_channels, base_filters),
            ResBlock3d(base_filters, base_filters),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            ResBlock3d(base_filters, base_filters * 2),
            ResBlock3d(base_filters * 2, base_filters * 2),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(
            ResBlock3d(base_filters * 2, base_filters * 4),
            ResBlock3d(base_filters * 4, base_filters * 4),
        )
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            ResBlock3d(base_filters * 4, base_filters * 8),
            ResBlock3d(base_filters * 8, base_filters * 8),
        )

        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ResBlock3d(base_filters * 8, base_filters * 4),
            ResBlock3d(base_filters * 4, base_filters * 4),
        )

        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ResBlock3d(base_filters * 4, base_filters * 2),
            ResBlock3d(base_filters * 2, base_filters * 2),
        )

        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ResBlock3d(base_filters * 2, base_filters),
            ResBlock3d(base_filters, base_filters),
        )

        self.out_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.out_conv(d1)
        return out
