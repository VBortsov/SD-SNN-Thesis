import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv1D(nn.Module):
    """Two Conv1d layers with normalization and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize layers and settings.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output component channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        return self.block(x)


class Down1D(nn.Module):
    """Downsampling block for the U-Net encoder."""
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize layers and settings.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output component channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        return self.block(x)


class Up1D(nn.Module):
    """Upsampling block that merges a skip connection."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """Initialize layers and settings.
        
        Args:
            in_channels: Number of input channels.
            skip_channels: Channels from the skip connection.
            out_channels: Number of output component channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x, skip):
        """Run the forward pass.
        
        Args:
            x: Input tensor.
            skip: Project value for this call.
        """
        x = self.up(x)

        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """1D U-Net for component reconstruction."""
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 32):
        """Initialize layers and settings.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output component channels.
            base_channels: Base channel width for the network.
        """
        super().__init__()

        self.inc = DoubleConv1D(in_channels, base_channels)
        self.down1 = Down1D(base_channels, base_channels * 2)
        self.down2 = Down1D(base_channels * 2, base_channels * 4)
        self.down3 = Down1D(base_channels * 4, base_channels * 8)

        self.bottleneck = Down1D(base_channels * 8, base_channels * 16)

        self.up1 = Up1D(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up1D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up1D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up1D(base_channels * 2, base_channels, base_channels)

        self.outc = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Run the forward pass.
        
        Args:
            x: Input tensor.
        """
        x1 = self.inc(x)     # [B, C, L]
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)