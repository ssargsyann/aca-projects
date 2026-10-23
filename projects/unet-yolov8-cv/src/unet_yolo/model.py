"""Notebook architecture, with configurable width and minimum-size validation."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two consecutive (Conv2d -> BatchNorm2d -> ReLU) blocks.

    This module is the core building block of the U-Net encoder and decoder.
    Padding is fixed to 1 so that the spatial resolution of the input is
    preserved through both convolutions.

    Attributes:
        double_conv: A sequential container holding the two convolutional
            sub-blocks.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initializes the DoubleConv block.

        Args:
            in_channels: Number of channels in the input feature map.
            out_channels: Number of channels produced by both convolutions.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the double convolution block.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``.

        Returns:
            Output tensor of shape ``(N, out_channels, H, W)``.
        """
        return self.double_conv(x)


class UNet(nn.Module):
    """A standard U-Net architecture for semantic segmentation.

    Implements the classic encoder-decoder design with skip connections
    described in Ronneberger et al. (2015), "U-Net: Convolutional Networks
    for Biomedical Image Segmentation".

    Attributes:
        inc: Initial DoubleConv block applied to the raw input.
        down1-down4: Encoder stages (MaxPool2d followed by DoubleConv).
        up1_transpose-up4_transpose: Decoder upsampling layers.
        up1_conv-up4_conv: Decoder DoubleConv blocks applied after
            concatenating the skip connection.
        outc: Final 1x1 convolution mapping to the desired number of
            output classes.
    """

    def __init__(
        self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64
    ) -> None:
        """Initializes the U-Net.

        Args:
            in_channels: Number of channels in the input image (e.g., 3 for RGB).
            num_classes: Number of output segmentation classes.
            base_channels: Width of the first encoder stage; 64 matches the original.
        """
        super().__init__()

        b = base_channels
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValueError("base_channels must be a positive integer")
        self.inc = DoubleConv(in_channels, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 4, b * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 8, b * 16))
        for stage, incoming, outgoing in [
            (1, b * 16, b * 8),
            (2, b * 8, b * 4),
            (3, b * 4, b * 2),
            (4, b * 2, b),
        ]:
            setattr(
                self,
                f"up{stage}_transpose",
                nn.ConvTranspose2d(incoming, outgoing, 2, 2),
            )
            setattr(self, f"up{stage}_conv", DoubleConv(outgoing * 2, outgoing))
        self.outc = nn.Conv2d(b, num_classes, 1)

    @staticmethod
    def _align(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Center-crops or zero-pads ``upsampled`` to match ``skip``'s H/W.

        Handles the off-by-one spatial mismatches that occur when the input
        image's height/width is not an exact multiple of 16, which would
        otherwise make ``torch.cat`` raise a shape-mismatch error.

        Args:
            upsampled: Decoder feature map produced by a transposed
                convolution, shape ``(N, C, H1, W1)``.
            skip: Corresponding encoder feature map, shape ``(N, C, H2, W2)``.

        Returns:
            ``upsampled`` resized (via center crop and/or pad) to shape
            ``(N, C, H2, W2)``.
        """
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)

        if diff_h != 0 or diff_w != 0:
            # Pad when the upsampled map is smaller, crop when it's larger.
            upsampled = F.pad(
                upsampled,
                [
                    diff_w // 2,
                    diff_w - diff_w // 2,
                    diff_h // 2,
                    diff_h - diff_h // 2,
                ],
            )
        return upsampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass through the U-Net.

        Args:
            x: Input tensor of shape ``(N, in_channels, H, W)``. Height and
                width must be at least 16; they are not required to be
                divisible by 16 thanks to the ``_align`` crop/pad step.

        Returns:
            Logits tensor of shape ``(N, num_classes, H, W)``.
        """
        if x.ndim != 4 or min(x.shape[-2:]) < 16:
            raise ValueError("Expected NCHW with height and width at least 16")
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck

        # Decoder, stage 1
        x_up1 = self._align(self.up1_transpose(x5), x4)
        x_out1 = self.up1_conv(torch.cat([x_up1, x4], dim=1))

        # Decoder, stage 2
        x_up2 = self._align(self.up2_transpose(x_out1), x3)
        x_out2 = self.up2_conv(torch.cat([x_up2, x3], dim=1))

        # Decoder, stage 3
        x_up3 = self._align(self.up3_transpose(x_out2), x2)
        x_out3 = self.up3_conv(torch.cat([x_up3, x2], dim=1))

        # Decoder, stage 4
        x_up4 = self._align(self.up4_transpose(x_out3), x1)
        x_out4 = self.up4_conv(torch.cat([x_up4, x1], dim=1))

        # Output projection to class logits
        logits = self.outc(x_out4)
        return logits
