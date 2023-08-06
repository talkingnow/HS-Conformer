#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""ConvolutionModule definition."""

from typing import Optional, Tuple

import torch
from torch import nn
# from typeguard import check_argument_types


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        # assert check_argument_types()
        super().__init__()

        self.pointwise_conv1 = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm2d(channels)

        self.pointwise_conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time)
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(0, 3, 1, 2)  # (#batch, channels, frq, time)

        if self.lorder > 0:
            x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            assert (x.size(2) > self.lorder)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (#batch, 2*channels, frq, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, frq, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))
        x = self.pointwise_conv2(x)

        return x.permute(0, 2, 3, 1)
    