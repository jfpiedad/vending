import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BlazeBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ) -> None:
        super().__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            conv_input = F.pad(input_tensor, (0, 2, 0, 2), mode="constant", value=0)
            input_tensor = self.max_pool(input_tensor)
        else:
            conv_input = input_tensor

        if self.channel_pad > 0:
            input_tensor = F.pad(
                input_tensor,
                (0, 0, 0, 0, 0, self.channel_pad),
                mode="constant",
                value=0,
            )

        return self.activation(self.convs(conv_input) + input_tensor)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        """
        Initializes the FinalBlazeBlock.

        Args:
            channels (int): Number of input and output channels.
            kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        """
        super(FinalBlazeBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=2,
                padding=0,
                groups=channels,
                bias=True,
            ),
            nn.BatchNorm2d(channels),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FinalBlazeBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the FinalBlazeBlock.
        """
        conv_input = F.pad(input_tensor, (0, 2, 0, 2), mode="constant", value=0)

        return self.activation(self.convs(conv_input))


class BlazeFace(nn.Module):
    def __init__(self, use_back_model: bool = False) -> None:
        """Initialize the BlazeFace model.

        Args:
            use_back_model (bool, optional): Whether to use the back model. Defaults to False.
        """
        super().__init__()

        self.num_anchors: int = 896
        self.use_back_model: bool = use_back_model

        scale: float = 256.0 if use_back_model else 128.0
        self.x_scale: float = scale
        self.y_scale: float = scale
        self.h_scale: float = scale
        self.w_scale: float = scale
        self.min_score_thresh: float = 0.65 if use_back_model else 0.75

        self._define_layers()

    def _define_layers(self) -> None:
        """Define the layers of the BlazeFace model."""
        if self.use_back_model:
            self.backbone: nn.Sequential = nn.Sequential(
                nn.Conv2d(3, 24, 5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),
                *(BlazeBlock(24, 24) for _ in range(7)),
                BlazeBlock(24, 24, stride=2),
                *(BlazeBlock(24, 24) for _ in range(7)),
                BlazeBlock(24, 48, stride=2),
                *(BlazeBlock(48, 48) for _ in range(7)),
                BlazeBlock(48, 96, stride=2),
                *(BlazeBlock(96, 96) for _ in range(7)),
            )
            self.final: FinalBlazeBlock = FinalBlazeBlock(96)
            self.classifier_8: nn.Conv2d = nn.Conv2d(96, 6, 1, bias=True)
            self.classifier_16: nn.Conv2d = nn.Conv2d(96, 18, 1, bias=True)
            self.regressor_8: nn.Conv2d = nn.Conv2d(96, 8, 1, bias=True)
            self.regressor_16: nn.Conv2d = nn.Conv2d(96, 24, 1, bias=True)
        else:
            self.backbone1: nn.Sequential = nn.Sequential(
                nn.Conv2d(3, 24, 5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )
            self.backbone2: nn.Sequential = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                *(BlazeBlock(96, 96) for _ in range(4)),
            )
            self.classifier_8: nn.Conv2d = nn.Conv2d(88, 6, 1, bias=True)
            self.classifier_16: nn.Conv2d = nn.Conv2d(96, 18, 1, bias=True)
            self.regressor_8: nn.Conv2d = nn.Conv2d(88, 8, 1, bias=True)
            self.regressor_16: nn.Conv2d = nn.Conv2d(96, 24, 1, bias=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BlazeFace model."""
        input_tensor = F.pad(input_tensor, (1, 2, 1, 2), mode="constant", value=0)
        batch_size: int = input_tensor.shape[0]

        if self.use_back_model:
            processed_tensor = self.backbone(input_tensor)
            features = self.final(processed_tensor)
        else:
            processed_tensor = self.backbone1(input_tensor)
            features = self.backbone2(processed_tensor)

        classification_8 = (
            self.classifier_8(processed_tensor)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, -1, 3)
        )

        classification_16 = (
            self.classifier_16(features).permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        )

        classification = torch.cat((classification_8, classification_16), dim=1)

        regression_8 = (
            self.regressor_8(processed_tensor)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, -1, 4)
        )
        regression_16 = (
            self.regressor_16(features).permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        )
        regression = torch.cat((regression_8, regression_16), dim=1)

        return torch.cat([regression, classification], dim=2)

    def _device(self) -> torch.device:
        """Get the device (CPU or GPU) the model is using."""
        return self.classifier_8.weight.device

    def load_weights(self, file_path: str) -> None:
        """Load model weights from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def load_anchors(self, file_path: str) -> None:
        """Load anchor points from a file."""
        self.anchors: torch.Tensor = torch.tensor(
            np.load(file_path),
            dtype=torch.float32,
            device=self._device(),
        )
        self.dbox_list: torch.Tensor = self.anchors

        assert self.anchors.ndim == 2
        assert self.anchors.shape == (self.num_anchors, 4)
