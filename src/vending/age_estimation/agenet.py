import torch
from torch import nn
from torch.nn import functional as F


class GenderClassificationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(2048, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Pytorch forward

        Args:
            input_tensor: input image (224x224)

        Returns: class logits

        """
        input_tensor = F.relu(self.conv_1_1(input_tensor))
        input_tensor = F.relu(self.conv_1_2(input_tensor))
        input_tensor = F.max_pool2d(input_tensor, 2, 2)
        input_tensor = F.relu(self.conv_2_1(input_tensor))
        input_tensor = F.relu(self.conv_2_2(input_tensor))
        input_tensor = F.max_pool2d(input_tensor, 2, 2)
        input_tensor = F.relu(self.conv_3_1(input_tensor))
        input_tensor = F.relu(self.conv_3_2(input_tensor))
        input_tensor = F.relu(self.conv_3_3(input_tensor))
        input_tensor = F.max_pool2d(input_tensor, 2, 2)
        input_tensor = F.relu(self.conv_4_1(input_tensor))
        input_tensor = F.relu(self.conv_4_2(input_tensor))
        input_tensor = F.relu(self.conv_4_3(input_tensor))
        input_tensor = F.max_pool2d(input_tensor, 2, 2)
        input_tensor = F.relu(self.conv_5_1(input_tensor))
        input_tensor = F.relu(self.conv_5_2(input_tensor))
        input_tensor = F.relu(self.conv_5_3(input_tensor))
        input_tensor = F.max_pool2d(input_tensor, 2, 2)
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = F.relu(self.fc6(input_tensor))
        input_tensor = F.dropout(input_tensor, 0.5, self.training)
        input_tensor = F.relu(self.fc7(input_tensor))
        input_tensor = F.dropout(input_tensor, 0.5, self.training)
        return F.sigmoid(self.fc8(input_tensor))


class AgeRangeModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
    ) -> None:
        super().__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

        self.adap = nn.AdaptiveAvgPool2d((2, 2))

        self.out_age = nn.Sequential(nn.Linear(2048, num_classes))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = input_tensor.shape[0]
        input_tensor = self.Conv1(input_tensor)
        input_tensor = self.Conv2(input_tensor)
        input_tensor = self.Conv3(input_tensor)

        input_tensor = self.adap(input_tensor)

        input_tensor = input_tensor.view(batch_size, -1)

        input_tensor = self.out_age(input_tensor)

        return input_tensor


class AgeEstimationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(9, 64)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

        self.adap = nn.AdaptiveAvgPool2d((2, 2))

        self.out_age = nn.Sequential(nn.Linear(2048 + 64, 1), nn.ReLU())

    def forward(
        self, image_tensor: torch.Tensor, metadata_tensor: torch.Tensor
    ) -> torch.Tensor:
        batch_size = image_tensor.shape[0]
        image_tensor = self.Conv1(image_tensor)
        image_tensor = self.Conv2(image_tensor)
        image_tensor = self.Conv3(image_tensor)

        image_tensor = self.adap(image_tensor)

        image_tensor = image_tensor.view(batch_size, -1)

        metadata_tensor = self.embedding_layer(metadata_tensor)

        combined_tensor = torch.cat([image_tensor, metadata_tensor], dim=1)

        output = self.out_age(combined_tensor)

        return output


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.gender_model = GenderClassificationModel()

        self.age_range_model = AgeRangeModel()

        self.age_estimation_model = AgeEstimationModel()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor[None, ...]

        predicted_genders = self.gender_model(input_tensor)

        age_ranges = self.age_range_model(input_tensor)

        predicted_age_range_indices = torch.argmax(age_ranges, dim=1).view(-1)

        estimated_ages = self.age_estimation_model(
            input_tensor, predicted_age_range_indices
        )

        return predicted_genders, estimated_ages
