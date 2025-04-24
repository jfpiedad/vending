from dataclasses import dataclass


@dataclass
class AgeModelParameters:
    """Class with all the model parameters"""

    batch_size: int = 128  # 128
    image_size: int = 64  # 64
    epochs: int = 100  # 100
    validation_split: float = 0.2
    model_path: str = "weights/agenet.pt"
