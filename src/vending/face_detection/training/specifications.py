from dataclasses import dataclass, field


@dataclass
class FaceModelParameters:
    """Class with all the model parameters"""

    batch_size: int = 256
    lr: float = 0.001
    scheduler_type: str = "ReduceLROnPlateau"
    lr_scheduler_patience: int = 10
    epochs: int = 100
    classes: list[str] = field(default_factory=lambda: ["face"])
    image_size: int = 128
    detection_threshold: float = 0.5
    blazeface_channels: int = 32
    model_path: str = "weights/blazeface.pt"
    augmentation: dict | None = None
