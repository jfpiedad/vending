import asyncio

import mediapipe
from mediapipe.tasks.python.components.containers.detections import DetectionResult


class VendingState:
    started: bool = False
    ready_choosing: asyncio.Event = asyncio.Event()
    currently_vending: bool = False
    number_of_frames: int = 0
    detection_data: list[tuple[DetectionResult, mediapipe.Image, int]] = []
    is_terminated: bool = False

    @classmethod
    def clear_detection_data(cls) -> None:
        cls.detection_data.clear()

    @classmethod
    def add_frame(
        cls, result: DetectionResult, image: mediapipe.Image, timestamp_ms: int
    ) -> None:
        cls.detection_data.append((result, image, timestamp_ms))
