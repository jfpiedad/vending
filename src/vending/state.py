import asyncio
from collections import deque

import mediapipe
from mediapipe.tasks.python.components.containers.detections import DetectionResult


class VendingState:
    currently_on: bool = False
    currently_ordering: bool = False
    processing_frames: bool = False
    ready_to_vend: asyncio.Event = asyncio.Event()
    recent_frames: deque = deque([], maxlen=10)
    last_face_detected_timestamp: int = 0

    @classmethod
    def frames_count(cls) -> int:
        """Returns the length of the queue."""

        return len(cls.recent_frames)

    @classmethod
    def add_frame(
        cls, result: DetectionResult, image: mediapipe.Image, timestamp_ms: int
    ) -> None:
        """Add frame to the queue."""

        cls.recent_frames.append((result, image, timestamp_ms))

    @classmethod
    def clear_frames(cls) -> None:
        """Remove all frames from the queue."""

        cls.recent_frames.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset vending to initial state."""

        cls.currently_ordering = False
        cls.processing_frames = False

        cls.ready_to_vend.clear()
        cls.recent_frames.clear()
