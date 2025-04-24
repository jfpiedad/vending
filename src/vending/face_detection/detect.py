from typing import Callable

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.face_detector import (
    FaceDetector,
    FaceDetectorOptions,
)


def initialize_face_detector(
    running_mode: VisionTaskRunningMode, callback: Callable | None = None
) -> FaceDetector:
    model_asset_path = "weights/blazeface.tflite"

    base_options = BaseOptions(model_asset_path=model_asset_path)
    options = FaceDetectorOptions(base_options=base_options, running_mode=running_mode)

    if callback is not None:
        options.result_callback = callback

    detector = FaceDetector.create_from_options(options=options)

    return detector
