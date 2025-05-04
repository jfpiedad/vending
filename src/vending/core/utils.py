import cv2
import mediapipe
import numpy as np
from mediapipe.tasks.python.components.containers.detections import DetectionResult

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def sort_detection_results(detection_result: DetectionResult) -> None:
    """Sorts the detection results in descending order based on the area of the bounding box."""

    detection_result.detections.sort(
        key=lambda detection: detection.bounding_box.width
        * detection.bounding_box.height,
        reverse=True,
    )


def annotate_image_with_bounding_box(
    image: np.ndarray | mediapipe.Image, detection_result: DetectionResult
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        image = image.numpy_view()

    image_copy = image.copy()

    for index, detection in enumerate(detection_result.detections):
        bounding_box = detection.bounding_box

        start_point = (bounding_box.origin_x, bounding_box.origin_y)
        end_point = (
            bounding_box.origin_x + bounding_box.width,
            bounding_box.origin_y + bounding_box.height,
        )

        if index == 0:
            box_color = GREEN
        else:
            box_color = YELLOW

        cv2.rectangle(image_copy, start_point, end_point, color=box_color, thickness=2)

    return image_copy
