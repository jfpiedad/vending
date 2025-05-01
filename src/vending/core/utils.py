import cv2
import mediapipe
import numpy as np
from mediapipe.tasks.python.components.containers.detections import DetectionResult

GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (255, 255, 0)


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
            box_color = GREEN_COLOR
        else:
            box_color = YELLOW_COLOR

        cv2.rectangle(image_copy, start_point, end_point, color=box_color, thickness=2)

    return image_copy
