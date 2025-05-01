from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import cv2
import mediapipe
import numpy as np
from mediapipe.tasks.python.components.containers.detections import DetectionResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

from vending.age_estimation.detect import AgeEstimator
from vending.core.utils import annotate_image_with_bounding_box
from vending.face_detection.detect import initialize_face_detector

AGE_DETECTOR = AgeEstimator()


executor = ThreadPoolExecutor()
_process: Future = None
annotated_image = None


def run_detection(
    callback: Callable[[DetectionResult, mediapipe.Image, int], None],
) -> None:
    video_feed = cv2.VideoCapture(0)

    face_detector = initialize_face_detector(
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        callback=callback,
    )

    with face_detector as detector:
        while True:
            success, frame = video_feed.read()

            if not success:
                continue

            timestamp_ms = video_feed.get(cv2.CAP_PROP_POS_MSEC)

            image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame)

            detector.detect_async(image, timestamp_ms=int(timestamp_ms))

            if annotated_image is not None:
                cv2.imshow("Face Detection", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    executor.shutdown()
    video_feed.release()
    cv2.destroyAllWindows()


def face_detection_callback(
    detection_result: DetectionResult, output_image: mediapipe.Image, timestamp_ms: int
) -> None:
    global annotated_image

    if detection_result.detections:
        bounding_box = detection_result.detections[0].bounding_box
        print(
            f"\r\033[KFace detected, bounding_box = {bounding_box}", end="", flush=True
        )
        annotated_image = annotate_image_with_bounding_box(
            output_image.numpy_view(), detection_result
        )
    else:
        print("\r\033[KNo face detected.", end="", flush=True)
        annotated_image = output_image.numpy_view()


def age_estimation_callback(
    detection_result: DetectionResult, output_image: mediapipe.Image, timestamp_ms: int
) -> None:
    global annotated_image
    global _process

    if detection_result.detections:
        bounding_box = detection_result.detections[0].bounding_box

        annotated_image = annotate_image_with_bounding_box(
            output_image.numpy_view(), detection_result
        )
        image_copy = np.copy(output_image.numpy_view())

        if _process is None or _process.done():
            _process = executor.submit(
                AGE_DETECTOR.predict_no_weather, bounding_box, image_copy
            )
    else:
        print("\r\033[KNo face detected.", end="", flush=True)
        annotated_image = output_image.numpy_view()


def weather_callback(
    detection_result: DetectionResult, output_image: mediapipe.Image, timestamp_ms: int
) -> None:
    global annotated_image
    global _process

    if detection_result.detections:
        bounding_box = detection_result.detections[0].bounding_box

        annotated_image = annotate_image_with_bounding_box(
            output_image.numpy_view(), detection_result
        )
        image_copy = np.copy(output_image.numpy_view())

        if _process is None or _process.done():
            _process = executor.submit(
                AGE_DETECTOR.predict_with_weather, bounding_box, image_copy
            )
    else:
        print("\r\033[KNo face detected.", end="", flush=True)
        annotated_image = output_image.numpy_view()


if __name__ == "__main__":
    callback_dict = {
        1: face_detection_callback,
        2: age_estimation_callback,
        3: weather_callback,
    }

    parser = ArgumentParser(description="Testing individual feature of the system.")

    parser.add_argument(
        "--flag",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="An integer flag, must be (1, 2, or 3).",
    )

    args = parser.parse_args()

    run_detection(callback=callback_dict[args.flag])
