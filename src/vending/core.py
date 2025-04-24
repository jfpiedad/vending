import math
import time

import cv2
import mediapipe
import numpy as np
from fastapi import Request
from mediapipe.tasks.python.components.containers.detections import DetectionResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

from vending.age_estimation.detect import AgeEstimator
from vending.constants import VENDING_DRINKS
from vending.detection_data import FinalDetectionResults
from vending.enums import AgeGroup
from vending.face_detection.detect import initialize_face_detector
from vending.services import get_current_weather
from vending.state import VendingState

MARGIN = 50  # pixels
ROW_SIZE = 50  # pixels
FONT_SIZE = 5
FONT_THICKNESS = 3
TEXT_COLOR = (255, 0, 0)  # red

AGE_DETECTOR = AgeEstimator()


def process_results(ages: list[int]) -> None:
    age = int(sum(ages) / len(ages))

    if age < 13:
        age_group = AgeGroup.CHILD
    elif 13 <= age <= 19:
        age_group = AgeGroup.TEEN
    elif 20 <= age <= 59:
        age_group = AgeGroup.ADULT
    else:
        age_group = AgeGroup.SENIOR

    weather = get_current_weather()

    FinalDetectionResults.age = age
    FinalDetectionResults.age_group = age_group
    FinalDetectionResults.weather = weather

    suggested_drinks = VENDING_DRINKS[age_group][weather]
    suggested_drinks = suggested_drinks.split(", ")

    for index, _ in enumerate(suggested_drinks):
        suggested_drinks[index] = suggested_drinks[index].title()

    FinalDetectionResults.suggested_drinks = suggested_drinks


def livestream_detection_callback(
    detection_result: DetectionResult, output_image: mediapipe.Image, timestamp_ms: int
) -> None:
    if detection_result.detections:
        if VendingState.number_of_frames >= 10:
            VendingState.currently_vending = True
            ages = []

            for result, image, _ in VendingState.detection_data:
                bounding_box = result.detections[0].bounding_box
                image_copy = np.copy(image.numpy_view())

                ages.append(AGE_DETECTOR.predict_frame(bounding_box, image_copy))

            process_results(ages=ages)

            # Processing frames are done and results are ready.
            # User can now choose a drink from the vending machine
            VendingState.ready_choosing.set()

            # Wait until the user has chosen a drink or decided to cancel
            while (
                VendingState.ready_choosing.is_set() and not VendingState.is_terminated
            ):
                time.sleep(0.5)

            VendingState.number_of_frames = 0
            VendingState.clear_detection_data()
            VendingState.currently_vending = False
        else:
            if VendingState.number_of_frames < 10:
                VendingState.add_frame(detection_result, output_image, timestamp_ms)
                VendingState.number_of_frames += 1


async def run_detection_livestream(request: Request):
    if VendingState.started:
        return

    video_feed = cv2.VideoCapture(0)

    face_detector = initialize_face_detector(
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        callback=livestream_detection_callback,
    )

    print("Vending Machine is turned on.")
    try:
        with face_detector as detector:
            while True:
                if await request.is_disconnected():
                    break

                success, frame = video_feed.read()

                if not success:
                    continue

                timestamp_ms = video_feed.get(cv2.CAP_PROP_POS_MSEC)

                image = mediapipe.Image(
                    image_format=mediapipe.ImageFormat.SRGB, data=frame
                )

                if VendingState.started and not VendingState.currently_vending:
                    detector.detect_async(image, timestamp_ms=int(timestamp_ms))

                ret, buffer = cv2.imencode(".jpg", frame)

                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
    finally:
        print("Vending Machine is turned off.")
        VendingState.is_terminated = True

        # Trigger a flag when the browser tab is closed to signal that vending is turned off
        # and not wait for a user anymore which will properly close the websocket connection.
        VendingState.ready_choosing.set()

        video_feed.release()
        cv2.destroyAllWindows()


def run_detection_image() -> None:
    image_feed = cv2.VideoCapture("random_samples/child.jpg")

    face_detector = initialize_face_detector(VisionTaskRunningMode.IMAGE)
    age_detector = AgeEstimator()

    with face_detector as detector:
        while True:
            success, frame = image_feed.read()

            if not success:
                break

            image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame)

            detection_result = detector.detect(image)

            if not detection_result.detections:
                continue

            image_copy = np.copy(image.numpy_view())

            bounding_box = detection_result.detections[0].bounding_box

            age_detector.predict_frame(bounding_box, frame)

            annotated_image = visualize(image_copy, detection_result)

            cv2.imshow("test", annotated_image)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    image_feed.release()
    cv2.destroyAllWindows()


def run_detection_video() -> None:
    video_feed = cv2.VideoCapture("random_samples/testvideo2.mp4")
    fps = int(video_feed.get(cv2.CAP_PROP_FPS))
    timestamp = 0

    face_detector = initialize_face_detector(VisionTaskRunningMode.VIDEO)
    age_detector = AgeEstimator()

    with face_detector as detector:
        while True:
            success, frame = video_feed.read()

            if not success:
                break

            image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame)

            detection_result = detector.detect_for_video(image, timestamp_ms=timestamp)
            timestamp += int(1000 / fps)

            if not detection_result.detections:
                continue

            image_copy = np.copy(image.numpy_view())

            bounding_box = detection_result.detections[0].bounding_box

            age_detector.predict_frame(bounding_box, image_copy)

            annotated_image = visualize(image_copy, detection_result)

            cv2.imshow("Test", annotated_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_feed.release()
    cv2.destroyAllWindows()


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> tuple[int, int] | None:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height
            )
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = "" if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (bbox.origin_x, bbox.origin_y)
        cv2.putText(
            annotated_image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return annotated_image


if __name__ == "__main__":
    run_detection_livestream()
