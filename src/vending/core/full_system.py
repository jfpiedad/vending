import time
from typing import AsyncGenerator

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
from vending.core.utils import annotate_image_with_bounding_box
from vending.detection_data import FinalDetectionResults
from vending.enums import AgeGroup
from vending.face_detection.detect import initialize_face_detector
from vending.services import get_current_weather
from vending.state import VendingState

AGE_DETECTOR = AgeEstimator()

annotated_image = None


async def run_detection_livestream(request: Request) -> AsyncGenerator[bytes, None]:
    global annotated_image

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
                else:
                    annotated_image = frame

                if annotated_image is not None:
                    ret, buffer = cv2.imencode(".jpg", annotated_image)

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


def livestream_detection_callback(
    detection_result: DetectionResult, output_image: mediapipe.Image, timestamp_ms: int
) -> None:
    global annotated_image

    if detection_result.detections:
        annotated_image = annotate_image_with_bounding_box(
            output_image.numpy_view(), detection_result
        )

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
    else:
        annotated_image = output_image.numpy_view()


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
