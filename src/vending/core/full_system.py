import math
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

from vending.constants import AGE_DETECTOR, VENDING_DRINKS
from vending.core.utils import annotate_image_with_bounding_box, sort_detection_results
from vending.detection_data import FinalDetectionResults
from vending.enums import AgeGroup
from vending.face_detection.detect import initialize_face_detector
from vending.services import get_current_weather
from vending.state import VendingState

annotated_image = None


async def run_detection_livestream(request: Request) -> AsyncGenerator[bytes, None]:
    global annotated_image

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

                timestamp_ms = int(video_feed.get(cv2.CAP_PROP_POS_MSEC))

                image = mediapipe.Image(
                    image_format=mediapipe.ImageFormat.SRGB, data=frame
                )

                if not VendingState.processing_frames:
                    detector.detect_async(image, timestamp_ms=timestamp_ms)
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
        VendingState.currently_on = False

        # Trigger a flag when the browser tab is closed to signal that vending is turned off
        # and not wait for a user anymore which will properly close the websocket connection.
        VendingState.ready_to_vend.set()

        video_feed.release()
        cv2.destroyAllWindows()


def livestream_detection_callback(
    detection_result: DetectionResult, output_image: mediapipe.Image, timestamp_ms: int
) -> None:
    global annotated_image

    if detection_result.detections:
        VendingState.last_face_detected_timestamp = timestamp_ms

        # Sort detection results
        if len(detection_result.detections) > 1:
            sort_detection_results(detection_result=detection_result)

        annotated_image = annotate_image_with_bounding_box(
            output_image.numpy_view(), detection_result
        )

        if VendingState.currently_ordering and VendingState.frames_count() >= 10:
            VendingState.processing_frames = True
            ages = []

            for result, image, _ in VendingState.recent_frames:
                bounding_box = result.detections[0].bounding_box
                image_copy = np.copy(image.numpy_view())

                ages.append(AGE_DETECTOR.predict_frame(bounding_box, image_copy))

            process_results(ages=ages)

            # Processing frames are done and results are ready.
            # User can now choose a drink from the vending machine
            VendingState.ready_to_vend.set()

            # Wait until the user has chosen a drink or decided to cancel
            while VendingState.ready_to_vend.is_set() and VendingState.currently_on:
                time.sleep(0.5)

            VendingState.currently_ordering = False
            VendingState.processing_frames = False
        elif VendingState.currently_ordering and VendingState.frames_count < 10:
            VendingState.currently_ordering = False
            VendingState.ready_to_vend.set()
        else:
            VendingState.add_frame(detection_result, output_image, timestamp_ms)
    else:
        annotated_image = output_image.numpy_view()

        if VendingState.currently_ordering:
            VendingState.currently_ordering = False
            VendingState.ready_to_vend.set()


def process_results(ages: list[int]) -> None:
    ages.sort()

    length = len(ages)
    trim_ratio = 0.2

    trim_length = math.floor(length * trim_ratio)

    if 2 * trim_length >= length:
        raise ValueError(f"Trim ratio too large for a list of length {length}")

    trimmed_ages_list = ages[trim_length : length - trim_length]

    age = round(np.mean(trimmed_ages_list))

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
