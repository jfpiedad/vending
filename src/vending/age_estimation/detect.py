import argparse
import os

import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe
import numpy as np
import torch
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from PIL import Image
from torchvision import transforms as T

from vending.age_estimation.agenet import Model
from vending.face_detection.detect import initialize_face_detector
from vending.services import get_current_weather


class AgeEstimator:
    def __init__(
        self,
        face_size: int = 64,
        weights: str = "weights/agenet.pt",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        tpx: int = 500,
    ) -> None:
        self.thickness_per_pixels = tpx
        self.face_size = (
            (face_size, face_size) if isinstance(face_size, int) else face_size
        )
        self.device = device

        # Initialize models
        self.model = Model().to(self.device)
        self.model.eval()

        if weights:
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
            print(f"Weights loaded successfully from path: {weights}")
            print("=" * 60)

    def transform(self, image):
        """Transform input face image for the model."""
        return T.Compose(
            [
                T.Resize(self.face_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )(image)

    @staticmethod
    def plot_box_and_label(
        image, lw, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
    ):
        """Add a labeled bounding box to the image."""
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv.rectangle(image, p1, p2, color, thickness=lw, lineType=cv.LINE_AA)
        if label:
            tf = max(lw - 1, 1)
            w, h = cv.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = p1[1] - h - 3 >= 0
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv.rectangle(image, p1, p2, color, -1, cv.LINE_AA)
            cv.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv.LINE_AA,
            )

    def padding_face(self, box, padding=10):
        """Apply padding to bounding box."""
        return [box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding]

    def predict_no_weather(self, bounding_box, frame) -> int:
        age = self.predict_frame(bounding_box, frame)
        print(
            f"\r\033[KFace detected, bounding_box = {bounding_box} | {age} years old",
            end="",
            flush=True,
        )

    def predict_with_weather(self, bounding_box, frame) -> int:
        age = self.predict_frame(bounding_box, frame)
        weather = get_current_weather()
        print(
            f"\r\033[KFace detected, bounding_box = {bounding_box} | {age} years old | Weather: {weather}",
            end="",
            flush=True,
        )

    def predict_frame(self, bounding_box, frame) -> int:
        """Process a single video frame for real-time predictions."""
        bboxes = []

        x1 = bounding_box.origin_x
        y1 = bounding_box.origin_y
        x2 = x1 + bounding_box.width
        y2 = y1 + bounding_box.height

        bboxes.append([x1, y1, x2, y2])

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        ndarray_image = np.array(frame)

        if bboxes is None:
            return ndarray_image

        face_images = []
        for box in bboxes:
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            padding = max(ndarray_image.shape) * 5 / self.thickness_per_pixels
            padding = int(max(padding, 10))
            box = self.padding_face(box, padding)
            face = image.crop(box)
            transformed_face = self.transform(face)
            face_images.append(transformed_face)

        if not face_images:
            return ndarray_image

        face_images = torch.stack(face_images, dim=0)
        genders, ages = self.model(face_images)
        genders = torch.round(genders)
        ages = torch.round(ages).long()

        for i, box in enumerate(bboxes):
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            age = ages[i].item()

        return age

    def predict(self, img_path, min_prob=0.9):
        """Process an image file for predictions."""
        test_feed = cv.VideoCapture(img_path)

        bboxes = []

        face_detector = initialize_face_detector(VisionTaskRunningMode.IMAGE)

        with face_detector as detector:
            while True:
                success, frame = test_feed.read()

                if not success:
                    break

                temp_image = mediapipe.Image(
                    image_format=mediapipe.ImageFormat.SRGB, data=frame
                )

                detection_result = detector.detect(temp_image)

                if not detection_result.detections:
                    continue

                # image_copy = np.copy(temp_image.numpy_view())

                bounding_box = detection_result.detections[0].bounding_box

                x1 = bounding_box.origin_x
                y1 = bounding_box.origin_y
                x2 = x1 + bounding_box.width
                y2 = y1 + bounding_box.height

                bboxes.append([x1, y1, x2, y2])

        test_feed.release()
        cv.destroyAllWindows()

        image = Image.open(img_path)
        ndarray_image = np.array(image)

        image_shape = ndarray_image.shape

        if bboxes is None:
            return ndarray_image

        face_images = []
        for box in bboxes:
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            padding = max(image_shape) * 5 / self.thickness_per_pixels
            padding = int(max(padding, 10))
            box = self.padding_face(box, padding)
            face = image.crop(box)
            transformed_face = self.transform(face)
            face_images.append(transformed_face)

        if not face_images:
            return ndarray_image

        face_images = torch.stack(face_images, dim=0)
        genders, ages = self.model(face_images)
        genders = torch.round(genders)
        ages = torch.round(ages).long()

        for i, box in enumerate(bboxes):
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            thickness = max(image_shape) // 400
            thickness = int(max(np.ceil(thickness), 1))
            label = (
                f"{'Man' if genders[i] == 0 else 'Woman'}: {ages[i].item()} years old"
            )
            print(label)
            self.plot_box_and_label(
                ndarray_image, thickness, box, label, color=(255, 0, 0)
            )

        return ndarray_image


def main(
    image_path,
    weights="weights/agenet.pt",
    face_size=64,
    device="cpu",
    save_result=False,
):
    print(f"Processing image: {image_path}")
    model = AgeEstimator(weights=weights, face_size=face_size, device=device)
    predicted_image = model.predict_frame()

    if save_result:
        save_dir = os.path.join("runs", "predict")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "results.jpg")
        plt.imsave(save_path, predicted_image)
        print(f"Result saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to the input image.",
        default="random_samples/teen.jpg",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/agenet.pt",
        help="Path to the model weights.",
    )
    parser.add_argument(
        "--face-size", type=int, default=64, help="Face size for the model."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save the resulting image with annotations.",
    )
    args = parser.parse_args()

    main(**vars(args))
