from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import albumentations as A
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from vending.face_detection.blazeface import BlazeFace
from vending.face_detection.training.data_loader import OpenImagesV7Dataset
from vending.face_detection.training.specifications import FaceModelParameters
from vending.face_detection.utils import MultiBoxLoss, od_collate_fn


def train_model(
    net: torch.nn.Module,
    dataloaders_dict: dict[str, Any],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    model_params: FaceModelParameters,
    device: torch.device,
) -> None:
    """Train the model.

    Args:
        net (torch.nn.Module): The neural network model.
        dataloaders_dict (dict): Dictionary containing training and validation dataloaders.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        model_params (ModelParameters): Model parameters.
        device (torch.device): Device to run the model on.
    """
    net = net.to(device)

    for epoch in range(model_params.epochs):
        # Train
        running_loss = 0.0
        running_loc_loss = 0.0
        running_class_loss = 0.0
        for images, targets in tqdm.tqdm(dataloaders_dict["train"]):
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            loss_l, loss_c = criterion(outputs, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loc_loss += loss_l.item()
            running_class_loss += loss_c.item()

        # Eval
        net.eval()
        val_loss = 0.0
        val_loc_loss = 0.0
        val_class_loss = 0.0
        with torch.no_grad():
            for images, targets in dataloaders_dict["valid"]:
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]
                outputs = net(images)
                loss_l, loss_c = criterion(outputs, targets)
                loss = loss_l + loss_c
                val_loss += loss.item()
                val_loc_loss += loss_l.item()
                val_class_loss += loss_c.item()

        train_loss = running_loss / len(dataloaders_dict["train"])
        train_loc_loss = running_loc_loss / len(dataloaders_dict["train"])
        train_class_loss = running_class_loss / len(dataloaders_dict["train"])
        val_loss = val_loss / len(dataloaders_dict["valid"])
        print(
            f"[{epoch + 1}] train loss: {train_loss:.3f} | validation loss: {val_loss:.3f}"
        )
        print(
            f"train loc loss: {train_loc_loss:.3f} | train class loss: {train_class_loss:.3f}"
        )
        scheduler.step(val_loss)
        # Save model
        torch.save(net.state_dict(), model_params.model_path)


if __name__ == "__main__":
    dataset_directory = str(Path.cwd() / "datasets" / "face_detection")

    parser = ArgumentParser(description="Training blaze face model")

    parser.add_argument(
        "--dataset", help="the dataset path", type=str, default=dataset_directory
    )
    parser.add_argument("--batch_size", help="the batch size", type=int, default=256)
    parser.add_argument("--epochs", help="the number of epochs", type=int, default=10)
    parser.add_argument(
        "--lr", help="the initial learning rate", type=float, default=0.001
    )
    parser.add_argument(
        "--detection_threshold", help="the detection threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--img_size", help="the resized image size", type=int, default=128
    )
    parser.add_argument(
        "--channels", help="BlazeFace input channels", type=int, default=32
    )
    parser.add_argument(
        "--original",
        help="Use original architecture",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    model_params = FaceModelParameters(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        image_size=args.img_size,
        detection_threshold=args.detection_threshold,
        blazeface_channels=args.channels,
    )

    augment = A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2),
            A.HorizontalFlip(p=0.5),
            A.RandomCropFromBorders(
                crop_left=0.05,
                crop_right=0.05,
                crop_top=0.05,
                crop_bottom=0.05,
                p=0.9,
            ),
            A.Affine(
                rotate=(-30, 30),
                scale=(0.8, 1.1),
                keep_ratio=True,
                translate_percent=(-0.05, 0.05),
                p=0.9,
            ),
        ],
        bbox_params=A.BboxParams(format="albumentations"),
    )

    model_params.augmentation = augment.to_dict()

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Data Loaders
    train_dataset = OpenImagesV7Dataset(
        args.dataset + "/labels/train/",
        image_size=model_params.image_size,
        augment=augment,
    )
    valid_dataset = OpenImagesV7Dataset(
        args.dataset + "/labels/val/",
        image_size=model_params.image_size,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=od_collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=od_collate_fn,
    )

    dataloaders_dict = {"train": train_dataloader, "valid": valid_dataloader}

    if model_params.image_size == 256:
        model = BlazeFace(use_back_model=True)
    else:
        model = BlazeFace()

    if not Path("anchors.npy").exists():
        raise Exception("anchors file does not exist.")

    model.load_anchors("anchors.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = MultiBoxLoss(
        jaccard_thresh=0.5, neg_pos=3, device=device, dbox_list=model.dbox_list
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=model_params.lr_scheduler_patience
    )

    # Train the model
    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        scheduler,
        model_params,
        device=device,
    )
