import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms as T


class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.filename_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.filename_list)

    def age_to_class(self, age):
        range_ages = [0, 4, 9, 15, 25, 35, 45, 60, 75]
        if age > max(range_ages):
            return len(range_ages) - 1
        for i in range(len(range_ages) - 1):
            if range_ages[i] <= age <= range_ages[i + 1]:
                return i

    def __getitem__(self, idx):
        filename = self.filename_list[idx]

        info = filename.split("_")
        age = int(info[0])
        age_label = self.age_to_class(age)
        gender = int(info[1].removesuffix(".jpg"))

        filename = os.path.join(self.root_dir, filename)

        image = Image.open(filename)

        if self.transform:
            image = self.transform(image)

        return image, gender, age_label, age


def get_dataloader(
    root_dir, image_size=64, batch_size=128, shuffle=True, num_workers=1
) -> DataLoader:
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    train_transform = T.Compose(
        [
            T.Resize(image_size),
            T.RandomHorizontalFlip(0.2),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = UTKFaceDataset(root_dir, transform=train_transform)
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return trainloader


def split_dataloader(train_data, validation_split=0.2) -> tuple[DataLoader, DataLoader]:
    train_ratio = 1 - validation_split
    train_size = int(train_ratio * len(train_data.dataset))

    indices = list(range(len(train_data.dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    dataset = train_data.dataset
    batch_size = train_data.batch_size
    num_workers = train_data.num_workers

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_data = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_data = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_data, val_data
