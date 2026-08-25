import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = (
    Path(__file__).resolve().parent
    / "RIDER-Clean-Slices"
)
LOCAL_ROOT = (
    Path(__file__).resolve().parent
    / "RIDER-Lung-CT-Processed"
    / "RIDER-Lung-CT-Processed"
)
IMAGE_SIZE = 256
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 5
VAL_FRACTION = 0.2
SEED = 42
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pt"
FIGURE_PATH = Path(__file__).resolve().parent / "predictions.png"
TRAIN_REPEATS = 4
NEGATIVES_PER_SCAN = 3


def find_pairs(root):
    records = []
    image_dir = root / "images"
    mask_dir = root / "aligned_masks"
    for image_path in sorted(image_dir.glob("RIDER-*.jpg")):
        patient_id = image_path.stem.rsplit("_", 1)[0]
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise RuntimeError(f"Missing mask for {image_path.name}")
        records.append((patient_id, image_path, mask_path, False))
    if not records:
        raise RuntimeError(f"No image/mask pairs found under {root}")

    clean_patients = {row[0] for row in records}
    for scan_dir in sorted(LOCAL_ROOT.glob("RIDER-*_*/")):
        patient_id = scan_dir.name.rsplit("_", 1)[0]
        if patient_id not in clean_patients:
            continue
        empty_images = []
        for image_path in sorted((scan_dir / "images").glob("*.png")):
            mask_path = scan_dir / "masks" / image_path.name
            with Image.open(mask_path) as mask_file:
                if np.asarray(mask_file.convert("L"), dtype=np.uint8).any():
                    continue
            empty_images.append(image_path)
        if not empty_images:
            continue
        positions = np.linspace(
            0, len(empty_images) - 1, NEGATIVES_PER_SCAN + 2, dtype=int
        )[1:-1]
        for position in sorted(set(positions)):
            records.append((patient_id, empty_images[position], None, True))
    return records


def split_by_patient(records):
    patients = sorted({row[0] for row in records})
    random.Random(SEED).shuffle(patients)
    number_of_val_patients = max(1, round(len(patients) * VAL_FRACTION))
    val_patients = set(patients[:number_of_val_patients])
    train_patients = set(patients[number_of_val_patients:])
    train_records = [row for row in records if row[0] in train_patients]
    val_records = [row for row in records if row[0] in val_patients]
    assert train_patients.isdisjoint(val_patients)
    return train_records, val_records, train_patients, val_patients


class RiderDataset(Dataset):
    def __init__(self, records, augment=False, repeats=1):
        self.records = records
        self.augment = augment
        self.repeats = repeats

    def __len__(self):
        return len(self.records) * self.repeats

    def __getitem__(self, index):
        _, image_path, mask_path, flip_vertical = self.records[
            index % len(self.records)
        ]

        with Image.open(image_path) as image_file:
            image = image_file.convert("L")
            if flip_vertical:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
        if mask_path is None:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        else:
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert("L").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
                )
            mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)

        image = np.asarray(image, dtype=np.float32)
        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        image = normalize_ct(image / 255.0)
        return torch.from_numpy(image[None]), torch.from_numpy(mask[None])


def foreground_pos_weight(records):
    foreground = 0
    total = 0
    for _, image_path, mask_path, _ in records:
        if mask_path is None:
            with Image.open(image_path) as image_file:
                total += image_file.width * image_file.height
        else:
            with Image.open(mask_path) as mask_file:
                mask = np.asarray(mask_file.convert("L"), dtype=np.uint8) > 0
            foreground += int(mask.sum())
            total += mask.size
    return min(25.0, (total - foreground) / foreground)


def normalize_ct(image):
    image = np.asarray(image, dtype=np.float32)
    return (image - image.mean()) / (image.std() + 1e-6)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = DoubleConv(1, 16)
        self.encoder2 = DoubleConv(16, 32)
        self.bottleneck = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(32, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        x = self.bottleneck(self.pool(skip2))
        x = self.decoder2(torch.cat((self.up2(x), skip2), dim=1))
        x = self.decoder1(torch.cat((self.up1(x), skip1), dim=1))
        return self.output(x)


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probabilities * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probabilities.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


@torch.no_grad()
def validation_dice(model, loader, device):
    model.eval()
    intersection = 0.0
    predicted_pixels = 0.0
    target_pixels = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        predictions = (torch.sigmoid(model(images)) >= 0.5).float()
        intersection += (predictions * masks).sum().item()
        predicted_pixels += predictions.sum().item()
        target_pixels += masks.sum().item()
    return (2.0 * intersection + 1e-6) / (
        predicted_pixels + target_pixels + 1e-6
    )


def visualize_predictions(model, dataset, device):
    examples = []
    chosen_indices = set()
    for index in range(len(dataset)):
        image, mask = dataset[index]
        if mask.any():
            examples.append((image, mask))
            chosen_indices.add(index)
        if len(examples) == 5:
            break

    for index in range(len(dataset)):
        if len(examples) == 5:
            break
        if index not in chosen_indices:
            examples.append(dataset[index])

    model.eval()
    figure, axes = plt.subplots(5, 4, figsize=(14, 17), squeeze=False)
    column_titles = [
        "Original CT",
        "Ground-truth mask",
        "Predicted mask",
        "Prediction overlay",
    ]
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title)

    with torch.no_grad():
        for row, (image, mask) in enumerate(examples):
            logits = model(image.unsqueeze(0).to(device))
            prediction = (torch.sigmoid(logits)[0, 0] >= 0.5).cpu().numpy()
            ct = image[0].numpy()
            target = mask[0].numpy()

            axes[row, 0].imshow(ct, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].imshow(ct, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].imshow(
                np.ma.masked_where(~prediction, prediction),
                cmap="autumn",
                alpha=0.55,
                vmin=0,
                vmax=1,
            )
            for axis in axes[row]:
                axis.axis("off")

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    records = find_pairs(DATA_ROOT)
    train_records, val_records, train_patients, val_patients = split_by_patient(records)
    train_dataset = RiderDataset(
        train_records, augment=True, repeats=TRAIN_REPEATS
    )
    val_dataset = RiderDataset(val_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_samples_per_epoch = len(train_dataset)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    print(f"Device: {device}")
    positive_pairs = sum(mask_path is not None for _, _, mask_path, _ in records)
    negative_pairs = len(records) - positive_pairs
    print(
        f"Aligned pairs: {positive_pairs} positive, "
        f"{negative_pairs} confirmed-empty"
    )
    print(
        f"Train: {len(train_patients)} patients, {len(train_records):,} unique slices "
        f"({train_samples_per_epoch:,} augmented samples/epoch) | Validation: "
        f"{len(val_patients)} patients, {len(val_dataset):,} slices"
    )
    print(f"Validation patients: {sorted(val_patients)}")

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = foreground_pos_weight(train_records)
    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    print(f"BCE positive-pixel weight: {pos_weight:.2f}")
    best_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        val_dice = validation_dice(model, val_loader, device)
        train_loss = running_loss / train_samples_per_epoch
        print(
            f"Epoch {epoch}/{EPOCHS} | train loss: {train_loss:.4f} | "
            f"validation Dice: {val_dice:.4f}"
        )
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    visualize_predictions(model, val_dataset, device)
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved visualization: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
