"""Data augmentation for training dataset."""

import os
import random
from PIL import Image, ImageEnhance
from typing import List, Tuple
from .utils import ensure_dir


def augment_image_and_labels(
    img_path: str,
    label_path: str,
    output_dir: str,
    num_augments: int = 3,
) -> List[Tuple[str, str]]:
    """
    Create augmented versions of image and labels.

    Augmentations:
    - Brightness adjustment
    - Contrast adjustment
    - Color saturation

    Returns list of (img_path, label_path) tuples for augmented versions.
    """
    img = Image.open(img_path)

    # Read labels
    with open(label_path, "r") as f:
        labels = f.read()

    # Base name
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    augmented = []

    for i in range(num_augments):
        # Random augmentations
        aug_img = img.copy()

        # Brightness (0.8 to 1.2)
        brightness = random.uniform(0.8, 1.2)
        aug_img = ImageEnhance.Brightness(aug_img).enhance(brightness)

        # Contrast (0.8 to 1.2)
        contrast = random.uniform(0.8, 1.2)
        aug_img = ImageEnhance.Contrast(aug_img).enhance(contrast)

        # Saturation (0.8 to 1.2)
        saturation = random.uniform(0.8, 1.2)
        aug_img = ImageEnhance.Color(aug_img).enhance(saturation)

        # Save augmented image
        aug_img_name = f"{base_name}_aug{i}.jpg"
        aug_img_path = os.path.join(output_dir, "images", "train", aug_img_name)
        aug_img.save(aug_img_path, quality=95)

        # Copy labels (boxes unchanged for these augmentations)
        aug_label_name = f"{base_name}_aug{i}.txt"
        aug_label_path = os.path.join(output_dir, "labels", "train", aug_label_name)
        with open(aug_label_path, "w") as f:
            f.write(labels)

        augmented.append((aug_img_path, aug_label_path))

    return augmented


def augment_dataset(yolo_dir: str, num_augments: int = 3):
    """Augment training split only."""
    train_img_dir = os.path.join(yolo_dir, "images", "train")
    train_label_dir = os.path.join(yolo_dir, "labels", "train")

    if not os.path.exists(train_img_dir):
        print("No training images found")
        return

    images = [f for f in os.listdir(train_img_dir) if f.endswith((".jpg", ".png"))]
    print(f"Augmenting {len(images)} training images...")

    total_augmented = 0
    for img_file in images:
        img_path = os.path.join(train_img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(train_label_dir, label_file)

        if not os.path.exists(label_path):
            continue

        try:
            augmented = augment_image_and_labels(
                img_path, label_path, yolo_dir, num_augments
            )
            total_augmented += len(augmented)
        except Exception as e:
            print(f"Error augmenting {img_file}: {e}")

    print(f"Created {total_augmented} augmented images")
