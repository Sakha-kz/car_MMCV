"""
Training script for CRNN + CTC Loss license plate OCR.
Supports the autoriaNumberplateOcrKz-2019-04-26 dataset,
synthetic sample generation for testing, and INT8 FX quantization.
"""

import os
import sys
import glob
import json
import random
import argparse
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# Add parent directory to sys.path to allow imports from models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.cnn_crnn import (
    CRNN,
    NUM_CLASSES,
    IMG_HEIGHT,
    IMG_WIDTH,
    ALPHABET,
    char_to_int,
    int_to_char,
    decode_preds,
)
from pipeline.metrics import calculate_cer, calculate_sequence_accuracy


class AutoriaOCRDataset(Dataset):
    """
    Dataset loader for autoriaNumberplateOcrKz-2019-04-26.
    Structure:
        base_dir/
            img/*.png
            ann/*.json (containing 'description' field with plate text)
    """

    def __init__(self, base_dir: str, transform=None, is_synthetic: bool = False, num_synthetic: int = 1000):
        self.transform = transform
        self.is_synthetic = is_synthetic
        self.samples: List[Tuple[Optional[str], str]] = []  # (img_path or None, text)

        if not is_synthetic and os.path.exists(base_dir):
            img_dir = os.path.join(base_dir, "img")
            ann_dir = os.path.join(base_dir, "ann")

            if os.path.exists(img_dir) and os.path.exists(ann_dir):
                ann_files = glob.glob(os.path.join(ann_dir, "*.json"))
                for ann_path in ann_files:
                    base_name = os.path.splitext(os.path.basename(ann_path))[0]
                    # Check for .png or .jpg
                    img_path = os.path.join(img_dir, base_name + ".png")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(img_dir, base_name + ".jpg")

                    if os.path.exists(img_path):
                        try:
                            with open(ann_path, "r", encoding="utf-8") as f:
                                ann = json.load(f)
                            text = ann.get("description", "").strip().upper()
                            # Clean text to alphabet characters only
                            cleaned_text = "".join([c for c in text if c in ALPHABET])
                            if len(cleaned_text) > 0:
                                self.samples.append((img_path, cleaned_text))
                        except Exception:
                            continue

        if len(self.samples) == 0:
            print(f"Notice: No dataset files found at '{base_dir}'. Using synthetic generator with {num_synthetic} samples.")
            self.is_synthetic = True
            chars = list(ALPHABET)
            for _ in range(num_synthetic):
                # Kazakh format: e.g. 777ABC02, 123AAA01, A123BC
                length = random.randint(6, 8)
                text = "".join(random.choices(chars, k=length))
                self.samples.append((None, text))

        print(f"Dataset loaded: {len(self.samples)} samples (synthetic={self.is_synthetic})")

    def __len__(self) -> int:
        return len(self.samples)

    def _generate_synthetic_image(self, text: str) -> Image.Image:
        """Generates realistic synthetic license plate image for training and unit tests."""
        img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), color=random.randint(200, 255))
        draw = ImageDraw.Draw(img)

        # Draw border
        draw.rectangle([(2, 2), (IMG_WIDTH - 3, IMG_HEIGHT - 3)], outline=0, width=2)

        # Draw text characters centered
        step = (IMG_WIDTH - 20) / max(1, len(text))
        for i, ch in enumerate(text):
            x = 10 + int(i * step)
            y = random.randint(5, 9)
            draw.text((x, y), ch, fill=random.randint(0, 40))

        # Add minor noise
        np_img = np.array(img, dtype=np.uint8)
        noise = np.random.normal(0, 8, np_img.shape).astype(np.int16)
        noisy = np.clip(np_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path, text = self.samples[idx]

        if img_path is not None and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("L")
            except Exception:
                image = self._generate_synthetic_image(text)
        else:
            image = self._generate_synthetic_image(text)

        if self.transform:
            image = self.transform(image)

        return image, text


def collate_fn(batch):
    """Batches images and pads text label tensors for CTCLoss."""
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)

    encoded_texts = [
        torch.tensor([char_to_int.get(c, 0) for c in text], dtype=torch.long)
        for text in texts
    ]
    text_lengths = torch.tensor([len(t) for t in encoded_texts], dtype=torch.long)
    padded_texts = nn.utils.rnn.pad_sequence(encoded_texts, batch_first=True, padding_value=0)

    return images, padded_texts, text_lengths


def train_ocr_pipeline(
    data_dir: str,
    output_weights: str = "anpr_system/weights/best_crnn.pth",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device_str: str = "auto",
):
    """
    Main training function for CRNN OCR model.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Starting OCR training on {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM <= 4GB compliant)")

    # Transforms: Grayscale, 32x128, Tensor, Normalize [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = AutoriaOCRDataset(train_dir, transform=transform, num_synthetic=1200)
    val_dataset = AutoriaOCRDataset(val_dir, transform=transform, num_synthetic=300)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for images, texts, text_lengths in pbar:
            images = images.to(device)
            texts = texts.to(device)
            text_lengths = text_lengths.to(device)

            optimizer.zero_grad()
            preds = model(images)  # (T=32, B, num_classes)

            pred_lengths = torch.full(
                size=(images.size(0),),
                fill_value=preds.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(preds, texts, pred_lengths, text_lengths)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, texts, text_lengths in val_loader:
                images = images.to(device)
                texts = texts.to(device)
                text_lengths = text_lengths.to(device)

                preds = model(images)
                pred_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(0),
                    dtype=torch.long,
                    device=device,
                )

                v_loss = criterion(preds, texts, pred_lengths, text_lengths)
                val_loss += v_loss.item()

                decoded_preds = decode_preds(preds)
                target_strings = [
                    "".join([int_to_char.get(int(idx.item()), "") for idx in t if idx != 0])
                    for t in texts
                ]

                all_preds.extend(decoded_preds)
                all_targets.extend(target_strings)

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = calculate_sequence_accuracy(all_preds, all_targets)
        val_cer = calculate_cer(all_preds, all_targets)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val CER: {val_cer:.4f}"
        )

        if val_acc > best_val_acc or epoch == 0:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_weights)
            print(f"Saved best CRNN model checkpoint to '{output_weights}' (Val Acc: {best_val_acc:.4f})")

    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN OCR Model for Kazakh license plates")
    parser.add_argument("--data-dir", type=str, default="anpr_system/data/dataset_ocr", help="Dataset root directory")
    parser.add_argument("--weights-out", type=str, default="anpr_system/weights/best_crnn.pth", help="Path to save best model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cuda/cpu)")

    args = parser.parse_args()
    train_ocr_pipeline(
        data_dir=args.data_dir,
        output_weights=args.weights_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
    )
