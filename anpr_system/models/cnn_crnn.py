"""
CRNN (Convolutional Recurrent Neural Network) + CTC Loss Architecture for ANPR/ALPR.
Matches Kazakh license plate OCR trained on autoriaNumberplateOcrKz dataset.
"""

from typing import List, Tuple, Dict, Optional, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Kazakh / Standard ANPR alphabet: 36 symbols (0-9, A-Z)
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(ALPHABET) + 1  # 37 classes (+1 for CTC blank token at index 0)
BLANK_IDX = 0
IMG_HEIGHT = 32
IMG_WIDTH = 128

# Dictionaries for encoding/decoding
char_to_int: Dict[str, int] = {char: i + 1 for i, char in enumerate(ALPHABET)}
int_to_char: Dict[int, str] = {i: char for char, i in char_to_int.items()}


class CRNN(nn.Module):
    """
    CRNN architecture:
    1. CNN feature extractor (Conv2d, BatchNorm2d, ReLU, MaxPool2d)
    2. Map-to-Sequence (Reshape & Permute)
    3. Bidirectional LSTM (2 layers, hidden_size=256, output feature dim=512)
    4. Linear classification head -> Log Softmax for CTCLoss
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(CRNN, self).__init__()
        self.num_classes = num_classes

        # --- CNN Feature Extractor ---
        # Input: (B, 1, 32, 128)
        self.cnn = nn.Sequential(
            # Stage 1 -> (B, 64, 16, 64)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2 -> (B, 128, 8, 32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3 -> (B, 256, 8, 32)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Stage 4 -> (B, 256, 4, 32)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Stage 5 -> (B, 512, 4, 32)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Stage 6 -> (B, 512, 2, 32)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # --- RNN Sequence Encoder ---
        # 512 channels * 2 height = 1024 input features
        self.rnn = nn.LSTM(
            input_size=512 * 2,
            hidden_size=256,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
        )

        # --- Classification Head ---
        # Bidirectional 256*2 = 512 features
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 1, 32, 128)
        Returns:
            Log-probabilities of shape (T=32, B, num_classes) formatted for CTCLoss
        """
        # CNN backbone -> (B, 512, 2, 32)
        feats = self.cnn(x)

        batch, channels, height, width = feats.size()
        # Flatten channels & height into feature dimension -> (B, 1024, 32)
        feats = feats.reshape(batch, channels * height, width)

        # Permute for RNN: (B, seq_len=32, feat_dim=1024)
        feats = feats.permute(0, 2, 1)

        # BiLSTM -> (B, seq_len=32, 512)
        recurrent_out, _ = self.rnn(feats)

        # Linear projection -> (B, seq_len=32, num_classes)
        logits = self.classifier(recurrent_out)

        # Permute to (seq_len=32, B, num_classes) for PyTorch CTCLoss
        logits = logits.permute(1, 0, 2)
        log_probs = F.log_softmax(logits, dim=2)

        return log_probs


def get_ctc_loss(blank: int = BLANK_IDX, zero_infinity: bool = True) -> nn.CTCLoss:
    """Returns standard CTC Loss configured for CRNN."""
    return nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)


def decode_preds(preds: torch.Tensor, mapping: Optional[Dict[int, str]] = None) -> List[str]:
    """
    Greedy CTC decoding. Collapses repeat characters and removes blank token (0).
    Args:
        preds: Tensor of shape (T, B, C) or (B, T, C)
        mapping: Dictionary mapping class index to character (defaults to int_to_char)
    Returns:
        List of decoded text strings for each item in batch.
    """
    if mapping is None:
        mapping = int_to_char

    # Normalize to shape (B, T, C)
    if preds.dim() == 3 and preds.size(1) != preds.size(0):
        # If shape is (T, B, C), permute to (B, T, C)
        preds_b = preds.permute(1, 0, 2)
    else:
        preds_b = preds

    argmax_indices = torch.argmax(preds_b, dim=2)  # (B, T)
    decoded_texts: List[str] = []

    for sequence in argmax_indices:
        decoded_seq: List[str] = []
        last_idx = 0
        for idx_tensor in sequence:
            idx = int(idx_tensor.item())
            if idx != BLANK_IDX and idx != last_idx:
                decoded_seq.append(mapping.get(idx, ""))
            last_idx = idx
        decoded_texts.append("".join(decoded_seq))

    return decoded_texts


def preprocess_plate_image(
    image: Union[np.ndarray, Image.Image],
    target_height: int = IMG_HEIGHT,
    target_width: int = IMG_WIDTH,
) -> torch.Tensor:
    """
    Preprocess license plate crop for CRNN model inference.
    Args:
        image: NumPy array (BGR/Grayscale) or PIL Image.
    Returns:
        Normalized tensor of shape (1, 1, 32, 128) in range [-1.0, 1.0].
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            # Convert BGR to Grayscale using luminance formula
            if image.shape[2] == 3:
                gray = (0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]).astype(np.uint8)
            else:
                gray = image[:, :, 0]
        else:
            gray = image
        pil_img = Image.fromarray(gray).convert("L")
    else:
        pil_img = image.convert("L")

    resized = pil_img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0  # [0, 1]

    # Normalize mean=[0.5], std=[0.5] -> [-1, 1]
    norm_img = (img_np - 0.5) / 0.5

    # Shape: (1, 1, H, W)
    tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0)
    return tensor


def load_crnn_model(
    weights_path: Optional[str] = None,
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
) -> CRNN:
    """
    Instantiate CRNN model and load weights if provided.
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = CRNN(num_classes=num_classes)

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=dev)
        # Handle state_dict wrapped in 'model' key or direct state_dict
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded CRNN weights from {weights_path}")
    else:
        if weights_path:
            print(f"Weights path '{weights_path}' not found. Initialized with random weights.")

    model.to(dev)
    model.eval()
    return model


def predict_plate_text(
    model: CRNN,
    plate_img: Union[np.ndarray, Image.Image],
    device: str = "cpu",
) -> str:
    """
    End-to-end OCR prediction for a single cropped license plate.
    """
    dev = next(model.parameters()).device
    tensor = preprocess_plate_image(plate_img).to(dev)
    with torch.no_grad():
        preds = model(tensor)
        texts = decode_preds(preds)
    return texts[0] if texts else ""
