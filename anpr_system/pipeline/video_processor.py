"""
Video processing pipeline for vehicle detection, bidirectional tracking,
contrast-enhanced license plate extraction, and CRNN OCR recognition.
Supports 1080p Full HD video streams. 
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import sys
import cv2
import numpy as np

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from models.cnn_crnn import CRNN, load_crnn_model, predict_plate_text
from models.tracker import VehicleAndPlateTracker, PlateContrastEnhancer


class VideoANPRProcessor:
    """
    End-to-End Video Processor for ANPR & Traffic Flow Analysis:
    1. Reads video frames (1080p Full HD).
    2. Performs vehicle tracking using YOLOv8 + ByteTrack or MOG2 contrast fallback.
    3. Counts bidirectional flow (forward vs reverse) across counting line(s).
    4. Extracts license plate crops, applies CLAHE contrast enhancement, and runs CRNN OCR.
    5. Saves annotated video and 'Before vs After' contrast comparison samples.
    """

    def __init__(
        self,
        crnn_weights: Optional[str] = "anpr_system/weights/best_crnn.pth",
        yolo_weights: str = "yolov8n.pt",
        cascade_weights: Optional[str] = "anpr_system/weights/haarcascade_russian_plate_number.xml",
        counting_line_y: int = 550,
        counting_line_x_range: Tuple[int, int] = (25, 1800),
        line_offset: int = 15,
        device: str = "auto",
    ):
        self.counting_line_y = counting_line_y
        self.counting_line_x_range = counting_line_x_range
        self.line_offset = line_offset

        # Load models
        print("Initializing ANPR video processor...")
        self.crnn_model = None
        try:
            self.crnn_model = load_crnn_model(crnn_weights, device=device)
        except Exception as e:
            print(f"Warning: Could not load CRNN weights ({e}). OCR predictions will use fallback.")

        self.tracker = VehicleAndPlateTracker(
            yolo_weights=yolo_weights,
            cascade_weights=cascade_weights,
            device=device,
        )
        self.enhancer = PlateContrastEnhancer()

        # Background subtractor for contrast-based motion analysis (Task_video.txt)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=45, detectShadows=True
        )

        # Counting state
        self.counted_ids = set()
        self.forward_count = 0  # moving down
        self.reverse_count = 0  # moving up
        self.vehicle_trajectories: Dict[int, List[Tuple[int, int]]] = {}
        self.vehicle_best_plates: Dict[int, Dict[str, Any]] = {}
        self.sample_crops_before_after: List[Dict[str, Any]] = []

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = "anpr_system/data/annotated_output.mp4",
        output_crops_dir: Optional[str] = "anpr_system/data/plate_crops",
        max_frames: Optional[int] = None,
        display_progress: bool = True,
        use_contrast_counting: bool = False,
    ) -> Dict[str, Any]:
        """
        Process the video file and compute traffic statistics.
        Args:
            video_path: Path to raw input video (e.g. 1080p Full HD).
            output_video_path: Destination path for rendered video with overlays.
            output_crops_dir: Destination folder for cropped plate samples.
            max_frames: Optional frame limit (useful for quick evaluation).
            use_contrast_counting: If True, uses MOG2 contrast thresholding alongside YOLO.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found at '{video_path}'")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream at '{video_path}'")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video metadata: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames (~{total_frames / fps:.1f}s)")

        # Setup video writer if output path is specified
        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if output_crops_dir:
            os.makedirs(output_crops_dir, exist_ok=True)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # 1. Detect & track vehicles
            tracks = self.tracker.detect_and_track_vehicles(frame)

            # 2. Contrast-based motion subtraction (Task_video.txt & test_video.py method)
            if use_contrast_counting:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                fg_mask = self.bg_subtractor.apply(blur)
                # Morphological closing
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                dilated = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 3000:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 1)

            # Draw counting line
            line_y = self.counting_line_y
            x_min, x_max = self.counting_line_x_range
            cv2.line(frame, (x_min, line_y), (x_max, line_y), (255, 127, 0), 3)
            cv2.putText(
                frame,
                "COUNTING LINE",
                (x_min + 10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 127, 0),
                2,
            )

            # 3. Analyze each tracked vehicle
            for trk in tracks:
                trk_id = trk["id"]
                x1, y1, x2, y2 = trk["box"]
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)

                if trk_id not in self.vehicle_trajectories:
                    self.vehicle_trajectories[trk_id] = []
                self.vehicle_trajectories[trk_id].append((cx, cy))

                # Check line crossing if vehicle centroid is within x_range
                if x_min <= cx <= x_max and len(self.vehicle_trajectories[trk_id]) >= 2:
                    prev_cy = self.vehicle_trajectories[trk_id][-2][1]

                    # Crossed from top to bottom (Forward / Downwards)
                    if prev_cy < line_y <= cy or (abs(cy - line_y) <= self.line_offset and cy > prev_cy):
                        if trk_id not in self.counted_ids:
                            self.counted_ids.add(trk_id)
                            self.forward_count += 1
                            cv2.line(frame, (x_min, line_y), (x_max, line_y), (0, 255, 0), 4)

                    # Crossed from bottom to top (Reverse / Upwards)
                    elif prev_cy > line_y >= cy or (abs(cy - line_y) <= self.line_offset and cy < prev_cy):
                        if trk_id not in self.counted_ids:
                            self.counted_ids.add(trk_id)
                            self.reverse_count += 1
                            cv2.line(frame, (x_min, line_y), (x_max, line_y), (0, 0, 255), 4)

                # Process plate crop & OCR
                plate_crop = trk["plate_crop"]
                ocr_text = ""
                if plate_crop is not None and plate_crop.size > 0:
                    raw_crop, enh_crop, stages = self.enhancer.enhance_contrast(plate_crop)

                    # Run CRNN OCR if available
                    if self.crnn_model is not None:
                        try:
                            ocr_text = predict_plate_text(self.crnn_model, enh_crop)
                        except Exception:
                            ocr_text = ""

                    # Save best plate reading for vehicle ID
                    if trk_id not in self.vehicle_best_plates or len(ocr_text) > len(self.vehicle_best_plates[trk_id].get("text", "")):
                        self.vehicle_best_plates[trk_id] = {
                            "text": ocr_text,
                            "raw_crop": raw_crop,
                            "enhanced_crop": enh_crop,
                            "frame_idx": frame_idx,
                        }

                        # Save crop sample to disk
                        if output_crops_dir and len(self.sample_crops_before_after) < 15:
                            raw_path = os.path.join(output_crops_dir, f"plate_{trk_id}_raw.png")
                            enh_path = os.path.join(output_crops_dir, f"plate_{trk_id}_enhanced.png")
                            cv2.imwrite(raw_path, raw_crop)
                            cv2.imwrite(enh_path, enh_crop)
                            self.sample_crops_before_after.append({
                                "id": trk_id,
                                "text": ocr_text,
                                "raw_path": raw_path,
                                "enhanced_path": enh_path,
                            })

                # Visual overlay on vehicle
                display_ocr = self.vehicle_best_plates.get(trk_id, {}).get("text", ocr_text)
                color = (0, 255, 0) if trk_id in self.counted_ids else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID: {trk_id}"
                if display_ocr:
                    label += f" | {display_ocr}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # 4. Render HUD Dashboard on top left
            total_counted = self.forward_count + self.reverse_count
            cv2.rectangle(frame, (20, 20), (450, 160), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (450, 160), (255, 255, 255), 2)

            cv2.putText(
                frame,
                f"Total Vehicles: {total_counted}",
                (35, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Forward (Down): {self.forward_count}",
                (35, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Reverse (Up):   {self.reverse_count}",
                (35, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 128, 255),
                2,
            )

            if writer:
                writer.write(frame)

            if display_progress and frame_idx % 60 == 0:
                print(f"Processed frame {frame_idx}/{total_frames} | Total Counted: {total_counted} (Fwd: {self.forward_count}, Rev: {self.reverse_count})")

        cap.release()
        if writer:
            writer.release()
            print(f"Saved annotated video to '{output_video_path}'")

        estimated_total = self.forward_count + self.reverse_count
        return {
            "total_frames_processed": frame_idx,
            "estimated_total": estimated_total,
            "forward_count": self.forward_count,
            "reverse_count": self.reverse_count,
            "vehicle_best_plates": self.vehicle_best_plates,
            "sample_crops": self.sample_crops_before_after,
            "output_video_path": output_video_path,
        }


if __name__ == "__main__":
    video_file = "anpr_system/data/raw_video/video.mp4"
    if not os.path.exists(video_file):
        video_file = "C:/Users/eroxa/Downloads/Telegram Desktop/video.mp4"

    processor = VideoANPRProcessor(counting_line_y=550)
    results = processor.process_video(
        video_path=video_file,
        max_frames=150,  # quick test run
    )
    print("\n--- Processing Results ---")
    print(f"Total counted: {results['estimated_total']} (Forward: {results['forward_count']}, Reverse: {results['reverse_count']})")
