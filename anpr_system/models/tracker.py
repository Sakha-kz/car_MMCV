"""
Vehicle detection and tracking module integrating YOLOv8 + ByteTrack / SORT,
with License Plate localization and contrast enhancement preprocessing.
"""

from typing import List, Tuple, Dict, Optional, Union, Any
import os
import cv2
import numpy as np


class PlateContrastEnhancer:
    """
    Image contrast enhancement and preprocessing methods for license plate crops
    before feeding to CRNN OCR.
    """

    @staticmethod
    def apply_clahe(
        gray_img: np.ndarray,
        clip_limit: float = 3.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray_img)

    @staticmethod
    def apply_adaptive_threshold(gray_img: np.ndarray) -> np.ndarray:
        """Applies Gaussian adaptive thresholding for clear character separation."""
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    @staticmethod
    def enhance_contrast(
        img: np.ndarray,
        target_size: Tuple[int, int] = (128, 32),  # (width, height)
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Complete license plate preprocessing pipeline.
        Args:
            img: Input image (BGR or Grayscale)
            target_size: (width, height) = (128, 32)
        Returns:
            (raw_resized, enhanced_resized, intermediate_stages_dict)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Raw resized (Before)
        raw_resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)

        # 1. Bilateral filter to reduce noise while keeping character edges sharp
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # 2. CLAHE for local contrast normalization
        clahe_out = PlateContrastEnhancer.apply_clahe(denoised, clip_limit=3.0)

        # 3. Morphological black-hat / sharpening to highlight text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(clahe_out, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(clahe_out, cv2.MORPH_BLACKHAT, kernel)
        sharpened = cv2.add(clahe_out, tophat)
        sharpened = cv2.subtract(sharpened, blackhat)

        # 4. Final resize to 32x128
        enhanced_resized = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_CUBIC)

        stages = {
            "raw": raw_resized,
            "denoised": cv2.resize(denoised, target_size),
            "clahe": cv2.resize(clahe_out, target_size),
            "enhanced": enhanced_resized,
        }

        return raw_resized, enhanced_resized, stages


class CentroidTrackerFallback:
    """
    Lightweight fallback tracker (SORT / Centroid tracker) used if ByteTrack / YOLO
    tracking dependencies are unavailable or for fast CPU execution.
    """

    def __init__(self, max_disappeared: int = 25, max_distance: float = 80.0):
        self.next_object_id = 1
        self.objects: Dict[int, Tuple[int, int]] = {}  # id -> centroid (cx, cy)
        self.boxes: Dict[int, Tuple[int, int, int, int]] = {}  # id -> (x1, y1, x2, y2)
        self.disappeared: Dict[int, int] = {}
        self.trajectories: Dict[int, List[Tuple[int, int]]] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], box: Tuple[int, int, int, int]) -> int:
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.boxes[obj_id] = box
        self.disappeared[obj_id] = 0
        self.trajectories[obj_id] = [centroid]
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id: int):
        self.objects.pop(obj_id, None)
        self.boxes.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """
        rects: List of (x1, y1, x2, y2)
        Returns: Dict of id -> (x1, y1, x2, y2)
        """
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.boxes

        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Distance matrix between current object centroids and new input centroids
            dists = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
            for i, (ox, oy) in enumerate(object_centroids):
                for j, (ix, iy) in enumerate(input_centroids):
                    dists[i, j] = np.hypot(ox - ix, oy - iy)

            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if dists[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.boxes[obj_id] = rects[col]
                self.disappeared[obj_id] = 0
                self.trajectories[obj_id].append(input_centroids[col])

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, dists.shape[0])).difference(used_rows)
            unused_cols = set(range(0, dists.shape[1])).difference(used_cols)

            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.boxes


class VehicleAndPlateTracker:
    """
    Integrates YOLOv8 object detector with ByteTrack / fallback tracker,
    vehicle classification (car, bus, truck, motorcycle),
    and license plate crop extraction.
    """

    # COCO vehicle class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]

    def __init__(
        self,
        yolo_weights: str = "yolov8n.pt",
        cascade_weights: Optional[str] = None,
        conf_thresh: float = 0.35,
        use_bytetrack: bool = True,
        device: str = "auto",
    ):
        self.conf_thresh = conf_thresh
        self.use_bytetrack = use_bytetrack
        self.device = device
        self.yolo_model = None
        self.cascade_detector = None
        self.fallback_tracker = CentroidTrackerFallback()
        self.enhancer = PlateContrastEnhancer()

        # Load YOLOv8 if ultralytics is installed
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_weights)
            print(f"Loaded YOLOv8 model from {yolo_weights}")
        except Exception as e:
            print(f"Warning: YOLOv8 could not be initialized directly ({e}). Using background/cascade fallback.")

        # Background subtractor fallback for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # Load Haar Cascade for plate detection if available
        if cascade_weights and os.path.exists(cascade_weights):
            self.cascade_detector = cv2.CascadeClassifier(cascade_weights)
            print(f"Loaded Haar Cascade plate detector from {cascade_weights}")

    def detect_and_track_vehicles(
        self, frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect and track vehicles on a given frame.
        Returns list of track dicts:
            {
                "id": int,
                "box": (x1, y1, x2, y2),
                "conf": float,
                "class_id": int,
                "class_name": str,
                "plate_crop": Optional[np.ndarray],
            }
        """
        results: List[Dict[str, Any]] = []

        if self.yolo_model is not None:
            try:
                # Use YOLOv8 tracking with ByteTrack
                tracks = self.yolo_model.track(
                    source=frame,
                    persist=True,
                    classes=self.VEHICLE_CLASS_IDS,
                    conf=self.conf_thresh,
                    tracker="bytetrack.yaml" if self.use_bytetrack else None,
                    verbose=False,
                    device=None if self.device == "auto" else self.device,
                )

                if tracks and len(tracks) > 0 and tracks[0].boxes is not None:
                    boxes_obj = tracks[0].boxes
                    for i in range(len(boxes_obj)):
                        box = boxes_obj.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        conf = float(boxes_obj.conf[i].cpu().numpy())
                        cls_id = int(boxes_obj.cls[i].cpu().numpy())

                        # Track ID if available, otherwise fallback
                        if boxes_obj.id is not None and len(boxes_obj.id) > i and boxes_obj.id[i] is not None:
                            track_id = int(boxes_obj.id[i].cpu().numpy())
                        else:
                            track_id = i + 1

                        # Crop plate candidate from vehicle bounding box
                        plate_crop = self.extract_plate_crop(frame, (x1, y1, x2, y2))

                        results.append({
                            "id": track_id,
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "class_id": cls_id,
                            "class_name": self.yolo_model.names.get(cls_id, "vehicle"),
                            "plate_crop": plate_crop,
                        })
                    return results
            except Exception as e:
                # If YOLO track fails, fallback to CentroidTracker
                pass

        # Fallback detection using MOG2 background subtractor + morphological filtering
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(blur)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 50 and h >= 50:
                rects.append((x, y, x + w, y + h))

        tracked = self.fallback_tracker.update(rects)
        for obj_id, (x1, y1, x2, y2) in tracked.items():
            crop = self.extract_plate_crop(frame, (x1, y1, x2, y2))
            results.append({
                "id": obj_id,
                "box": (x1, y1, x2, y2),
                "conf": 0.85,
                "class_id": 2,
                "class_name": "car",
                "plate_crop": crop if (crop is not None and crop.size > 0) else None,
            })

        return results

    def extract_plate_crop(
        self, frame: np.ndarray, vehicle_box: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extracts license plate candidate from a detected vehicle bounding box.
        1. Checks cascade detector inside vehicle ROI.
        2. If not found, extracts heuristic lower-central ROI (where plates are mounted).
        """
        h_frame, w_frame = frame.shape[:2]
        vx1, vy1, vx2, vy2 = vehicle_box
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(w_frame, vx2), min(h_frame, vy2)

        v_w = vx2 - vx1
        v_h = vy2 - vy1
        if v_w < 30 or v_h < 30:
            return None

        vehicle_roi = frame[vy1:vy2, vx1:vx2]

        # 1. Try Haar Cascade inside vehicle ROI
        if self.cascade_detector is not None and vehicle_roi.size > 0:
            gray_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            plates = self.cascade_detector.detectMultiScale(
                gray_roi, scaleFactor=1.1, minNeighbors=3, minSize=(30, 10)
            )
            if len(plates) > 0:
                px, py, pw, ph = plates[0]
                plate_crop = vehicle_roi[py : py + ph, px : px + pw]
                if plate_crop.size > 0:
                    return plate_crop

        # 2. Heuristic plate extraction: lower 45% of car, centered 60%
        y_start = int(0.55 * v_h)
        y_end = int(0.95 * v_h)
        x_start = int(0.20 * v_w)
        x_end = int(0.80 * v_w)

        heuristic_crop = vehicle_roi[y_start:y_end, x_start:x_end]
        if heuristic_crop.size > 0:
            return heuristic_crop

        return vehicle_roi
