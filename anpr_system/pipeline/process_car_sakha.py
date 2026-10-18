"""
Полноценный скрипт обработки видео Car_Sakha.mp4:
- Детекция автомобилей через YOLOv8n (только классы car/bus/truck/motorcycle)
- Трекинг через ByteTrack для уникальных ID
- Подсчёт автомобилей в двух направлениях через контрольную линию
- Извлечение и контрастирование номерных знаков
- OCR распознавание номеров через CRNN
- Генерация аннотированного видео с результатами
"""

import os
import sys
import cv2
import numpy as np
import argparse
from collections import defaultdict

# Fix encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add parent to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def enhance_plate_crop(img):
    """Контрастирование кропа номера: CLAHE + Bilateral + TopHat."""
    if img is None or img.size == 0:
        return None, None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    raw = cv2.resize(gray, (128, 32))

    # Bilateral filter + CLAHE + morphological sharpening
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    result = cv2.add(enhanced, tophat)
    result = cv2.subtract(result, blackhat)
    enhanced_resized = cv2.resize(result, (128, 32))

    return raw, enhanced_resized


def extract_plate_from_vehicle(frame, box, cascade=None):
    """Извлекает кроп номера из bbox автомобиля."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    vw, vh = x2 - x1, y2 - y1
    if vw < 40 or vh < 30:
        return None

    vehicle_roi = frame[y1:y2, x1:x2]

    # Try Haar cascade inside vehicle ROI
    if cascade is not None and vehicle_roi.size > 0:
        gray_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        plates = cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=3, minSize=(30, 10))
        if len(plates) > 0:
            px, py, pw, ph = max(plates, key=lambda p: p[2] * p[3])  # largest plate
            crop = vehicle_roi[py:py + ph, px:px + pw]
            if crop.size > 0:
                return crop

    # Heuristic: bottom 40% of vehicle, central 70%
    y_start = int(0.55 * vh)
    y_end = int(0.95 * vh)
    x_start = int(0.15 * vw)
    x_end = int(0.85 * vw)
    heuristic = vehicle_roi[y_start:y_end, x_start:x_end]
    if heuristic.size > 0:
        return heuristic
    return vehicle_roi


def run_crnn_ocr(model, plate_img, device):
    """Запускает CRNN OCR на кропе номера."""
    if model is None or plate_img is None:
        return ""
    try:
        from models.cnn_crnn import preprocess_plate_image, decode_preds
        tensor = preprocess_plate_image(plate_img).to(device)
        import torch
        with torch.no_grad():
            preds = model(tensor)
            texts = decode_preds(preds)
        return texts[0] if texts else ""
    except Exception:
        return ""


def process_car_sakha(
    video_path,
    output_video_path,
    counting_line_y=450,
    line_x_start=100,
    line_x_end=1800,
    line_offset=25,
    max_frames=None,
    crnn_weights=None,
    cascade_path=None,
):
    """Основной пайплайн обработки видео."""

    # --- 1. Загрузка YOLOv8 ---
    print("=" * 50)
    print("  ANPR Pipeline: Car_Sakha.mp4")
    print("=" * 50)

    yolo_model = None
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        print("[OK] YOLOv8n загружен успешно")
    except ImportError:
        print("[FAIL] ultralytics не установлен! Запустите: pip install ultralytics")
        return None

    # --- 2. Загрузка Haar Cascade для номеров ---
    cascade = None
    if cascade_path and os.path.exists(cascade_path):
        cascade = cv2.CascadeClassifier(cascade_path)
        print(f"[OK] Haar Cascade загружен: {cascade_path}")

    # --- 3. Загрузка CRNN OCR ---
    crnn_model = None
    device_str = "cpu"
    try:
        import torch
        from models.cnn_crnn import CRNN, NUM_CLASSES
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if crnn_weights and os.path.exists(crnn_weights):
            crnn_model = CRNN(NUM_CLASSES)
            crnn_model.load_state_dict(torch.load(crnn_weights, map_location=device_str))
            crnn_model.to(device_str)
            crnn_model.eval()
            print(f"[OK] CRNN OCR загружен: {crnn_weights} (device={device_str})")
        else:
            print("[WARN] CRNN weights не найдены, OCR будет пропущен")
    except Exception as e:
        print(f"[WARN] CRNN не загружен: {e}")

    # --- 4. Открытие видео ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FAIL] Не удалось открыть: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Видео: {width}x{height} @ {fps:.0f} FPS, {total_frames} кадров (~{total_frames/fps:.1f}с)")

    # Writer
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Crops dir
    crops_dir = os.path.join(os.path.dirname(output_video_path), "plate_crops_sakha")
    os.makedirs(crops_dir, exist_ok=True)

    # --- 5. Tracking state ---
    counted_ids = set()
    forward_count = 0  # движение слева направо / сверху вниз
    reverse_count = 0  # движение справа налево / снизу вверх
    track_history = defaultdict(list)  # id -> list of (cx, cy)
    best_plates = {}  # id -> {"text": str, "raw": ndarray, "enh": ndarray, "frame": int}
    saved_crops_count = 0

    # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes = [2, 3, 5, 7]

    print(f"\n[RUN] Обработка видео... Линия подсчёта Y={counting_line_y}")
    print("-" * 50)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        # --- YOLOv8 tracking ---
        results = yolo_model.track(
            source=frame,
            persist=True,
            classes=vehicle_classes,
            conf=0.3,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # Draw counting line
        cv2.line(frame, (line_x_start, counting_line_y), (line_x_end, counting_line_y), (0, 200, 255), 3)

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Get track ID
                track_id = -1
                if boxes.id is not None and len(boxes.id) > i:
                    track_id = int(boxes.id[i].cpu().numpy())

                if track_id < 0:
                    continue

                # Centroid
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                track_history[track_id].append((cx, cy))

                # Keep last 50 points
                if len(track_history[track_id]) > 50:
                    track_history[track_id] = track_history[track_id][-50:]

                # --- Counting logic ---
                if len(track_history[track_id]) >= 2 and track_id not in counted_ids:
                    prev_cy = track_history[track_id][-2][1]
                    curr_cy = cy

                    if line_x_start <= cx <= line_x_end:
                        # Crossed down (forward)
                        if prev_cy < counting_line_y and curr_cy >= counting_line_y:
                            counted_ids.add(track_id)
                            forward_count += 1
                        # Crossed up (reverse)
                        elif prev_cy > counting_line_y and curr_cy <= counting_line_y:
                            counted_ids.add(track_id)
                            reverse_count += 1

                # --- Plate extraction & OCR ---
                plate_crop = extract_plate_from_vehicle(frame, (x1, y1, x2, y2), cascade)
                ocr_text = ""
                if plate_crop is not None:
                    raw, enh = enhance_plate_crop(plate_crop)
                    if enh is not None:
                        ocr_text = run_crnn_ocr(crnn_model, enh, device_str)

                        # Save best plate per vehicle
                        prev_text = best_plates.get(track_id, {}).get("text", "")
                        if len(ocr_text) > len(prev_text):
                            best_plates[track_id] = {
                                "text": ocr_text,
                                "raw": raw,
                                "enh": enh,
                                "frame": frame_idx,
                            }
                            # Save crop images
                            if saved_crops_count < 30:
                                cv2.imwrite(os.path.join(crops_dir, f"id{track_id}_raw.png"), raw)
                                cv2.imwrite(os.path.join(crops_dir, f"id{track_id}_enh.png"), enh)
                                saved_crops_count += 1

                # --- Draw on frame ---
                color = (0, 255, 0) if track_id in counted_ids else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Class name
                cls_names = {2: "car", 3: "moto", 5: "bus", 7: "truck"}
                cls_name = cls_names.get(cls_id, "vehicle")

                display_text = best_plates.get(track_id, {}).get("text", ocr_text)
                label = f"ID:{track_id} {cls_name} {conf:.2f}"
                if display_text:
                    label += f" [{display_text}]"

                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Draw trajectory
                pts = track_history[track_id]
                if len(pts) > 1:
                    for j in range(1, len(pts)):
                        cv2.line(frame, pts[j-1], pts[j], color, 2)

        # --- HUD ---
        total = forward_count + reverse_count
        cv2.rectangle(frame, (15, 15), (420, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (15, 15), (420, 150), (255, 255, 255), 2)
        cv2.putText(frame, f"Total Vehicles: {total}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Forward: {forward_count}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Reverse: {reverse_count}", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (25, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

        if frame_idx % 100 == 0:
            print(f"  Кадр {frame_idx}/{total_frames} | Авто: {total} (Fwd:{forward_count} Rev:{reverse_count})")

    cap.release()
    writer.release()

    # --- Results ---
    print("\n" + "=" * 50)
    print("  РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    print("=" * 50)
    print(f"  Обработано кадров:         {frame_idx}")
    print(f"  Всего уникальных авто:     {forward_count + reverse_count}")
    print(f"  Прямой поток (Forward):    {forward_count}")
    print(f"  Обратный поток (Reverse):  {reverse_count}")
    print("-" * 50)
    print("  Распознанные номера:")
    for tid, data in sorted(best_plates.items()):
        txt = data["text"].strip()
        if txt:
            print(f"    ID #{tid}: '{txt}' (кадр {data['frame']})")
    print("-" * 50)
    print(f"  Видео сохранено: {output_video_path}")
    print(f"  Кропы номеров:  {crops_dir}/")
    print("=" * 50)

    return {
        "total_frames": frame_idx,
        "total_vehicles": forward_count + reverse_count,
        "forward": forward_count,
        "reverse": reverse_count,
        "plates": best_plates,
        "output_video": output_video_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANPR Pipeline for Car_Sakha.mp4")
    parser.add_argument("--video", default="anpr_system/data/raw_video/Car_Sakha.mp4")
    parser.add_argument("--output", default="anpr_system/data/annotated_Car_Sakha.mp4")
    parser.add_argument("--line-y", type=int, default=450, help="Y-coord counting line")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--crnn-weights", default="anpr_system/weights/best_crnn.pth")
    parser.add_argument("--cascade", default="anpr_system/weights/haarcascade_russian_plate_number.xml")
    args = parser.parse_args()

    process_car_sakha(
        video_path=args.video,
        output_video_path=args.output,
        counting_line_y=args.line_y,
        max_frames=args.max_frames,
        crnn_weights=args.crnn_weights,
        cascade_path=args.cascade,
    )
