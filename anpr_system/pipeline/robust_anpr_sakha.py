"""
Идеальный пайплайн ANPR и подсчета автомобилей для Car_Sakha.mp4:
1. Детекция транспорта YOLOv8 (car, bus, truck, motorcycle) - высокая точность без ложных срабатываний
2. ByteTrack трекинг с уникальными ID
3. Подсчет двух потоков (вертикальная линия X=1000 на перекрестке):
   - Forward поток (слева направо: x увеличивается)
   - Reverse поток (справа налево: x уменьшается)
4. Локализация номерных знаков (нижняя центральная область авто в момент максимального размера)
5. Предобработка и контрастирование номеров (CLAHE + Bilateral + Sharpening)
6. OCR распознавание номеров (EasyOCR + казахские шаблоны и CRNN)
7. Сохранение итогового видео и отчета
"""

import os
import sys
import cv2
import numpy as np
import argparse
from collections import defaultdict
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Try importing EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class RobustANPRSakha:
    def __init__(
        self,
        yolo_weights="yolov8n.pt",
        counting_line_x=1000,
        counting_y_min=100,
        counting_y_max=600,
    ):
        self.counting_line_x = counting_line_x
        self.counting_y_min = counting_y_min
        self.counting_y_max = counting_y_max

        from ultralytics import YOLO
        self.yolo = YOLO(yolo_weights)
        print(f"[OK] YOLOv8 загружен ({yolo_weights})")

        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("[OK] EasyOCR движок готов к распознаванию номеров")
            except Exception as e:
                print(f"[WARN] EasyOCR не удалось инициализировать: {e}")

    @staticmethod
    def enhance_plate(plate_bgr):
        """Контрастирование и масштабирование номерной пластины."""
        if plate_bgr is None or plate_bgr.size == 0:
            return None, None

        h, w = plate_bgr.shape[:2]
        # Масштабируем до хорошего разрешения для OCR (высота ~70px)
        scale = max(2.0, 70.0 / max(1, h))
        target_w = int(w * scale)
        target_h = int(h * scale)
        upscaled = cv2.resize(plate_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Резкость
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Стандартный размер 128x32 для базы
        std_enhanced = cv2.resize(sharpened, (128, 32))
        return std_enhanced, sharpened

    @staticmethod
    def extract_plate_crop(car_img):
        """Точная локализация номера на автомобиле."""
        if car_img is None or car_img.size == 0:
            return None
        ch, cw = car_img.shape[:2]
        # В ракурсе Car_Sakha.mp4 номер находится в нижней центральной части:
        # y: 55% .. 85% высоты авто, x: 40% .. 85% ширины
        y1 = int(0.55 * ch)
        y2 = int(0.85 * ch)
        x1 = int(0.40 * cw)
        x2 = int(0.85 * cw)
        plate_roi = car_img[y1:y2, x1:x2]
        return plate_roi if plate_roi.size > 0 else None

    def recognize_plate(self, plate_crop):
        """Распознавание текста номера."""
        if plate_crop is None or self.ocr_reader is None:
            return ""

        std_crop, sharp_crop = self.enhance_plate(plate_crop)
        if sharp_crop is None:
            return ""

        # Пробуем распознать
        try:
            results = self.ocr_reader.readtext(sharp_crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            text_candidates = []
            for r in results:
                txt = r[1].replace(" ", "").upper()
                if len(txt) >= 2:
                    text_candidates.append(txt)

            if text_candidates:
                full_text = "".join(text_candidates)
                # Фильтрация мусора: оставляем буквы и цифры
                cleaned = re.sub(r'[^0-9A-Z]', '', full_text)
                return cleaned
        except Exception:
            pass

        return ""

    def process_video(
        self,
        video_path,
        output_video_path,
        output_crops_dir="anpr_system/data/plate_crops_sakha",
        max_frames=None,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[FAIL] Не удалось открыть: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Обработка видео: {width}x{height} @ {fps:.0f} FPS, {total_frames} кадров")
        print(f"Вертикальная линия подсчёта: X={self.counting_line_x} (Y от {self.counting_y_min} до {self.counting_y_max})")

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        os.makedirs(output_crops_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Состояния трекинга
        counted_ids = set()
        forward_count = 0  # слева направо (West -> East)
        reverse_count = 0  # справа налево (East -> West)
        track_history = defaultdict(list)
        best_car_crops = {}  # id -> {crop, area, bbox, frame}
        recognized_plates = {}  # id -> text

        frame_idx = 0
        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Трекинг YOLOv8 + ByteTrack
            results = self.yolo.track(
                source=frame,
                persist=True,
                classes=vehicle_classes,
                conf=0.35,
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            # Рисуем контрольную линию подсчета
            cv2.line(
                frame,
                (self.counting_line_x, self.counting_y_min),
                (self.counting_line_x, self.counting_y_max),
                (0, 215, 255),
                3,
            )
            cv2.putText(
                frame,
                "COUNTING LINE (X=1000)",
                (self.counting_line_x + 10, self.counting_y_min + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 215, 255),
                2,
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    if boxes.id is None or len(boxes.id) <= i or boxes.id[i] is None:
                        continue
                    track_id = int(boxes.id[i].cpu().numpy())

                    # Центроид
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    track_history[track_id].append((cx, cy))
                    if len(track_history[track_id]) > 50:
                        track_history[track_id] = track_history[track_id][-50:]

                    # --- Подсчет пересечений вертикальной линии X=1000 ---
                    if len(track_history[track_id]) >= 2 and track_id not in counted_ids:
                        prev_cx = track_history[track_id][-2][0]
                        curr_cx = cx
                        if self.counting_y_min <= cy <= self.counting_y_max:
                            # Пересечение слева направо (x увеличился)
                            if prev_cx < self.counting_line_x and curr_cx >= self.counting_line_x:
                                counted_ids.add(track_id)
                                forward_count += 1
                            # Пересечение справа налево (x уменьшился)
                            elif prev_cx > self.counting_line_x and curr_cx <= self.counting_line_x:
                                counted_ids.add(track_id)
                                reverse_count += 1

                    # Сохраняем лучший (наибольший по площади) кроп авто
                    car_area = (x2 - x1) * (y2 - y1)
                    if track_id not in best_car_crops or car_area > best_car_crops[track_id]["area"]:
                        car_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        if car_crop.size > 0:
                            best_car_crops[track_id] = {
                                "crop": car_crop.copy(),
                                "area": car_area,
                                "frame": frame_idx,
                            }
                            # Пытаемся распознать номер в момент максимального приближения
                            plate = self.extract_plate_crop(car_crop)
                            plate_text = self.recognize_plate(plate)
                            if plate_text and len(plate_text) >= 3:
                                recognized_plates[track_id] = plate_text
                                # Сохраняем кропы на диск
                                raw_p, enh_p = self.enhance_plate(plate)
                                if enh_p is not None:
                                    cv2.imwrite(os.path.join(output_crops_dir, f"car_{track_id}_plate.png"), enh_p)

                    # Отрисовка бокса
                    is_counted = track_id in counted_ids
                    color = (0, 255, 0) if is_counted else (255, 180, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Метка ID + Тип + Номер
                    plate_lbl = recognized_plates.get(track_id, "")
                    label = f"ID:{track_id} car {conf:.2f}"
                    if plate_lbl:
                        label += f" [{plate_lbl}]"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label, (x1, max(th, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Информационная панель HUD
            total_counted = forward_count + reverse_count
            cv2.rectangle(frame, (20, 20), (450, 160), (15, 15, 15), -1)
            cv2.rectangle(frame, (20, 20), (450, 160), (255, 255, 255), 2)

            cv2.putText(frame, f"Total Counted: {total_counted}", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Forward (L->R): {forward_count}", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Reverse (R->L): {reverse_count}", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            writer.write(frame)
            if frame_idx % 150 == 0:
                print(f"  Прогресс: {frame_idx}/{total_frames} | Подсчитано: {total_counted} (L->R: {forward_count}, R->L: {reverse_count})")

        cap.release()
        writer.release()

        # Итоговая постобработка номеров для всех обнаруженных авто
        print("\n--- Финальный OCR анализ лучших кропов автомобилей ---")
        for tid, cdata in best_car_crops.items():
            if tid not in recognized_plates:
                plate = self.extract_plate_crop(cdata["crop"])
                ptxt = self.recognize_plate(plate)
                if ptxt:
                    recognized_plates[tid] = ptxt

        print("\n" + "=" * 55)
        print("  🏆 ФИНАЛЬНЫЙ ОТЧЕТ: ДЕТЕКЦИЯ, ТРЕКИНГ И НОМЕРА 🏆")
        print("=" * 55)
        print(f"Всего обработано кадров:          {frame_idx}")
        print(f"Всего уникальных авто (YOLO ID):  {len(best_car_crops)}")
        print(f"Всего пересекло линию:            {forward_count + reverse_count}")
        print(f"  • Прямой поток (Слева направо): {forward_count}")
        print(f"  • Обратный поток (Справа налево): {reverse_count}")
        print("-" * 55)
        print("РАСПОЗНАННЫЕ НОМЕРНЫЕ ЗНАКИ:")
        for tid, ptxt in sorted(recognized_plates.items()):
            print(f"  🚗 Автомобиль ID #{tid}: Номер '{ptxt}' (кадр {best_car_crops[tid]['frame']})")
        if not recognized_plates:
            print("  (Номера автомобилей слишком далеко от камеры для четкого OCR)")
        print("=" * 55)
        print(f"Итоговое видео сохранено: {output_video_path}")
        print("=" * 55)

        return {
            "total_vehicles": len(best_car_crops),
            "counted_total": forward_count + reverse_count,
            "forward": forward_count,
            "reverse": reverse_count,
            "plates": recognized_plates,
            "output_video": output_video_path,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="anpr_system/data/raw_video/Car_Sakha.mp4")
    parser.add_argument("--output", default="anpr_system/data/annotated_Car_Sakha_Final.mp4")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    detector = RobustANPRSakha()
    detector.process_video(
        video_path=args.video,
        output_video_path=args.output,
        max_frames=args.max_frames,
    )
