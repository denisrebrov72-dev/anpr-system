import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping, QConfig
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from collections import Counter

from config import (
    OCR_IMG_HEIGHT, OCR_IMG_WIDTH, OCR_ALPHABET,
    DETECTION_CONFIDENCE_THRESHOLD,
)
from utils import normalize_plate, to_cyrillic, is_valid_russian_plate


class CRNN(nn.Module):
    """CRNN: CNN (глаза) + LSTM (чтение) + Linear (классификатор)."""

    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.rnn = nn.LSTM(512 * 2, 256, bidirectional=True,
                           num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.reshape(batch, channels * height, width)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)
        x = nn.functional.log_softmax(x, dim=2)
        return x


class YOLODetector:
    """YOLOv8 детектор номеров + ByteTrack трекинг."""

    def __init__(self, model_path: str, device: torch.device):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        print("✅ Детектор YOLO успешно загружен.")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.model.predict(frame, verbose=False, device=self.device)
        results = []
        for det in detections[0].boxes.data:
            x1, y1, x2, y2, conf, _ = det.cpu().numpy()
            if conf >= DETECTION_CONFIDENCE_THRESHOLD:
                results.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                })
        return results

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.model.track(
            frame, persist=True, verbose=False,
            device=self.device, tracker="bytetrack.yaml"
        )
        results = []
        if detections[0].boxes.id is None:
            return results
        track_ids = detections[0].boxes.id.int().cpu().tolist()
        boxes = detections[0].boxes.xyxy.cpu().numpy()
        confs = detections[0].boxes.conf.cpu().numpy()
        for box, track_id, conf in zip(boxes, track_ids, confs):
            if conf >= DETECTION_CONFIDENCE_THRESHOLD:
                results.append({
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "confidence": float(conf),
                    "track_id": track_id,
                })
        return results


class CRNNRecognizer:
    """CRNN-распознаватель, загрузка INT8-модели с fallback на float."""

    def __init__(self, model_path: str, device: torch.device):
        self.device = torch.device("cpu")  # INT8-fbgemm — только x86 CPU
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Grayscale(),
            transforms.Resize((OCR_IMG_HEIGHT, OCR_IMG_WIDTH)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.int_to_char = {i + 1: char for i, char in enumerate(OCR_ALPHABET)}
        self.int_to_char[0] = ''
        num_classes = len(OCR_ALPHABET) + 1

        try:
            loaded = torch.load(model_path, map_location=self.device)
            if not isinstance(loaded, dict):
                self.model = loaded.eval()
                print("✅ Распознаватель OCR (INT8) успешно загружен (полная модель).")
                return
            state_dict = loaded
        except Exception as e:
            print(f"[WARN] Не удалось загрузить OCR напрямую: {e}")
            state_dict = None

        model_to_load = CRNN(num_classes).eval()
        example_inputs = (torch.randn(1, 1, OCR_IMG_HEIGHT, OCR_IMG_WIDTH),)

        loaded_ok = False
        for qscheme_weight in [torch.per_channel_symmetric, torch.per_tensor_affine]:
            try:
                if qscheme_weight == torch.per_channel_symmetric:
                    weight_obs = PerChannelMinMaxObserver.with_args(
                        dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
                else:
                    weight_obs = MinMaxObserver.with_args(
                        dtype=torch.qint8, qscheme=torch.per_tensor_affine)

                qconfig = QConfig(
                    activation=MinMaxObserver.with_args(
                        qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                    weight=weight_obs,
                )
                qconfig_mapping = QConfigMapping().set_global(qconfig)
                model_prepared = quantize_fx.prepare_fx(
                    model_to_load, qconfig_mapping, example_inputs)
                model_quantized = quantize_fx.convert_fx(model_prepared)

                if state_dict is not None:
                    model_quantized.load_state_dict(state_dict, strict=False, assign=True)

                self.model = model_quantized
                loaded_ok = True
                print(f"✅ Распознаватель OCR (INT8) загружен (qscheme={qscheme_weight}).")
                break
            except Exception as e:
                print(f"[WARN] OCR qscheme={qscheme_weight}: {e}")
                continue

        if not loaded_ok:
            print("[WARN] FX квантизация не удалась. Пробуем загрузить как float-модель...")
            try:
                float_model = CRNN(num_classes).eval()
                float_model.load_state_dict(
                    torch.load(model_path, map_location=self.device), strict=False)
                self.model = float_model
                print("✅ Распознаватель OCR загружен как float-модель (fallback).")
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить OCR никаким способом: {e}")

    @torch.no_grad()
    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        preprocessed = self.transform(plate_image).unsqueeze(0).to(self.device)
        preds = self.model(preprocessed)
        return self._decode_with_score(preds)

    @torch.no_grad()
    def predict(self, plate_image: np.ndarray) -> Tuple[str, str, np.ndarray, float]:
        text, score = self.recognize(plate_image)
        return text, to_cyrillic(text), plate_image, score

    def _decode_with_score(self, preds: torch.Tensor) -> Tuple[str, float]:
        """CTC-декодирование + геометрическое среднее вероятностей."""
        probs = preds.permute(1, 0, 2)[0]  # (T, C)
        idxs = probs.argmax(dim=1)
        max_log_probs = probs.gather(1, idxs.unsqueeze(1)).squeeze(1)

        decoded_seq = []
        chosen_probs = []
        last_char_idx = 0  # 0 = blank/CTC-разделитель

        for pos, char_idx in enumerate(idxs.tolist()):
            if char_idx == 0:
                last_char_idx = 0
                continue
            if char_idx != last_char_idx:
                decoded_seq.append(self.int_to_char.get(char_idx, ''))
                chosen_probs.append(max_log_probs[pos].item())
            last_char_idx = char_idx

        text = "".join(decoded_seq)
        if not chosen_probs:
            return text, 0.0

        mean_log_prob = sum(chosen_probs) / len(chosen_probs)
        score = float(torch.exp(torch.tensor(mean_log_prob)).item())
        score = max(0.0, min(1.0, score))
        return text, score


class Visualizer:
    @staticmethod
    def draw_results(frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            text = res.get('text', '')
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame


class ANPR_Pipeline:
    """Главный пайплайн: детекция → OCR → валидация → голосование по треку."""

    def __init__(self, recognizer: CRNNRecognizer):
        self.recognizer = recognizer
        self.track_history = {}
        self.TRACK_BUFFER_SIZE = 15
        self.MIN_VOTES_TO_SHOW = 5
        self.MIN_YOLO_CONF = 0.65
        self._MAX_TRACK_HISTORY = 500

    @staticmethod
    def preprocess_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
        """CLAHE + лёгкое размытие — для двухпроходного OCR."""
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop
        img = plate_crop
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = img if img.ndim == 2 else img[..., 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def recognize_two_pass(self, plate_crop: np.ndarray) -> Dict[str, Any]:
        """Два прохода OCR (сырой + CLAHE). Возвращает оба результата и победителя.

        winner: 'tie' (результаты совпали после нормализации) / 'raw' / 'proc'.
        Используется в app.py для сомнительных случаев (needs_review).
        """
        raw_text, raw_score = self.recognizer.recognize(plate_crop)
        raw_cyr = to_cyrillic(raw_text)
        proc_crop = self.preprocess_plate_crop(plate_crop)
        proc_text, proc_score = self.recognizer.recognize(proc_crop)
        proc_cyr = to_cyrillic(proc_text)

        norm_raw = normalize_plate(raw_text)
        norm_proc = normalize_plate(proc_text)

        if norm_raw and norm_raw == norm_proc:
            winner = "tie"
            best_text = norm_raw
            best_cyr = raw_cyr
            best_crop = plate_crop
        elif raw_score >= proc_score:
            winner = "raw"
            best_text = norm_raw
            best_cyr = raw_cyr
            best_crop = plate_crop
        else:
            winner = "proc"
            best_text = norm_proc
            best_cyr = proc_cyr
            best_crop = proc_crop

        return {
            "best_text": best_text,
            "best_cyr": best_cyr,
            "best_crop": best_crop,
            "winner": winner,
            "raw_text": raw_text,
            "raw_cyr": raw_cyr,
            "raw_crop": plate_crop,
            "raw_score": raw_score,
            "proc_text": proc_text,
            "proc_cyr": proc_cyr,
            "proc_crop": proc_crop,
            "proc_score": proc_score,
        }

    def _maybe_cleanup_track_history(self):
        """Ограничивает рост track_history при длинной работе."""
        if len(self.track_history) > self._MAX_TRACK_HISTORY:
            # dict сохраняет порядок вставки — удаляем самые старые половину
            n_drop = len(self.track_history) - self._MAX_TRACK_HISTORY // 2
            for k in list(self.track_history.keys())[:n_drop]:
                del self.track_history[k]

    def process_frame(self, frame: np.ndarray, detections: List) -> List:
        results = []
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            track_id = det.get('track_id', None)
            confidence = det['confidence']

            if confidence < 0.3:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            text, ocr_score = self.recognizer.recognize(plate_crop)
            plate_norm = normalize_plate(text)

            is_valid = is_valid_russian_plate(plate_norm)
            needs_review = not is_valid or confidence < self.MIN_YOLO_CONF

            if not needs_review:
                if track_id is not None:
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append(text)
                    if len(self.track_history[track_id]) > self.TRACK_BUFFER_SIZE:
                        self.track_history[track_id].pop(0)

                    counter = Counter(self.track_history[track_id])
                    most_common = counter.most_common(1)
                    if most_common:
                        voted_text, votes = most_common[0]
                        if votes >= self.MIN_VOTES_TO_SHOW and len(voted_text) >= 6:
                            text = voted_text
                            plate_norm = normalize_plate(text)
                        else:
                            continue
                else:
                    if len(text) < 6:
                        continue
            else:
                if track_id is not None:
                    review_key = f"review_{track_id}"
                    if review_key not in self.track_history:
                        self.track_history[review_key] = {
                            "last_text": "", "count": 0, "shown": False,
                        }

                    entry = self.track_history[review_key]
                    if entry["last_text"] != text:
                        entry["last_text"] = text
                        entry["count"] = 0
                        entry["shown"] = False

                    entry["count"] += 1

                    if entry["shown"]:
                        continue
                    if entry["count"] < 3:
                        continue
                    entry["shown"] = True

            results.append({
                "bbox": bbox,
                "confidence": confidence,
                "text": text,
                "plate_norm": plate_norm,
                "track_id": track_id,
                "is_valid": is_valid,
                "needs_review": needs_review,
                "ocr_score": ocr_score,
            })

        self._maybe_cleanup_track_history()
        return results


if __name__ == '__main__':
    print("Этот файл импортируется через app.py. Запускайте: python main.py")