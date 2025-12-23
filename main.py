import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from pylibdmtx import pylibdmtx
import numpy as np
import time
import sys
from queue import Empty


# ---------- поиск камер ----------
def list_cameras(max_devices: int = 4):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


# ---------- основное приложение ----------
class DataMatrixScanner:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DataMatrix Scanner")

        self.cap = None
        self.running = False

        # параметры логирования
        self.logged_params = False

        # учёт кодов
        self.seen_codes = set()
        self.code_counter = 0

        # трекинг рамок
        self.tracked = {}  # code -> {polygon, last_seen}
        self.TRACK_TIMEOUT = 0.5

        # ROI-трекинг
        self.active_roi = None
        self.roi_last_seen = 0
        self.ROI_TIMEOUT = 0.3
        self.frame_counter = 0

        # очереди для декодирования
        import queue
        self.decode_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()

        # worker поток
        import threading
        self.worker_thread = threading.Thread(target=self.decode_worker, daemon=True)
        self.worker_thread.start()

        # UI
        cams = list_cameras()
        if not cams:
            messagebox.showerror("Ошибка", "Камеры не найдены")
            root.destroy()
            return

        self.selected_camera = tk.IntVar(value=cams[0])

        ttk.Label(root, text="Выберите камеру:").pack(padx=10, pady=5)
        ttk.Combobox(
            root, values=cams, state="readonly",
            textvariable=self.selected_camera, width=10
        ).pack(padx=10, pady=5)

        ttk.Button(
            root, text="Открыть камеру",
            command=self.start
        ).pack(padx=10, pady=5)

        ttk.Button(
            root, text="Очистить скан",
            command=self.reset_scan
        ).pack(padx=10, pady=5)

        # горячая клавиша: C — очистить скан
        self.root.bind("<c>", lambda e: self.reset_scan())
        self.root.bind("<C>", lambda e: self.reset_scan())

    # ---------- звук ----------
    def beep(self):
        sys.stdout.write("\a")
        sys.stdout.flush()

    # ---------- очистка состояния ----------
    def reset_scan(self):
        self.seen_codes.clear()
        self.tracked.clear()
        self.code_counter = 0

        # очистка очередей decode
        try:
            while not self.decode_queue.empty():
                self.decode_queue.get_nowait()
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except Empty:
            pass

        print("Скан очищен — можно сканировать заново")

    # ---------- старт ----------
    def start(self):
        idx = self.selected_camera.get()
        self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", f"Не удалось открыть камеру {idx}")
            return

        # стабильный профиль (C920 / iPhone)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self.logged_params = False

        self.root.withdraw()
        self.loop()

    # ---------- ROI поиск ----------
    def find_rois(self, gray):
        edges = cv2.Canny(gray, 80, 160)
        dilated = cv2.dilate(edges, np.ones((3, 3)), iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = gray.shape
        rois = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            if cw < 40 or ch < 40:
                continue
            if cw > w * 0.9 or ch > h * 0.9:
                continue

            rois.append((x, y, cw, ch))

        return rois

    # ---------- worker декодирования ----------
    def decode_worker(self):
        while True:
            try:
                gray = self.decode_queue.get()
                results = []

                rois = self.find_rois(gray)

                # --- декод всех ROI ---
                for (x, y, w, h) in rois:
                    roi = gray[y:y+h, x:x+w]
                    decoded = pylibdmtx.decode(roi, timeout=5)

                    for r in decoded:
                        results.append((r, (x, y)))

                # --- fallback: раз в 10 кадров сканируем весь кадр ---
                if len(results) < 2:
                    decoded_full = pylibdmtx.decode(gray, timeout=10)
                    for r in decoded_full:
                        results.append((r, (0, 0)))

                if results:
                    self.result_queue.put(results)
            except Exception:
                pass

    # ---------- основной цикл ----------
    def loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            if not self.logged_params:
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                print(f"Фактические параметры: {w}x{h} @ {fps} FPS")
                self.logged_params = True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.time()

            self.frame_counter += 1

            # передаём КОПИЮ кадра, старый всегда выбрасываем
            try:
                while not self.decode_queue.empty():
                    self.decode_queue.get_nowait()
                self.decode_queue.put_nowait(gray.copy())
            except Empty:
                pass

            # обработка результатов из worker
            try:
                results = self.result_queue.get_nowait()
            except Empty:
                results = []

            for res, (ox, oy) in results:
                code = res.data.decode("utf-8", errors="ignore")

                if hasattr(res, "polygon") and res.polygon:
                    poly = [(p.x + ox, p.y + oy) for p in res.polygon]
                else:
                    rx, ry, rw, rh = res.rect
                    poly = [
                        (rx + ox, ry + oy), (rx + rw + ox, ry + oy),
                        (rx + rw + ox, ry + rh + oy), (rx + ox, ry + rh + oy)
                    ]

                if code not in self.seen_codes:
                    self.seen_codes.add(code)
                    self.code_counter += 1
                    print(f"{self.code_counter}. {code}")
                    self.beep()

                self.tracked[code] = {
                    "polygon": poly,
                    "last_seen": now
                }

            # --- очистка ушедших ---
            expired = [
                c for c, d in self.tracked.items()
                if now - d["last_seen"] > self.TRACK_TIMEOUT
            ]
            for c in expired:
                del self.tracked[c]

            # --- отрисовка ---
            for d in self.tracked.values():
                pts = d["polygon"]
                for i in range(len(pts)):
                    cv2.line(
                        frame, pts[i], pts[(i+1) % len(pts)],
                        (0, 255, 0), 2
                    )

            cv2.imshow("DataMatrix Scanner (Q — выход)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stop()

    # ---------- стоп ----------
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# ---------- entry ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = DataMatrixScanner(root)
    root.mainloop()