import cv2
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pylibdmtx import pylibdmtx
import numpy as np
import time
import sys
from queue import Empty
from typing import Optional, Tuple, List
from datetime import datetime


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
        self.root.geometry("400x600")
        
        # Центрирование окна
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        # shared frame (thread-safe)
        self.latest_frame = None
        self.frame_lock = None

        # параметры логирования
        self.logged_params = False

        # учёт кодов
        self.seen_codes = set()
        self.code_counter = 0
        self.scanned_codes = []  # Список всех отсканированных кодов
        self.scan_start_time = None
        self.time_for_10_codes = None

        # трекинг рамок
        self.tracked = {}  # code -> {polygon, last_seen}
        self.TRACK_TIMEOUT = 0.5

        # ROI-трекинг
        self.active_roi = None
        self.roi_last_seen = 0
        self.ROI_TIMEOUT = 0.3
        self.frame_counter = 0
        # лёгкий цифровой зум
        self.zoom_factor = 1.1

        # Настройки камеры
        self.camera_settings = {
            'width': 3840,
            'height': 2160,
            'fps': 60,
            'fourcc': cv2.VideoWriter.fourcc(*"MJPG")
        }

        # Доступные разрешения
        self.resolutions = [
            (640, 480),
            (800, 600),
            (1024, 768),
            (1280, 720),
            (3840, 2160),
        ]

        # очереди для декодирования
        import queue
        self.decode_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()

        # потокобезопасный lock для кадров
        import threading
        self.frame_lock = threading.Lock()

        # worker поток
        self.worker_thread = threading.Thread(target=self.decode_worker, daemon=True)
        self.worker_thread.start()

        # UI переменные
        cams = list_cameras()
        if not cams:
            messagebox.showerror("Ошибка", "Камеры не найдены")
            root.destroy()
            return

        self.selected_camera = tk.IntVar(value=cams[0])
        self.selected_resolution = tk.StringVar(value="3840x2160")

        # Создание отдельных окон
        self.create_main_window()
        self.create_codes_window()
        self.create_settings_window()

        # горячая клавиша: C — очистить скан
        self.root.bind("<c>", lambda e: self.reset_scan())
        self.root.bind("<C>", lambda e: self.reset_scan())

    def create_main_window(self):
        """Создание главного окна управления"""
        # Стиль
        style = ttk.Style()
        style.configure("TButton", padding=10, font=('Arial', 10))
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        title_label = ttk.Label(main_frame, text="DataMatrix Scanner", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))

        # Выбор камеры
        camera_frame = ttk.LabelFrame(main_frame, text="Настройка камеры", padding="10")
        camera_frame.pack(fill=tk.X, pady=5)

        ttk.Label(camera_frame, text="Выберите камеру:").grid(row=0, column=0, sticky=tk.W, padx=5)
        camera_combo = ttk.Combobox(
            camera_frame, values=list_cameras(), state="readonly",
            textvariable=self.selected_camera, width=15
        )
        camera_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(camera_frame, text="Разрешение:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=(10,0))
        resolution_combo = ttk.Combobox(
            camera_frame, 
            values=[f"{w}x{h}" for w, h in self.resolutions],
            state="readonly",
            textvariable=self.selected_resolution, 
            width=15
        )
        resolution_combo.grid(row=1, column=1, padx=5, pady=(10,0))

        # Кнопки управления
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)

        self.start_button = ttk.Button(
            button_frame, text="▶ Запустить сканирование",
            command=self.start, style="TButton"
        )
        self.start_button.pack(fill=tk.X, pady=5)

        self.clear_button = ttk.Button(
            button_frame, text="🗑 Очистить все коды",
            command=self.reset_scan
        )
        self.clear_button.pack(fill=tk.X, pady=5)

        self.settings_button = ttk.Button(
            button_frame, text="⚙ Настройки камеры",
            command=self.show_settings_window
        )
        self.settings_button.pack(fill=tk.X, pady=5)

        self.codes_button = ttk.Button(
            button_frame, text="📋 Показать отсканированные коды",
            command=self.show_codes_window
        )
        self.codes_button.pack(fill=tk.X, pady=5)

        # Счетчик времени
        self.time_frame = ttk.LabelFrame(main_frame, text="Время сканирования", padding="10")
        self.time_frame.pack(fill=tk.X, pady=10)

        self.time_label = ttk.Label(self.time_frame, text="Время для 10 кодов: --", 
                                   font=('Arial', 10))
        self.time_label.pack()

        self.counter_label = ttk.Label(self.time_frame, text="Отсканировано кодов: 0", 
                                      font=('Arial', 10))
        self.counter_label.pack()

        # Статус
        self.status_label = ttk.Label(main_frame, text="Готов к работе", 
                                     foreground="green", font=('Arial', 9))
        self.status_label.pack(pady=10)

        # Горячие клавиши
        hotkeys_frame = ttk.LabelFrame(main_frame, text="Горячие клавиши", padding="5")
        hotkeys_frame.pack(fill=tk.X, pady=5)

        hotkeys_text = "Q - Выход из режима сканирования\nC - Очистить все коды"
        ttk.Label(hotkeys_frame, text=hotkeys_text, justify=tk.LEFT).pack()

    def create_codes_window(self):
        """Создание окна для отсканированных кодов"""
        self.codes_window = tk.Toplevel(self.root)
        self.codes_window.title("Отсканированные коды")
        self.codes_window.geometry("600x400")
        self.codes_window.withdraw()
        
        # Центрирование окна
        self.codes_window.update_idletasks()
        width = self.codes_window.winfo_width()
        height = self.codes_window.winfo_height()
        x = (self.codes_window.winfo_screenwidth() // 2) - (width // 2) + 200
        y = (self.codes_window.winfo_screenheight() // 2) - (height // 2)
        self.codes_window.geometry(f'{width}x{height}+{x}+{y}')

        # Основной фрейм
        main_frame = ttk.Frame(self.codes_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        ttk.Label(main_frame, text="Отсканированные коды", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 10))

        # Текстовое поле с прокруткой
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.codes_text = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, width=60, height=20,
            font=('Consolas', 10)
        )
        self.codes_text.pack(fill=tk.BOTH, expand=True)

        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Очистить список", 
                  command=self.clear_codes_list).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Копировать все", 
                  command=self.copy_all_codes).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Закрыть", 
                  command=self.codes_window.withdraw).pack(side=tk.RIGHT, padx=5)

        # При закрытии окна - скрываем его
        self.codes_window.protocol("WM_DELETE_WINDOW", self.codes_window.withdraw)

    def create_settings_window(self):
        """Создание окна настроек камеры"""
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Настройки камеры")
        self.settings_window.geometry("500x400")
        self.settings_window.withdraw()
        
        # Центрирование окна
        self.settings_window.update_idletasks()
        width = self.settings_window.winfo_width()
        height = self.settings_window.winfo_height()
        x = (self.settings_window.winfo_screenwidth() // 2) - (width // 2) - 200
        y = (self.settings_window.winfo_screenheight() // 2) - (height // 2)
        self.settings_window.geometry(f'{width}x{height}+{x}+{y}')

        # Основной фрейм
        main_frame = ttk.Frame(self.settings_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Настройки разрешения камеры", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 20))

        # Кнопки разрешений
        resolutions_frame = ttk.Frame(main_frame)
        resolutions_frame.pack(fill=tk.X, pady=10)

        for i, (width, height) in enumerate(self.resolutions):
            btn_text = f"{width}x{height}"
            btn = ttk.Button(
                resolutions_frame, 
                text=btn_text,
                command=lambda w=width, h=height: self.set_resolution(w, h)
            )
            btn.pack(fill=tk.X, pady=2, padx=20)

        # Текущие настройки
        info_frame = ttk.LabelFrame(main_frame, text="Текущие настройки", padding="10")
        info_frame.pack(fill=tk.X, pady=20)

        self.settings_info = ttk.Label(
            info_frame, 
            text=f"Разрешение: {self.camera_settings['width']}x{self.camera_settings['height']}\n"
                 f"FPS: {self.camera_settings['fps']}\n"
                 f"Кодек: MJPG",
            justify=tk.LEFT
        )
        self.settings_info.pack()

        # Кнопка обновления настроек
        ttk.Button(main_frame, text="Применить настройки", 
                  command=self.apply_camera_settings).pack(pady=10)

        ttk.Button(main_frame, text="Закрыть", 
                  command=self.settings_window.withdraw).pack(pady=5)

        # При закрытии окна - скрываем его
        self.settings_window.protocol("WM_DELETE_WINDOW", self.settings_window.withdraw)

    def show_codes_window(self):
        """Показать окно с кодами"""
        self.codes_window.deiconify()
        self.codes_window.lift()

    def show_settings_window(self):
        """Показать окно настроек"""
        self.settings_window.deiconify()
        self.settings_window.lift()

    def set_resolution(self, width: int, height: int):
        """Установить разрешение"""
        self.camera_settings['width'] = width
        self.camera_settings['height'] = height
        self.selected_resolution.set(f"{width}x{height}")
        
        # Обновляем информацию в окне настроек
        self.settings_info.config(
            text=f"Разрешение: {width}x{height}\n"
                 f"FPS: {self.camera_settings['fps']}\n"
                 f"Кодек: MJPG"
        )

    def apply_camera_settings(self):
        """Применить настройки камеры"""
        if self.running and self.cap:
            messagebox.showinfo("Информация", 
                              "Настройки будут применены при следующем запуске сканирования")
        else:
            messagebox.showinfo("Информация", "Настройки сохранены")

    def update_codes_display(self):
        """Обновить отображение кодов в окне"""
        self.codes_text.delete(1.0, tk.END)
        
        if not self.scanned_codes:
            self.codes_text.insert(tk.END, "Нет отсканированных кодов")
            return
        
        for i, code in enumerate(self.scanned_codes, 1):
            timestamp = code.get('timestamp', '')
            code_text = code.get('code', '')
            self.codes_text.insert(tk.END, f"{i:3d}. [{timestamp}] {code_text}\n")
        
        # Прокрутка вниз
        self.codes_text.see(tk.END)

    def update_time_display(self):
        """Обновить отображение времени"""
        if self.time_for_10_codes:
            minutes = int(self.time_for_10_codes // 60)
            seconds = self.time_for_10_codes % 60
            self.time_label.config(
                text=f"Время для 10 кодов: {minutes:02d}:{seconds:05.2f}"
            )
        else:
            self.time_label.config(text="Время для 10 кодов: --")
        
        self.counter_label.config(text=f"Отсканировано кодов: {self.code_counter}")

    def clear_codes_list(self):
        """Очистить список кодов в окне"""
        self.scanned_codes.clear()
        self.update_codes_display()
        self.reset_scan()

    def copy_all_codes(self):
        """Копировать все коды в буфер обмена"""
        if not self.scanned_codes:
            return
        
        codes_text = "\n".join([f"{i+1}. {item['code']}" 
                               for i, item in enumerate(self.scanned_codes)])
        self.root.clipboard_clear()
        self.root.clipboard_append(codes_text)
        messagebox.showinfo("Скопировано", "Все коды скопированы в буфер обмена")

    # ---------- звук ----------
    def beep(self):
        sys.stdout.write("\a")
        sys.stdout.flush()

    # ---------- очистка состояния ----------
    def reset_scan(self):
        self.seen_codes.clear()
        self.tracked.clear()
        self.code_counter = 0
        self.scan_start_time = None
        self.time_for_10_codes = None
        
        # очистка очередей decode
        try:
            while not self.decode_queue.empty():
                self.decode_queue.get_nowait()
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except Empty:
            pass

        # Обновить отображение
        self.update_codes_display()
        self.update_time_display()
        
        self.status_label.config(text="Сканирование очищено", foreground="orange")
        print("Скан очищен — можно сканировать заново")

    # ---------- старт ----------
    def start(self):
        idx = self.selected_camera.get()
        self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", f"Не удалось открыть камеру {idx}")
            return

        # Применение настроек камеры
        self.cap.set(cv2.CAP_PROP_FOURCC, self.camera_settings['fourcc'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_settings['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_settings['height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_settings['fps'])

        import threading
        self.running = True
        self.logged_params = False

        # Запуск потока захвата кадров
        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            daemon=True
        )
        self.capture_thread.start()

        # Обновить статус
        self.status_label.config(text="Сканирование запущено", foreground="green")
        
        # Скрыть основное окно и запустить цикл через after
        self.root.withdraw()
        self.root.after(0, self.loop)

    def capture_loop(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                continue

            with self.frame_lock:
                self.latest_frame = frame

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
        if not self.running:
            self.stop()
            return

        # Получить последний кадр из потока
        with self.frame_lock:
            if self.latest_frame is None:
                self.root.after(1, self.loop)
                return
            frame = self.latest_frame.copy()

        # ---- цифровой зум (мягкий) ----
        if self.zoom_factor > 1.0:
            h, w = frame.shape[:2]
            cw = int(w / self.zoom_factor)
            ch = int(h / self.zoom_factor)

            x1 = (w - cw) // 2
            y1 = (h - ch) // 2

            frame = frame[y1:y1 + ch, x1:x1 + cw]
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        if not self.logged_params and self.cap is not None:
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
                
                # Добавить в список отсканированных кодов
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.scanned_codes.append({
                    'code': code,
                    'timestamp': timestamp,
                    'number': self.code_counter
                })
                
                # Запустить таймер для первого кода
                if self.code_counter == 1:
                    self.scan_start_time = now
                
                # Зафиксировать время для 10 кодов
                if self.code_counter == 10 and self.scan_start_time:
                    self.time_for_10_codes = now - self.scan_start_time
                
                print(f"{self.code_counter}. {code} [{timestamp}]")
                self.beep()
                
                # Обновить отображение
                self.update_codes_display()
                self.update_time_display()

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
        
        # Добавить информацию на кадр
        cv2.putText(frame, f"Codes: {self.code_counter}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if self.time_for_10_codes:
            time_text = f"Time for 10: {self.time_for_10_codes:.2f}s"
            cv2.putText(frame, time_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("DataMatrix Scanner (Q — выход, C — очистка)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.running = False
            self.stop()
            return
        elif key == ord("c"):
            self.reset_scan()

        self.root.after(1, self.loop)

    # ---------- стоп ----------
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.deiconify()
        self.status_label.config(text="Сканирование остановлено", foreground="red")


# ---------- entry ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = DataMatrixScanner(root)
    root.mainloop()
    
    