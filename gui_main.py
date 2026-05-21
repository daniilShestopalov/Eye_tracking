import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image
import time
import csv
from datetime import datetime
import numpy as np
import collections

# --- НОВЫЕ ИМПОРТЫ ДЛЯ 3D И СЕТИ ---
import subprocess
import socket
import json
import sys

# Импорты для графиков
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from camera_handler import CameraHandler
from face_tracker import FaceTracker
from coordinate_calculator import CoordinateCalculator

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class EyeTrackerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Eye Tracking Analytics Pro")
        self.geometry("1100x700")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.camera = None
        self.tracker = FaceTracker()
        self.calc = CoordinateCalculator()

        self.is_camera_on = False
        self.is_recording = False
        self.is_calibrated = False
        self.csv_file = None
        self.csv_writer = None
        self.save_path = ""

        # --- НАСТРОЙКИ СЕТИ ДЛЯ 3D ДВИЖКА ---
        self.process_3d = None
        self.udp_ip = "127.0.0.1"
        self.udp_port = 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # --- БАЗОВЫЕ НОРМАЛИЗОВАННЫЕ КООРДИНАТЫ ВЗГЛЯДА ---
        self.base_norm_x = 0.0
        self.base_norm_y = 0.0

        # --- НАСТРОЙКИ СГЛАЖИВАНИЯ (EMA FILTER) ---
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.alpha = 0.15  # Коэффициент доверия новым данным (0.15 - отличный баланс)

        # --- БУФЕРЫ ДЛЯ ГРАФИКОВ ---
        self.max_pts = 100
        self.time_data = collections.deque(maxlen=self.max_pts)
        self.h_x_data = collections.deque(maxlen=self.max_pts)
        self.h_y_data = collections.deque(maxlen=self.max_pts)

        self.eye_x_data = collections.deque(maxlen=self.max_pts)
        self.eye_y_data = collections.deque(maxlen=self.max_pts)

        self.frame_counter = 0

        # Предзаполняем нулями
        for _ in range(self.max_pts):
            self.time_data.append(0)
            self.h_x_data.append(0)
            self.h_y_data.append(0)
            self.eye_x_data.append(0)
            self.eye_y_data.append(0)

        # --- СОЗДАНИЕ ИНТЕРФЕЙСА ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="Tracker Control", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=(20, 30))

        self.btn_toggle_cam = ctk.CTkButton(self.sidebar, text="Включить камеру", command=self.toggle_camera)
        self.btn_toggle_cam.pack(pady=10, padx=20, fill="x")

        self.btn_calibrate = ctk.CTkButton(self.sidebar, text="Сброс и Калибровка", command=self.calibrate_system,
                                           state="disabled")
        self.btn_calibrate.pack(pady=10, padx=20, fill="x")

        # --- КНОПКА ЗАПУСКА 3D ---
        self.btn_launch_3d = ctk.CTkButton(self.sidebar, text="Запустить 3D Сцену", command=self.launch_3d_viewer,
                                           fg_color="#D97757", hover_color="#B85D3F")
        self.btn_launch_3d.pack(pady=(20, 10), padx=20, fill="x")

        self.lbl_path = ctk.CTkLabel(self.sidebar, text="Файл не выбран", font=ctk.CTkFont(size=11), text_color="gray",
                                     wraplength=200)
        self.lbl_path.pack(pady=(20, 0), padx=20)

        self.btn_choose_path = ctk.CTkButton(self.sidebar, text="Выбрать папку сохранения",
                                             command=self.choose_save_path, fg_color="transparent", border_width=1)
        self.btn_choose_path.pack(pady=(5, 20), padx=20, fill="x")

        self.btn_record = ctk.CTkButton(self.sidebar, text="▶ НАЧАТЬ ЗАПИСЬ", command=self.toggle_recording,
                                        state="disabled", fg_color="green", hover_color="darkgreen")
        self.btn_record.pack(pady=10, padx=20, fill="x")

        self.metrics_frame = ctk.CTkFrame(self.sidebar, corner_radius=5)
        self.metrics_frame.pack(pady=20, padx=20, fill="x")

        self.lbl_metrics_title = ctk.CTkLabel(self.metrics_frame, text="Текущие метрики:",
                                              font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_metrics_title.pack(pady=(10, 5))

        self.lbl_head_offset = ctk.CTkLabel(self.metrics_frame, text="Head Offset: X: 0 | Y: 0",
                                            font=ctk.CTkFont(size=12))
        self.lbl_head_offset.pack(pady=2)

        self.lbl_l_open = ctk.CTkLabel(self.metrics_frame, text="L Open: 0 px", font=ctk.CTkFont(size=12))
        self.lbl_l_open.pack(pady=2)

        self.lbl_r_open = ctk.CTkLabel(self.metrics_frame, text="R Open: 0 px", font=ctk.CTkFont(size=12))
        self.lbl_r_open.pack(pady=(2, 10))

        self.status_label = ctk.CTkLabel(self.sidebar, text="Статус: Ожидание...", font=ctk.CTkFont(size=14))
        self.status_label.pack(side="bottom", pady=20)

        self.right_container = ctk.CTkFrame(self, fg_color="transparent")
        self.right_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.tabview = ctk.CTkTabview(self.right_container)
        self.tabview.pack(fill="both", expand=True)

        self.tabview.add("Видеопоток")
        self.tabview.add("Графики аналитики")

        self.video_label = ctk.CTkLabel(self.tabview.tab("Видеопоток"), text="")
        self.video_label.pack(fill="both", expand=True)

        self.setup_matplotlib_graphs()
        self.pTime = 0

    # --- МЕТОД ЗАПУСКА 3D ---
    def launch_3d_viewer(self):
        """Запускает Ursina Engine как независимый фоновый процесс"""
        import sys  # Добавь импорт, если его нет в начале файла

        if self.process_3d is None or self.process_3d.poll() is not None:
            print("Запуск 3D-движка...")

            # --- ХИТРОСТЬ ДЛЯ PYINSTALLER ---
            if getattr(sys, 'frozen', False):
                # Если программа собрана в .exe, мы запускаем САМИ СЕБЯ с секретным флагом
                cmd = [sys.executable, "--run-3d"]
            else:
                # Если запускаем из исходников (PyCharm), просто вызываем файл скрипта
                cmd = [sys.executable, "visualizer_3d.py"]

            self.process_3d = subprocess.Popen(cmd)
            self.status_label.configure(text="Статус: 3D Сцена открыта", text_color="yellow")

    def setup_matplotlib_graphs(self):
        self.fig = Figure(figsize=(10, 4.5), dpi=100, facecolor='#2b2b2b')
        self.fig.subplots_adjust(left=0.06, right=0.96, bottom=0.15, top=0.88, wspace=0.25)

        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        axes = [self.ax1, self.ax2]
        titles = ["Смещение головы (пиксели)", "Направление взгляда (оба глаза)"]

        bounds = [150, 0.25]

        for i, ax in enumerate(axes):
            ax.set_facecolor('#1a1a1a')
            ax.set_title(titles[i], color='white', fontsize=12, weight="bold")
            ax.tick_params(colors='#888888', labelsize=10)

            limit = bounds[i]
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)

            # Отзеркаленные оси
            ax.invert_yaxis()
            ax.invert_xaxis()

            ax.axhline(0, color='#555555', linewidth=1.5, zorder=1)
            ax.axvline(0, color='#555555', linewidth=1.5, zorder=1)

            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.grid(True, color='#333333', linestyle='--', zorder=0)

        self.dot_head, = self.ax1.plot([0], [0], marker='o', color='red', markersize=12, linestyle='None', zorder=5)
        self.dot_eye, = self.ax2.plot([0], [0], marker='o', color='cyan', markersize=12, linestyle='None', zorder=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Графики аналитики"))
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

    def toggle_camera(self):
        if not self.is_camera_on:
            try:
                self.camera = CameraHandler(0)
                self.is_camera_on = True
                self.btn_toggle_cam.configure(text="Выключить камеру", fg_color="darkred", hover_color="red")
                self.btn_calibrate.configure(state="normal")
                self.status_label.configure(text="Статус: Камера работает")

                self.update_video_loop()
                self.update_graphs_loop()
            except Exception as e:
                print(f"Ошибка камеры: {e}")
        else:
            self.stop_system_safely()

    def calibrate_system(self):
        if self.is_recording:
            self.toggle_recording()

        success, frame = self.camera.get_frame()
        if success:
            raw_points = self.tracker.get_landmarks(frame)
            if raw_points:
                self.calc.calibrate(raw_points['nose'])
                self.is_calibrated = True

                results = self.calc.process(raw_points)
                l_norm = results['left_eye_normalized']
                r_norm = results['right_eye_normalized']

                # Запоминаем базовую нормализованную позицию взгляда
                self.base_norm_x = (l_norm[0] + r_norm[0]) / 2.0
                self.base_norm_y = (l_norm[1] + r_norm[1]) / 2.0

                self.status_label.configure(text="Статус: Откалибровано", text_color="cyan")

                # Сбрасываем фильтр сглаживания
                self.smooth_x = 0.0
                self.smooth_y = 0.0

                # Обнуляем массивы графиков
                for d in [self.h_x_data, self.h_y_data, self.eye_x_data, self.eye_y_data]:
                    for i in range(len(d)): d[i] = 0

                if self.save_path:
                    self.btn_record.configure(state="normal")
                else:
                    self.status_label.configure(text="Выберите папку для записи!", text_color="orange")
            else:
                self.status_label.configure(text="Лицо не найдено!", text_color="red")

    def choose_save_path(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"eye_tracking_{datetime.now().strftime('%H-%M-%S')}.csv",
            title="Сохранить лог как",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.save_path = filename
            display_path = "..." + filename[-25:] if len(filename) > 25 else filename
            self.lbl_path.configure(text=display_path)

            if self.is_calibrated:
                self.btn_record.configure(state="normal")

    def toggle_recording(self):
        if not self.is_recording:
            try:
                self.csv_file = open(self.save_path, mode='w', newline='', encoding='utf-8')
                self.csv_writer = csv.writer(self.csv_file, delimiter=',')
                self.csv_writer.writerow([
                    'Timestamp', 'Head_Dev_X', 'Head_Dev_Y',
                    'L_Pupil_Rel_X', 'L_Pupil_Rel_Y', 'R_Pupil_Rel_X', 'R_Pupil_Rel_Y',
                    'L_Eye_Openness_Px', 'R_Eye_Openness_Px', 'L_Eye_Width_Px', 'R_Eye_Width_Px'
                ])
                self.is_recording = True
                self.btn_record.configure(text="⏹ ОСТАНОВИТЬ ЗАПИСЬ", fg_color="darkred", hover_color="red")
                self.status_label.configure(text="Статус: ИДЕТ ЗАПИСЬ", text_color="red")
            except Exception as e:
                print(f"Ошибка создания файла: {e}")
        else:
            self.is_recording = False
            if self.csv_file:
                self.csv_file.close()
            self.btn_record.configure(text="▶ НАЧАТЬ ЗАПИСЬ", fg_color="green", hover_color="darkgreen")
            self.status_label.configure(text="Статус: Откалибровано", text_color="cyan")

    def update_video_loop(self):
        if not self.is_camera_on:
            return

        success, frame = self.camera.get_frame()
        if success:
            raw_points = self.tracker.get_landmarks(frame)

            if raw_points:
                results = self.calc.process(raw_points)

                l_center = results['left_eye_absolute']
                r_center = results['right_eye_absolute']

                l_edge = results['left_iris_edges'][0]
                l_radius = int(np.hypot(l_center[0] - l_edge[0], l_center[1] - l_edge[1]))
                r_edge = results['right_iris_edges'][0]
                r_radius = int(np.hypot(r_center[0] - r_edge[0], r_center[1] - r_edge[1]))

                cv2.circle(frame, l_center, l_radius, (255, 255, 0), 1)
                cv2.circle(frame, r_center, r_radius, (255, 255, 0), 1)

                def draw_smooth_ellipse(points):
                    ext, int_c, top, bot = points[0], points[1], points[2], points[3]
                    center = ((ext[0] + int_c[0]) // 2, (top[1] + bot[1]) // 2)
                    axis_x = abs(ext[0] - int_c[0]) // 2
                    axis_y = abs(bot[1] - top[1]) // 2
                    angle = np.degrees(np.arctan2(int_c[1] - ext[1], int_c[0] - ext[0]))
                    cv2.ellipse(frame, center, (axis_x, axis_y), angle, 0, 360, (0, 255, 0), 1)

                draw_smooth_ellipse(results['left_contour'])
                draw_smooth_ellipse(results['right_contour'])

                for pt in results['left_contour']:
                    cv2.line(frame, l_center, pt, (0, 255, 0), 1)
                for pt in results['right_contour']:
                    cv2.line(frame, r_center, pt, (0, 255, 0), 1)

                cv2.circle(frame, l_center, 2, (0, 0, 255), -1)
                cv2.circle(frame, r_center, 2, (0, 0, 255), -1)
                cv2.circle(frame, results['head_absolute'], 3, (0, 255, 0), -1)

                dev_x, dev_y = results['head_deviation']
                self.lbl_head_offset.configure(text=f"Head Offset: X: {dev_x} | Y: {dev_y}")
                self.lbl_l_open.configure(text=f"L Open: {results['left_eye_openness']} px")
                self.lbl_r_open.configure(text=f"R Open: {results['right_eye_openness']} px")

                # --- РАСЧЕТ И СГЛАЖИВАНИЕ ВЗГЛЯДА ---
                if self.is_calibrated:
                    self.frame_counter += 1

                    l_norm = results['left_eye_normalized']
                    r_norm = results['right_eye_normalized']

                    # Находим среднюю нормализованную точку
                    avg_norm_x = (l_norm[0] + r_norm[0]) / 2.0
                    avg_norm_y = (l_norm[1] + r_norm[1]) / 2.0

                    # Вычитаем базу
                    raw_dev_x = avg_norm_x - self.base_norm_x
                    raw_dev_y = avg_norm_y - self.base_norm_y

                    # ПРИМЕНЯЕМ ФИЛЬТР СГЛАЖИВАНИЯ (EMA)
                    self.smooth_x = (self.alpha * raw_dev_x) + ((1.0 - self.alpha) * self.smooth_x)
                    self.smooth_y = (self.alpha * raw_dev_y) + ((1.0 - self.alpha) * self.smooth_y)

                    self.time_data.append(self.frame_counter)
                    self.h_x_data.append(dev_x)
                    self.h_y_data.append(dev_y)

                    # Записываем в буфер графиков отфильтрованные данные
                    self.eye_x_data.append(self.smooth_x)
                    self.eye_y_data.append(self.smooth_y)

                    # --- ОТПРАВКА ДАННЫХ В 3D ДВИЖОК ---
                    # Складываем движение глаз и смещение головы (деленное на коэффициент, чтобы уравнять масштабы)
                    final_3d_x = self.smooth_x + (dev_x / 800.0)
                    final_3d_y = self.smooth_y + (dev_y / 800.0)

                    try:
                        msg = json.dumps({'x': final_3d_x, 'y': final_3d_y})
                        self.sock.sendto(msg.encode('utf-8'), (self.udp_ip, self.udp_port))
                    except Exception as e:
                        pass  # Если 3D движок выключен, просто игнорируем ошибку

                # CSV пишется с сырыми данными для максимальной точности
                if self.is_recording and self.is_calibrated:
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    l_pupil_x, l_pupil_y = results['left_eye_relative']
                    r_pupil_x, r_pupil_y = results['right_eye_relative']

                    self.csv_writer.writerow([
                        timestamp, dev_x, dev_y,
                        l_pupil_x, l_pupil_y, r_pupil_x, r_pupil_y,
                        results['left_eye_openness'], results['right_eye_openness'],
                        results['left_eye_width'], results['right_eye_width']
                    ])
            else:
                self.lbl_head_offset.configure(text="Head Offset: Нет лица")

            if self.is_recording:
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)

            target_w = self.tabview.tab("Видеопоток").winfo_width()
            target_h = self.tabview.tab("Видеопоток").winfo_height()

            if target_w > 10 and target_h > 10:
                frame_h, frame_w, _ = frame.shape
                scale = min(target_w / frame_w, target_h / frame_h)
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)

                resized_frame = cv2.resize(frame, (new_w, new_h))
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_w, new_h))
                self.video_label.configure(image=ctk_image)

        if self.is_camera_on:
            self.after(15, self.update_video_loop)

    def update_graphs_loop(self):
        if not self.is_camera_on:
            return

        if self.is_calibrated and len(self.h_x_data) > 0:
            cur_h_x = self.h_x_data[-1]
            cur_h_y = self.h_y_data[-1]

            cur_eye_x = self.eye_x_data[-1]
            cur_eye_y = self.eye_y_data[-1]

            self.dot_head.set_data([cur_h_x], [cur_h_y])
            self.dot_eye.set_data([cur_eye_x], [cur_eye_y])

            self.canvas.draw_idle()

        if self.is_camera_on:
            self.after(100, self.update_graphs_loop)

    def stop_system_safely(self):
        self.is_camera_on = False
        if self.is_recording:
            self.toggle_recording()

        if self.camera:
            self.camera.release()

        self.btn_toggle_cam.configure(text="Включить камеру", fg_color=["#3B8ED0", "#1F6AA5"],
                                      hover_color=["#36719F", "#144870"])
        self.btn_calibrate.configure(state="disabled")
        self.btn_record.configure(state="disabled")
        self.status_label.configure(text="Статус: Камера выключена", text_color="white")

        self.lbl_head_offset.configure(text="Head Offset: X: 0 | Y: 0")
        self.lbl_l_open.configure(text="L Open: 0 px")
        self.lbl_r_open.configure(text="R Open: 0 px")

        self.video_label.configure(image=None)
        self.is_calibrated = False
        self.base_norm_x = 0.0
        self.base_norm_y = 0.0
        self.smooth_x = 0.0
        self.smooth_y = 0.0

    def on_closing(self):
        print("Закрытие приложения... Сохранение данных.")
        self.stop_system_safely()
        # Аккуратно завершаем 3D процесс, если окно закрывают
        if self.process_3d is not None and self.process_3d.poll() is None:
            self.process_3d.terminate()
        self.destroy()


if __name__ == "__main__":
    import sys

    # Если процесс запущен с флагом, превращаем его в 3D-движок
    if "--run-3d" in sys.argv:
        import runpy

        # Запускаем спрятанный внутри сборки модуль visualizer_3d
        runpy.run_module('visualizer_3d', run_name='__main__')
    else:
        # Иначе запускаем обычный интерфейс
        app = EyeTrackerApp()
        app.mainloop()