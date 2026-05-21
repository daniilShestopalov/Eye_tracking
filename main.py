import cv2
import time
import csv
from datetime import datetime
from camera_handler import CameraHandler
from face_tracker import FaceTracker
from coordinate_calculator import CoordinateCalculator


def main():
    print("Инициализация системы расширенного трекинга...")

    try:
        camera = CameraHandler(0)
        tracker = FaceTracker()
        calc = CoordinateCalculator()
    except Exception as e:
        print(f"Ошибка при инициализации: {e}")
        return

    # 1. Генерируем имя файла с меткой времени
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"extended_eye_tracking_{current_time_str}.csv"

    # 2. Подготовка CSV файла с расширенными заголовками
    csv_file = open(csv_filename, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file, delimiter=',')

    csv_writer.writerow([
        'Timestamp',
        'Head_Dev_X', 'Head_Dev_Y',
        'L_Pupil_Rel_X', 'L_Pupil_Rel_Y',
        'R_Pupil_Rel_X', 'R_Pupil_Rel_Y',
        'L_Eye_Openness_Px', 'R_Eye_Openness_Px',
        'L_Eye_Width_Px', 'R_Eye_Width_Px'
    ])

    print("\n" + "=" * 60)
    print("СИСТЕМА РАСШИРЕННОГО ТРЕКИНГА ГОТОВА")
    print(f"Лог записи: {csv_filename}")
    print("-" * 60)
    print("Инструкция:")
    print("1. Смотрите прямо, нажмите 'C' для калибровки и начала записи.")
    print("2. На экране отображаются: зрачки, контуры век и края радужки.")
    print("3. Нажмите 'Q' для выхода и сохранения данных.")
    print("=" * 60 + "\n")

    pTime = 0

    while True:
        success, frame = camera.get_frame()
        if not success:
            break

        # Получаем обновленные landmarks (включая контуры)
        raw_points = tracker.get_landmarks(frame)

        if raw_points:
            # Проводим расчеты через калькулятор
            results = calc.process(raw_points)

            # --- ВИЗУАЛИЗАЦИЯ ---
            # 1. Центры (Нос и зрачки)
            cv2.circle(frame, results['head_absolute'], 5, (0, 255, 0), -1)
            cv2.circle(frame, results['left_eye_absolute'], 3, (0, 0, 255), -1)
            cv2.circle(frame, results['right_eye_absolute'], 3, (255, 0, 0), -1)

            # 2. Контуры век (желтые точки)
            for pt in results['left_contour'] + results['right_contour']:
                cv2.circle(frame, pt, 1, (0, 255, 255), -1)

            # 3. Края радужки (голубые точки)
            for pt in results['left_iris_edges'] + results['right_iris_edges']:
                cv2.circle(frame, pt, 1, (255, 255, 0), -1)

            # --- ТЕКСТОВАЯ ИНФОРМАЦИЯ ---
            dev_x, dev_y = results['head_deviation']
            cv2.putText(frame, f"Head Offset: X:{dev_x} Y:{dev_y}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Степень открытия глаз
            cv2.putText(frame, f"L Open: {results['left_eye_openness']}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"R Open: {results['right_eye_openness']}px", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if not calc.calibrated_nose_point:
                cv2.putText(frame, "WAITING FOR CALIBRATION (Press 'C')", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # --- ЗАПИСЬ ДАННЫХ ---
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                l_pupil_x, l_pupil_y = results['left_eye_relative']
                r_pupil_x, r_pupil_y = results['right_eye_relative']

                csv_writer.writerow([
                    timestamp,
                    dev_x, dev_y,
                    l_pupil_x, l_pupil_y,
                    r_pupil_x, r_pupil_y,
                    results['left_eye_openness'], results['right_eye_openness'],
                    results['left_eye_width'], results['right_eye_width']
                ])

        # Расчет FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow('Advanced Eye Tracking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and raw_points:
            calc.calibrate(raw_points['nose'])
            print("Калибровка выполнена. Запись всех компонентов глаза запущена.")

    # Закрытие ресурсов
    csv_file.close()
    camera.release()
    print(f"\nФайл успешно сохранен: {csv_filename}")


if __name__ == "__main__":
    main()