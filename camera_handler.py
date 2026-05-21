import cv2

class CameraHandler:
    def __init__(self, camera_id=0):
        """
        Инициализирует подключение к камере.
        :param camera_id: 0 для встроенной веб-камеры, 1/2/3 для внешних.
        """
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Ошибка: Не удалось открыть камеру с ID {self.camera_id}. Проверьте подключение.")

    def get_frame(self):
        """
        Считывает один кадр с камеры.
        :return: (success (bool), frame (numpy.ndarray или None))
        """
        success, frame = self.cap.read()#формат BGR!!!!!!!!
        if not success:
            print("Предупреждение: Не удалось получить кадр с камеры.")
            return False, None

        return True, frame

    def release(self):
        """
        Освобождает ресурсы видеокамеры и закрывает все окна OpenCV.
        """
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    print("Запуск теста модуля CameraHandler...")
    try:
        camera = CameraHandler(camera_id=0)
        print("Камера успешно подключена! Нажми 'q' (на английской раскладке) для выхода.")

        while True:
            success, frame = camera.get_frame()
            if not success:
                break

            cv2.imshow('Camera Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(e)
    finally:
        if 'camera' in locals():
            camera.release()
        print("Камера выключена. Тест завершен.")