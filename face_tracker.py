import cv2
import mediapipe as mp
import urllib.request
import os


class FaceTracker:
    def __init__(self):
        """
        Инициализирует современную модель MediaPipe Tasks API.
        """
        model_path = 'face_landmarker.task'

        if not os.path.exists(model_path):
            print("Скачиваю модель face_landmarker.task (около 2.6 МБ)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Модель успешно скачана!")

        # Для Tasks API параметры
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    # def get_landmarks(self, frame):
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    #
    #     result = self.landmarker.detect(mp_image)
    #
    #     if not result.face_landmarks:
    #         return None
    #
    #     landmarks = result.face_landmarks[0]
    #     h, w, _ = frame.shape
    #
    #     nose_point = (int(landmarks[1].x * w), int(landmarks[1].y * h))
    #     left_eye = (int(landmarks[468].x * w), int(landmarks[468].y * h))
    #     right_eye = (int(landmarks[473].x * w), int(landmarks[473].y * h))
    #
    #     return {
    #         'nose': nose_point,
    #         'left_eye': left_eye,
    #         'right_eye': right_eye
    #     }

    def get_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        def get_pt(index):
            return (int(landmarks[index].x * w), int(landmarks[index].y * h))

        nose_point = get_pt(1)
        left_pupil = get_pt(468)
        right_pupil = get_pt(473)

        left_eye_contour = [get_pt(33), get_pt(133), get_pt(159), get_pt(145)]

        right_eye_contour = [get_pt(362), get_pt(263), get_pt(386), get_pt(374)]

        left_iris_edges = [get_pt(469), get_pt(470), get_pt(471), get_pt(472)]

        right_iris_edges = [get_pt(474), get_pt(475), get_pt(476), get_pt(477)]

        return {
            'nose': nose_point,
            'left_eye': left_pupil,
            'right_eye': right_pupil,
            'left_contour': left_eye_contour,
            'right_contour': right_eye_contour,
            'left_iris_edges': left_iris_edges,
            'right_iris_edges': right_iris_edges
        }


# Блок для независимого тестирования модуля
if __name__ == "__main__":
    from camera_handler import CameraHandler

    print("Запуск теста модуля FaceTracker (Tasks API)...")
    camera = CameraHandler(0)
    tracker = FaceTracker()

    while True:
        success, frame = camera.get_frame()
        if not success:
            break

        points = tracker.get_landmarks(frame)

        if points:
            cv2.circle(frame, points['nose'], 5, (0, 255, 0), -1)
            cv2.circle(frame, points['left_eye'], 3, (0, 0, 255), -1)
            cv2.circle(frame, points['right_eye'], 3, (255, 0, 0), -1)

        cv2.imshow('Face Tracker Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()