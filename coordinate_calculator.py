class CoordinateCalculator:
    def __init__(self):
        """
        Инициализирует калькулятор. Хранит точку калибровки (нейтральное положение головы).
        """
        self.calibrated_nose_point = None

    def calibrate(self, current_nose_point):
        """
        Устанавливает текущее положение носа как "центр" (нулевую координату).
        """
        self.calibrated_nose_point = current_nose_point
        print(f"Система откалибрована! Новый центр: {self.calibrated_nose_point}")

    def process(self, points):
        """
        Рассчитывает все необходимые метрики.
        :param points: Словарь с сырыми точками
        :return: Словарь с готовыми вычисленными координатами
        """
        if not points:
            return None

        nose = points['nose']
        left_eye = points['left_eye']
        right_eye = points['right_eye']

        # 1. Отклонение головы от центра
        head_deviation = (0, 0)
        if self.calibrated_nose_point:
            dx = nose[0] - self.calibrated_nose_point[0]
            dy = nose[1] - self.calibrated_nose_point[1]
            head_deviation = (dx, dy)

        # 2. Вычисление открытия глаз
        left_eye_openness = points['left_contour'][3][1] - points['left_contour'][2][1]
        right_eye_openness = points['right_contour'][3][1] - points['right_contour'][2][1]

        # 3. Вычисляем ширину глаз
        left_eye_width = abs(points['left_contour'][1][0] - points['left_contour'][0][0])
        right_eye_width = abs(points['right_contour'][1][0] - points['right_contour'][0][0])

        # 4. ИДЕАЛЬНЫЙ РАСЧЕТ: Жесткие якоря (без math)
        # --- Левый глаз ---
        l_ext, l_int = points['left_contour'][0], points['left_contour'][1]
        l_socket_center_x = (l_ext[0] + l_int[0]) / 2.0
        l_socket_center_y = (l_ext[1] + l_int[1]) / 2.0
        rel_left_eye = (left_eye[0] - l_socket_center_x, left_eye[1] - l_socket_center_y)

        # --- Правый глаз ---
        r_ext, r_int = points['right_contour'][0], points['right_contour'][1]
        r_socket_center_x = (r_ext[0] + r_int[0]) / 2.0
        r_socket_center_y = (r_ext[1] + r_int[1]) / 2.0
        rel_right_eye = (right_eye[0] - r_socket_center_x, right_eye[1] - r_socket_center_y)

        # 5. НОРМАЛИЗАЦИЯ (В процентах от ширины глаза)
        l_width_safe = left_eye_width if left_eye_width > 0 else 1
        r_width_safe = right_eye_width if right_eye_width > 0 else 1

        l_norm_x = rel_left_eye[0] / l_width_safe
        l_norm_y = rel_left_eye[1] / l_width_safe

        r_norm_x = rel_right_eye[0] / r_width_safe
        r_norm_y = rel_right_eye[1] / r_width_safe

        return {
            'head_absolute': nose,
            'head_deviation': head_deviation,
            'left_eye_absolute': left_eye,
            'right_eye_absolute': right_eye,
            'left_eye_relative': rel_left_eye,
            'right_eye_relative': rel_right_eye,
            'left_eye_normalized': (l_norm_x, l_norm_y),
            'right_eye_normalized': (r_norm_x, r_norm_y),
            'left_eye_openness': left_eye_openness,
            'right_eye_openness': right_eye_openness,
            'left_eye_width': left_eye_width,
            'right_eye_width': right_eye_width,
            'left_contour': points['left_contour'],
            'right_contour': points['right_contour'],
            'left_iris_edges': points['left_iris_edges'],
            'right_iris_edges': points['right_iris_edges']
        }