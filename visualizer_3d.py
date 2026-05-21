from ursina import *
import socket
import json
import math

# 1. ИНИЦИАЛИЗАЦИЯ ДВИЖКА
app = Ursina(title='Eye Tracking FOV Simulation', size=(1000, 700))
window.color = color.rgb(20, 20, 25)

# 2. НАСТРОЙКА СЕТИ
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# 3. ПОСТРОЕНИЕ СЦЕНЫ
Z_DIST = 15

# Голова
head = Entity(model='sphere', color=color.cyan, scale=1.5, position=(0, 0, 0), unlit=True)

# Экран
screen = Entity(
    model='cube',
    color=color.azure,
    scale=(40, 30, 0.1),
    position=(0, 0, Z_DIST),
    transparent=True,
    alpha=0.3,
    double_sided=True,
    unlit=True
)

# Лазерная точка фокуса
gaze_dot = Entity(model='sphere', color=color.red, scale=0.6, position=(0, 0, Z_DIST), unlit=True)

# 4. КОНУС И ЛАЗЕРНЫЙ ЛУЧ
cone_pivot = Entity(position=(0, 0, 0))

# Центральный лазерный луч
laser_beam = Entity(
    parent=cone_pivot,
    model='cube',
    color=color.red,
    scale=(0.05, 0.05, 1),
    transparent=True,
    alpha=0.8,
    unlit=True
)

# Полупрозрачный желтый конус
fov_cone = Entity(
    parent=cone_pivot,
    model=Cone(resolution=32),
    color=color.yellow,
    transparent=True,
    alpha=0.25,
    double_sided=True,
    unlit=True
)

fov_cone.rotation_x = -90

EditorCamera()


# 5. ГЛАВНЫЙ ЦИКЛ 3D
def update():
    try:
        data, _ = sock.recvfrom(1024)
        message = json.loads(data.decode('utf-8'))

        target_x = -message.get('x', 0.0) * 40
        target_y = -message.get('y', 0.0) * 40

        gaze_dot.x = lerp(gaze_dot.x, target_x, time.dt * 15)
        gaze_dot.y = lerp(gaze_dot.y, target_y, time.dt * 15)

        cone_pivot.look_at(gaze_dot)

        dist = distance(head.position, gaze_dot.position)

        radius = dist * math.tan(math.radians(25))

        fov_cone.scale = (radius * 2, dist, radius * 2)
        fov_cone.z = dist / 2

        laser_beam.scale_z = dist
        laser_beam.z = dist / 2

    except BlockingIOError:
        pass
    except Exception as e:
        pass


app.run()