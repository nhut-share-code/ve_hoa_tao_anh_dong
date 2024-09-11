import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import math
import os
import colorsys
import cv2
from scipy.ndimage.filters import gaussian_filter

canvas_width = 600
canvas_height = 600
world_width = 0.05
world_height = 0.05

# Tham số cho trái tim giữa
points = None
fixed_point_size = 20000
fixed_scale_range = (4, 4.3)
min_scale = np.array([1.0, 1.0, 1.0]) * 0.9
max_scale = np.array([1.0, 1.0, 1.0]) * 0.9
min_heart_scale = -15
max_heart_scale = 16

# Tham số cho trái tim ngẫu nhiên
random_point_size = 7000
random_scale_range = (3.5, 3.9)
random_point_maxvar = 0.2

# Tham số thuật toán trái tim
mid_point_ignore = 0.95

# Tham số camera
camera_close_plane = 0.1
camera_position = np.array([0.0, -2.0, 0.0])

# Màu sắc của điểm
hue = 0.55  # Đổi sang màu xanh dương (0.55)
color_strength = 255

# Bộ đệm để render
render_buffer = np.empty((canvas_width, canvas_height, 3), dtype=int)
strength_buffer = np.empty((canvas_width, canvas_height), dtype=float)

# Tệp tin điểm ngẫu nhiên
points_file = "temp.txt"

# Số khung hình kết quả
total_frames = 30
output_dir = "./output"

# Định dạng ảnh
image_fmt = "jpg"

# Hàm chuyển đổi màu
def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    string = '#'
    for i in value:
        a1 = i // 16
        a2 = i % 16
        string += digit[a1] + digit[a2]
    return string

# Hàm tính toán hình trái tim
def heart_func(x, y, z, scale):
    bscale = scale
    bscale_half = bscale / 2
    x = x * bscale - bscale_half
    y = y * bscale - bscale_half
    z = z * bscale - bscale_half
    return (x**2 + 9/4*(y**2) + z**2 - 1)**3 - (x**2)*(z**3) - 9/200*(y**2)*(z**3)

# Hàm nội suy tuyến tính vector
def lerp_vector(a, b, ratio):
    result = a.copy()
    for i in range(3):
        result[i] = a[i] + (b[i] - a[i]) * ratio
    return result

# Hàm nội suy tuyến tính giá trị nguyên
def lerp_int(a, b, ratio):
    return int(a + (b - a) * ratio)

# Hàm nội suy tuyến tính giá trị thực
def lerp_float(a, b, ratio):
    return a + (b - a) * ratio

# Hàm tính khoảng cách giữa các điểm
def distance(point):
    return (point[0]**2 + point[1]**2 + point[2]**2) ** 0.5

# Hàm tính điểm ngẫu nhiên bên trong trái tim
def inside_rand(tense):
    x = random.random()
    y = -tense * math.log(x)
    return y

# Tạo điểm cho trái tim giữa
def genPoints(pointCount, heartScales):
    result = np.empty((pointCount, 3))
    index = 0
    while index < pointCount:
        x = random.random()
        y = random.random()
        z = random.random()

        mheartValue = heart_func(x, 0.5, z, heartScales[1])
        mid_ignore = random.random()
        if mheartValue < 0 and mid_ignore < mid_point_ignore:
            continue
        
        heartValue = heart_func(x, y, z, heartScales[0])
        z_shrink = 0.01
        sz = z - z_shrink
        sheartValue = heart_func(x, y, sz, heartScales[1])

        if heartValue < 0 and sheartValue > 0:
            result[index] = [x - 0.5, y - 0.5, z - 0.5]

            len = 0.7
            result[index] = result[index] * (1 - len * inside_rand(0.2))

            newY = random.random() - 0.5
            rheartValue = heart_func(result[index][0] + 0.5, newY + 0.5, result[index][2] + 0.5, heartScales[0])
            if rheartValue > 0:
                continue
            result[index][1] = newY

            dist = distance(result[index])
            if dist < 0.12:
                continue
            
            index = index + 1
            if index % 100 == 0:
                print(f"{index} generated {index / pointCount * 100}%")

    return result

# Tạo điểm cho trái tim ngẫu nhiên
def genRandPoints(pointCount, heartScales, maxVar, ratio):
    result = np.empty((pointCount, 3))
    index = 0
    while index < pointCount:
        x = random.random()
        y = random.random()
        z = random.random()
        mheartValue = heart_func(x, 0.5, z, heartScales[1])
        mid_ignore = random.random()
        if mheartValue < 0 and mid_ignore < mid_point_ignore:
            continue

        heartValue = heart_func(x, y, z, heartScales[0])
        sheartValue = heart_func(x, y, z, heartScales[1])

        if heartValue < 0 and sheartValue > 0:
            result[index] = [x - 0.5, y - 0.5, z - 0.5]
            dist = distance(result[index])
            if dist < 0.12:
                continue

            len = 0.7
            result[index] = result[index] * (1 - len * inside_rand(0.2))
            index = index + 1

    for i in range(pointCount):
        var = maxVar * ratio
        randScale = 1 + random.normalvariate(0, var)
        result[i] = result[i] * randScale

    return result

# Chuyển đổi tọa độ từ không gian thế giới sang không gian cục bộ của camera
def world_2_cameraLocalSpace(world_point):
    new_point = world_point.copy()
    new_point[1] = new_point[1] + camera_position[1]
    return new_point

# Chuyển đổi từ không gian cục bộ của camera sang không gian camera
def cameraLocal_2_cameraSpace(cameraLocalPoint):
    depth = distance(cameraLocalPoint)
    cx = cameraLocalPoint[0] * (camera_close_plane / cameraLocalPoint[1])
    cz = -cameraLocalPoint[2] * (cx / cameraLocalPoint[0])
    cameraLocalPoint[0] = cx
    cameraLocalPoint[1] = cz
    return cameraLocalPoint, depth

# Chuyển đổi từ không gian camera sang tọa độ màn hình
def camerSpace_2_screenSpace(cameraSpace):
    x = cameraSpace[0]
    y = cameraSpace[1]

    centerx = canvas_width / 2
    centery = canvas_height / 2
    ratiox = canvas_width / world_width
    ratioy = canvas_height / world_height

    viewx = centerx + x * ratiox
    viewy = canvas_height - (centery + y * ratioy)

    cameraSpace[0] = viewx
    cameraSpace[1] = viewy
    return cameraSpace.astype(int)

# Vẽ điểm lên màn hình
def draw_point(worldPoint):
    cameraLocal = world_2_cameraLocalSpace(worldPoint)
    cameraSpsace, depth = cameraLocal_2_cameraSpace(cameraLocal)
    screeSpace = camerSpace_2_screenSpace(cameraSpsace)

    draw_size = int(random.random() * 3 + 1)
    draw_on_buffer(screeSpace, depth, draw_size)

# Vẽ điểm lên bộ đệm
def draw_on_buffer(screenPos, depth, draw_size):
    if draw_size == 0:
        return
    elif draw_size == 1:
        draw_point_on_buffer(screenPos[0], screenPos[1], color_strength, depth)
    elif draw_size == 2:
        draw_point_on_buffer(screenPos[0], screenPos[1], color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1] + 1, color_strength, depth)
    elif draw_size == 3:
        draw_point_on_buffer(screenPos[0], screenPos[1], color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1] + 1, color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1], color_strength, depth)
    elif draw_size == 4:
        draw_point_on_buffer(screenPos[0], screenPos[1], color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1], color_strength, depth)
        draw_point_on_buffer(screenPos[0], screenPos[1] + 1, color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1] + 1, color_strength, depth)

# Lấy màu dựa trên sức mạnh màu
def get_color(strength):
    result = None
    if strength >= 1:
        result = colorsys.hsv_to_rgb(hue, 2 - strength, 1)
    else:
        result = colorsys.hsv_to_rgb(hue, 1, strength)
    r = min(result[0] * 256, 255)
    g = min(result[1] * 256, 255)
    b = min(result[2] * 256, 255)
    return np.array((r, g, b), dtype=int)

# Vẽ điểm lên bộ đệm
def draw_point_on_buffer(x, y, color, depth):
    if x < 0 or x >= canvas_width or y < 0 or y >= canvas_height:
        return
    strength = float(color) / 255
    strength_buffer[x, y] = strength_buffer[x, y] + strength

# Vẽ hình ảnh từ bộ đệm
def draw_buffer_on_canvas(output = None):
    render_buffer.fill(0)
    for i in range(render_buffer.shape[0]):
        for j in range(render_buffer.shape[1]):
            render_buffer[i, j] = get_color(strength_buffer[i, j])
    im = Image.fromarray(np.uint8(render_buffer))
    im = im.rotate(-90)
    if output is None:
        plt.imshow(im)
        plt.show()
    else:
        im.save(output)

# Hàm chính để tạo ảnh trái tim
def paint_heart(ratio, randratio, outputFile = None):
    global strength_buffer
    global render_buffer
    global points

    strength_buffer.fill(0)

    for i in range(fixed_point_size):
        point = points[i] * lerp_vector(min_scale, max_scale, ratio)

        dist = distance(point)
        radius = 0.4
        sphere_scale = radius / dist
        point = point * lerp_float(0.9, sphere_scale, ratio * 0.3)

        draw_point(point)

    randPoints = genRandPoints(random_point_size, random_scale_range, random_point_maxvar, randratio)
    for i in range(random_point_size):
        draw_point(randPoints[i])

    for i in range(1):
        strength_buffer = gaussian_filter(strength_buffer, sigma=0.8)

    draw_buffer_on_canvas(outputFile)

# Hiển thị hình ảnh
def show_images():
    img = None
    for i in range(total_frames):
        save_name = "{name}.{fmt}".format(name=i, fmt=image_fmt)
        save_path = os.path.join(output_dir, save_name)
        img = cv2.imread(save_path, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("Img", img)
        cv2.waitKey(25)

# Tạo hình ảnh
def gen_images():
    global points

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    if not os.path.exists(points_file):
        print("Không tìm thấy tệp, tạo mới")
        points = genPoints(fixed_point_size, fixed_scale_range)
        np.savetxt(points_file, points)
    else:
        print("Đã có tệp điểm, bỏ qua tạo mới")
        points = np.loadtxt(points_file)

    for i in range(total_frames):
        print(f"Đang xử lý ảnh {i} ...")
        frame_ratio = float(i) / (total_frames - 1)
        frame_ratio = frame_ratio ** 2
        ratio = math.sin(frame_ratio * math.pi) * 0.743144
        randratio = math.sin(frame_ratio * math.pi * 2 + total_frames / 2)
        save_name = "{name}.{fmt}".format(name=i, fmt=image_fmt)
        save_path = os.path.join(output_dir, save_name)
        paint_heart(ratio, randratio, save_path)
        print(f"Ảnh đã lưu tại {save_path}")

# Chạy chương trình
if __name__ == "__main__":
    gen_images()
    while True:
        show_images()
