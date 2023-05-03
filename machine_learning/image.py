import numpy as np
import math
import cv2

colors = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 0.5],
    [0, 0.5, 0],
    [0.5, 0, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5],
    [0.5, 0.5, 0],
    [0.5, 1, 0.5],
    [1, 0.5, 0.5],
    [0.5, 0.5, 1],
]


def draw_angled_rec(loc, img, color, thickness):
    y0, x0, width, height, angle = loc
    b = math.cos(angle) * 0.5
    a = math.sin(angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, color, thickness)
    cv2.line(img, pt1, pt2, color, thickness)
    cv2.line(img, pt2, pt3, color, thickness)
    cv2.line(img, pt3, pt0, color, thickness)

    rect = cv2.minAreaRect(np.array([pt0, pt1, pt2, pt3]))
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(img, [box], 0, color, -1)


def map_to_img(data_input):
    default_scale = 530 / 0.34

    data_input[0] = data_input[0] * default_scale
    data_input[1] = data_input[1] * default_scale + 320
    data_input[2:4] *= default_scale
    data_input[4] -= 1.57
    data_input[4] *= -1
    return data_input


def get_pos_angles(idx, data_type):
    objects_num = 4 + idx // 10000
    dataset = np.loadtxt('dataset/%s/num_%d.txt' % (data_type, objects_num))
    id_scaled = idx % 10000
    sample = dataset[id_scaled]
    pos_angle = np.zeros(45)
    j = 0
    for i in range(len(sample)):
        if i % 5 == 0 or (i-1) % 5 == 0 or (i-4) % 5 == 0:
            pos_angle[j] = sample[i]
            j += 1
    return pos_angle


def get_img(idx, data_type):
    """
    :param idx: can range from 0 to 119999
    :param data_type: 'input' or 'label'
    :return: np.array 480x640x3
    """
    objects_num = 4 + idx // 10000
    dataset = np.loadtxt('dataset/%s/num_%d.txt' % (data_type, objects_num))
    id_scaled = idx % 10000
    sample = dataset[id_scaled]
    bg = np.ones((480, 640, 3))
    thickness = 3
    objects_num = len(sample) // 5

    for i in range(objects_num):
        loc = sample[i * 5:i * 5 + 5]
        loc = map_to_img(loc)
        draw_angled_rec(loc, bg, colors[i % 15], thickness)
    return bg


if __name__ == "__main__":
    pos_angles = get_pos_angles(119999, 'label')
    cv2.imshow('Image', get_img(119999, 'label'))
    cv2.waitKey(0)
