import numpy as np
import cv2
import os

from PIL import Image
from pathlib import Path

def calculate_img_size(x_move, y_move):
    '''
    x: 左加右减
    y: 上加下减
    '''
    # print(x_move,y_move)
    x, x_min, x_max = 0, 0, 0
    y, y_min, y_max = 0, 0, 0
    for i in range(len(x_move)):
        x += x_move[i]
        if x_min > x:
            x_min = x
        if x_max < x:
            x_max = x
    # print(x_min, x_max)
    for i in range(len(y_move)):
        y += y_move[i]
        if y_min > y:
            y_min = y
        if y_max < y:
            y_max = y
    # print(y_min, y_max)

    sum_x = abs(x_max) + abs(x_min)
    sum_y = abs(y_max) + abs(y_min)

    if x_min< 0 and x_max > 0:
        x = sum_x - abs(x_min)
    else:
        x = x_max

    if y_min< 0 and y_max > 0:
        y = sum_y - abs(y_min)
    else:
        y = y_max
    # print(x, y)
    return int(sum_x), int(sum_y), int(x), int(y)


def create_img(img_size, sum_x, sum_y):
    """
    img_size：图像原尺寸
    sum_x：图像x轴增长尺寸
    sum_y：图像y轴增长尺寸
    """
    bg_img = np.zeros((img_size[1] + sum_y, img_size[0] + sum_x, 3), dtype=np.uint8)  # 注意此处x，y是相反的,矩阵由(列x行)表示

    return bg_img


def combine_first_img(img, bg_img, x, y):
    """
    img：待拼接图像
    bg_img：背景图像（全黑色）
    注意！！！bg_img是由行x列组成（x,y），左上角为坐标原点。
    """
    bg_img = Image.fromarray(np.uint8(bg_img))
    img = Image.fromarray(np.uint8(img))
    bg_img.paste(img, (x, y))
    bg_img = np.asarray(bg_img)  # 得到含有初始图像的背景图像
    center_point = [x, y]

    return center_point, bg_img


def combine_image(input_dir, bg_img, count, x_move, y_move, center_point, img_to_stitch):
    if not os.path.exists(input_dir + '/result/'):
        os.makedirs(input_dir + '/result/')

    img1 = Image.fromarray(np.uint8(img_to_stitch))
    bg_img = Image.fromarray(np.uint8(bg_img))
    center_point[0] += -int(x_move[0])
    center_point[1] += -int(y_move[0])
    bg_img.paste(img1, (center_point[0], center_point[1]))
    bg_img = np.asarray(bg_img)  # 得到拼接图像

    cv2.imwrite(input_dir + '/result/result0.jpg', bg_img)

    return count + 1, bg_img

def combine(img_first_stitching, img_to_stitch, x_move, y_move, input_dir, count):
    sum_x, sum_y, x, y = calculate_img_size(x_move, y_move)
    bg_img = create_img([img_first_stitching.shape[1], img_first_stitching.shape[0]], sum_x, sum_y)
    center_point, bg_img = combine_first_img(img_first_stitching, bg_img, x, y)
    count, result_img = combine_image(input_dir, bg_img, count, x_move, y_move, center_point, img_to_stitch)
    return count, result_img