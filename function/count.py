import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from utils import decode_base642numpy, encode_numpy2base64


def count_levelout(lines, original, level, map):
    """
    对识别到的边缘线计算与水尺所在直线的坐标 针对13号点位
    :param lines: 得到的边缘线
    :param original: 原始图片
    :param level: 计算交点参数
    :param map: 映射水位参数
    :return: 返回水位图像和水位数据
    """
    # 读取图像和参数
    original_image = decode_base642numpy(original)
    lines = np.array(lines)
    level = np.array(level)
    map = np.array(map)
    a, b, c = level[0], level[1], level[2]
    map_w, map_b = map[0], map[1]

    # 计算水位
    if len(lines) > 0:
        cross_point = count_cross_point(lines, a, b, c)  # 计算交点坐标
        cross_point = np.array(cross_point)
        water_level = cross_point[:, 1]
        water_level = np.max(water_level)
        water_level = (map_w * water_level) + map_b

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeft = (10, 50)
        fontScale = 1
        fontColor = (0, 0, 0)
        lineType = 2

        cv2.putText(
            original_image,
            f"Water Level: {level}",
            bottomLeft,
            font,
            fontScale,
            fontColor,
            lineType
        )
    else:
        water_level = -1

    # 返回png图像
    img_str = encode_numpy2base64(original_image)

    return img_str, water_level


def count_levelin(lines, original, level, map):
    """
    对识别到的边缘线计算与水尺所在直线的坐标 针对14 15号点位
    :param lines: 得到的边缘线
    :param original: 原始图片
    :param level: 计算交点参数
    :param map: 映射水位参数
    :return: 返回水位图像和水位数据
    """
    # 读取图像和参数
    original_image = decode_base642numpy(original)
    lines = np.array(lines)
    level = np.array(level)
    map = np.array(map)
    a, b, c = level[0], level[1], level[2]
    map_w, map_b = map[0], map[1]

    # 计算水位
    if len(lines) > 0:
        cross_point = count_cross_point(lines, a, b, c)  # 计算交点坐标
        cross_point = np.array(cross_point)
        water_level = cross_point[:, 1]
        water_level = np.min(water_level)
        water_level = (map_w * water_level) + map_b

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeft = (10, 50)
        fontScale = 1
        fontColor = (0, 0, 0)
        lineType = 2

        cv2.putText(
            original_image,
            f"Water Level: {level}",
            bottomLeft,
            font,
            fontScale,
            fontColor,
            lineType
        )
    else:
        water_level = -1

    # 返回png图像和参数
    img_str = encode_numpy2base64(original_image)

    return img_str, water_level


def count_cross_point(lines, level_a, level_b, level_c):
    """
    对识别到的边缘线计算与水尺所在直线的坐标
    :param lines: 得到的边缘线
    :param level_a: 水尺参数a
    :param level_b: 水池参数b
    :param level_c: 水池参数c
    :return: 交点坐标
    """
    cross_point = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # a, b, c = 15.599510603588907, 1, 3483.0489396411094
        # a, b, c = 414, 1, 95226
        a, b, c = level_a, level_b, level_c

        # Convert line1 to the form ax + by + c = 0
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1

        # Calculate the determinant
        det = A1 * b - a * B1

        # Lines intersect, calculate x and y
        x = (b * C1 - B1 * c) / det
        y = (A1 * c - a * C1) / det
        cross_point.append([x, y])
    return cross_point


def map_water_level(data, map_w, map_b):
    """
    坐标映射到真实水位
    :param data: 坐标
    :param map_w: 映射权重
    :param map_b: 映射偏置
    :return: 真实水位
    """
    return (map_w * data) + map_b


def countdistance(base_str, center, brick, weight):
    """
    计算两板之间的距离

    :param base_str: base64图像
    :param center: 两中心点坐标
    :param brick: 两板的像素长度
    :param weight: 计算像素与实际距离的权重
    :return: 返回识别结果图像和结构缝间距
    """
    # 读取图像和参数
    center_image = decode_base642numpy(base_str)
    # 计算间距
    distance_pixel = np.sqrt((center[0][0] - center[1][0]) ** 2 + (center[0][1] - center[1][1]) ** 2)
    distance_pixel = distance_pixel - brick
    distance = distance_pixel * weight

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeft = (10,50)
    fontScale = 1
    fontColor = (0,0,0)
    lineType = 2

    cv2.putText(
        center_image,
        f"Distance: {distance}",
        bottomLeft,
        font,
        fontScale,
        fontColor,
        lineType
    )

    # 返回png图像
    img_str = encode_numpy2base64(center_image)

    return img_str, distance

