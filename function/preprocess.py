import base64
from io import BytesIO
from utils import encode_numpy2base64, decode_base642numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def mean(base_str, mean_bgr):
    """
    对图像进行归一化
    :param base_str:base64编码的图片
    :param mean_bgr: 归一化参数
    :return: png图片
    """
    # 读取图像和归一化参数
    image = decode_base642numpy(base_str).astype(np.uint8)
    mean_bgr = np.array(mean_bgr).astype(np.uint8)

    # 归一化图像
    mean_bgr = mean_bgr[0:3]
    image -= mean_bgr

    # 返回png图像
    img_str = encode_numpy2base64(image)

    return img_str


def resize(base_str, image_size):
    """
    对图片进行resize操作
    :param base_str: base64图片
    :param image_size: 输入DexiNed的图像大小
    :return: png图片
    """
    # 读取图像
    image = decode_base642numpy(base_str)

    # 读取原始图片的大小
    original_size = image.shape

    # resize图片
    resized_image = cv2.resize(image, (image_size, image_size))

    # 返回png图片
    img_str = encode_numpy2base64(resized_image)

    return img_str, original_size


def crop(base_str, position):
    """
    对图片按照指定区域进行裁剪
    :param base_str: base64编码的图像
    :param position: 裁剪位置的坐标
    :return:png图片
    """
    # 读取图片
    image = decode_base642numpy(base_str)
    pts = np.array(position)

    # 裁剪图片
    max1 = int(np.amax(pts, axis=0)[0])
    max2 = int(np.amax(pts, axis=0)[1])
    min1 = int(np.amin(pts, axis=0)[0])
    min2 = int(np.amin(pts, axis=0)[1])

    cropped = image[min2:max2, min1:max1]

    # 返回png图片
    img_str = encode_numpy2base64(cropped)

    return img_str


def blur_image(base_str):
    """
    将灰度图进行直方图均衡化和高斯模糊
    :param base_str: base64图像
    :return:png图像
    """
    # 读取图像
    image = decode_base642numpy(base_str)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 直方图均衡化和高斯模糊
    equ_image = cv2.equalizeHist(image)
    blurred_image = cv2.GaussianBlur(equ_image, (5, 5), 0)

    # 返回png图像
    img_str = encode_numpy2base64(blurred_image)

    return img_str


def gray_Image(base_str):
    """
    对裁剪后的图片进行灰度变换
    :param base_str: base64图像
    :return: png图像
    """
    # 读取图像
    image = decode_base642numpy(base_str)

    # 灰度变换
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 返回png图像
    img_str = encode_numpy2base64(gray_image)

    return img_str


def remap(base_str, mapx_path, mapy_path):
    """
    对图像进行正畸变换
    :param base_str: base64图像
    :param mapx_path: 正畸矩阵x路径
    :param mapy_path: 正畸矩阵y路径
    :return: png图像
    """
    # 读取图片和正畸矩阵
    image = decode_base642numpy(base_str)
    mapx = np.load(mapx_path)
    mapy = np.load(mapy_path)
    old_height, old_width = mapx.shape

    # 获取待矫正图像的长宽，然后对remap矩阵进行缩放
    new_height, new_width, channels = image.shape
    # 计算宽度和高度的缩放比例
    scale_width = new_width / old_width
    scale_height = new_height / old_height
    # 缩放mapx和mapy
    # 注，np.indices返回的是按(height, width)顺序的索引数组
    new_y_indices, new_x_indices = np.indices((new_height, new_width))
    mapx_scaled = cv2.resize(mapx, None, fx=scale_width, fy=scale_height, interpolation=cv2.INTER_LINEAR)
    mapy_scaled = cv2.resize(mapy, None, fx=scale_width, fy=scale_height, interpolation=cv2.INTER_LINEAR)
    mapx_scaled = mapx_scaled * scale_width
    mapy_scaled = mapy_scaled * scale_height

    image_remap = cv2.remap(image, mapx_scaled, mapy_scaled, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 返回png图片
    img_str = encode_numpy2base64(image_remap)

    return img_str


def perspective_shift(base_str, position):
    """
    对图片进行透视变换
    :param base_str: base64图片
    :param position: 透视变换的坐标
    :return: 返回png图片
    """
    # 读取图片和坐标
    image = decode_base642numpy(base_str)
    pts = np.array(position,dtype=np.float32)

    # 对伸缩缝区域做裁剪和透视变换
    width = max(int(np.linalg.norm(pts[0] - pts[1])), int(np.linalg.norm(pts[2] - pts[3])))
    height = max(int(np.linalg.norm(pts[0] - pts[3])), int(np.linalg.norm(pts[1] - pts[2])))
    pts2 = np.array([[0, 0], [width, 0], [width, height], [0, height]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, pts2)
    perspective_image = cv2.warpPerspective(image, M, (width, height))

    # 返回png图片
    img_str = encode_numpy2base64(perspective_image)

    return img_str


def hsv(base_str):
    """
    将图像转换为HSV空间
    :param base_str:base64图像
    :return:返回png图片
    """
    # 读取图片
    image = decode_base642numpy(base_str)

    # 转换HSV空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 返回png图片
    img_str = encode_numpy2base64(hsv_image)

    return img_str
