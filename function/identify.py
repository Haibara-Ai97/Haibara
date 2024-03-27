import os.path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import torch
from model import DexiNed
from utils import decode_base642numpy, encode_numpy2base64, decode_base642numpy_gray
# from preprocess_crack import perspective_shift, crop, HSV_Image


def filterlines(lines, original, angle_down_threshold, angle_up_threshold, length_threshold):
    """
    对得到的直线进行过滤
    针对角度和长度不符合水面线的直线进行过滤
    :param lines:边缘线
    :param original:原始图像base64编码

    :param angle_down_threshold:最小角度
    :param angle_up_threshold:最大角度
    :param length_threshold:最小线段长度
    :return:png图像
    """
    # 读取图像和直线数据
    original_image = decode_base642numpy(original)
    lines = np.array(lines)

    # 过滤直线
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if angle_down_threshold < abs(angle) < angle_up_threshold and length > length_threshold:
                filtered_lines.append(line)

    # 在原始图像上画出直线
    image_with_lines = np.copy(original_image)
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # 返回png图像和直线数据
    img_str = encode_numpy2base64(image_with_lines)
    lines_list = [arr.tolist() for arr in filtered_lines]

    return img_str, lines_list


def houghlines(edges, original, threshold, minLineLength, maxLineGap):
    """
    对检测到的边缘使用霍夫直线变换
    霍夫直线变换将相近的边缘线连接成直线
    :param edges:边缘点
    :param original:原始图像base64编码
    :param threshold:霍夫变换的阈值
    :param minLineLength:最小线段长度
    :param maxLineGap:最大间隔距离
    :return:png图像
    """
    # 读取图像
    image = decode_base642numpy_gray(edges)
    original_image = decode_base642numpy(original)

    # Hough直线变换
    lines = cv2.HoughLinesP(image, 1, np.pi / 360, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # 在原始图像上画出直线
    image_with_lines = np.copy(original_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1,y1), (x2,y2), (255,0,0), 3)

    # 返回png图像
    img_str = encode_numpy2base64(image_with_lines)
    lines_list = lines.tolist()

    return img_str, lines_list


def DexiNed_edge(base_str, img_shape, threshold):
    """
    使用DexiNed算法进行边缘检测
    :param base_str: base64图像
    :param img_shape: 输入DexiNed模型的图像大小
    :param threshold: 归纳边缘时的阈值
    :return: png图像
    """
    # 读取图像
    image = decode_base642numpy(base_str)

    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

    # 加载模型
    model = DexiNed().to(device)
    checkpoint = 'ckpt/10_model.pth'
    mean_bgr = [103.939, 116.779, 123.68]
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"There is no ckpt:{checkpoint}")
    print(f"Restoring weights from:{checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # 预处理图像
    image = np.array(image, dtype=np.float32)
    image -= mean_bgr
    image = image.transpose((2,0,1))
    image = torch.from_numpy(image.copy()).float()
    image = image.unsqueeze(0)
    image = image.to(device)

    # 推理
    with torch.no_grad():
        preds = model(image)
        edge_map = []
        for i in preds:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            edge_map.append(tmp)
        preds = np.array(edge_map)

        tmp = preds[:, 0,...]
        tmp = np.squeeze(tmp)

        results = []
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            img_min, img_max, epsilon = 0, 255, 1e-12
            tmp_img = np.float32(tmp_img)
            tmp_img = (tmp_img - np.min(tmp_img)) * (img_max - img_min) / \
                  ((np.max(tmp_img) - np.min(tmp_img)) + epsilon) + img_min
            tmp_img = np.uint8(tmp_img)
            if not tmp_img.shape[1] == img_shape[0] or not tmp_img.shape[0] == img_shape[1]:
                tmp_img = cv2.resize(tmp_img, (img_shape[1], img_shape[0]))
            results.append(tmp_img)
            if i == 6:
                fuse = tmp_img
                fuse[fuse < threshold] = 0
                fuse[fuse >= threshold] = 255
                fuse = fuse.astype(np.uint8)

        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))
        print(average.dtype)

    # 返回png图像
    img_str = encode_numpy2base64(fuse)

    return img_str


def canny(base_str, canny_threshold1, canny_threshold2):
    """
    对预处理过的图片进行边缘检测计算
    边缘检测使用DexiNed模型 也可以用opencv提供的Canny算法
    :param base_str:base64图像
    :param canny_threshold1:确定边缘点的低阈值
    :param canny_threshold2:确定边缘点的高阈值
    :return:边缘点数据
    """
    # 读取图像
    image = decode_base642numpy(base_str)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # canny边缘检测
    canny_edges = cv2.Canny(image, canny_threshold1, canny_threshold2)

    # 返回png图像
    img_str = encode_numpy2base64(canny_edges)

    return img_str


def countcenters(base_str, contours):
    """
    计算两个轮廓的中心点坐标
    :param base_str: base64图像
    :param contours: 轮廓数据
    :return: png图像和中心点坐标
    """
    # 读取图像
    contoured_image = decode_base642numpy(base_str)

    # 将contours列表转换为NumPy数组
    contours = [np.array(contour, dtype=np.int32).reshape((-1, 1, 2)) for contour in contours]

    # 计算两个轮廓的中心点
    boxes = [cv2.minAreaRect(c) for c in contours]
    centers = [box[0] for box in boxes]

    for center in centers:
        center_int = (int(center[0]), int(center[1]))
        cv2.circle(contoured_image, center_int, 10,(255, 0, 0), -1)

    # 返回png图片和中心点坐标
    img_str = encode_numpy2base64(contoured_image)
    centers_list = [list(center) for center in centers]

    return img_str, centers_list


def bluemask(base_str, low, up):
    """
    为蓝色标识板区域创建掩码

    :param base_str: HSV图像
    :param low: HSV通道上蓝色的最低值
    :param up: HSV通道上蓝色的最高值
    :return: 蓝色掩码
    """
    # 读取图像和蓝色通道值
    image = decode_base642numpy(base_str)
    lower_blue = np.array(low)
    upper_blue = np.array(up)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 获得蓝色掩码
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # 返回png图像
    img_str = encode_numpy2base64(mask)

    return img_str


def findcontour(original, base_str):
    """
    寻找蓝色区域的边界
    :param original:原始图像base64编码
    :param base_str:蓝色掩码base64编码
    :return:边缘图像和边缘点数据
    """
    # 读取原始图像和蓝色掩码
    original_image = decode_base642numpy(original)
    mask = decode_base642numpy_gray(base_str)

    # 计算边缘点
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # 将边缘点画在原始图像上
    cv2.drawContours(original_image, contours, -1, (0,0,255), 6)

    # 返回png图像和边缘点数据
    img_str = encode_numpy2base64(original_image)
    contours_list = [[[int(point[0][0]), int(point[0][1])] for point in contour] for contour in contours]

    return img_str, contours_list

