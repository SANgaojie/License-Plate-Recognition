"""
图像处理工具模块
提供基本的图像处理功能
"""

import cv2
import numpy as np
from modules.utils.exceptions import ImageLoadError
from modules.utils.logger import logger

def imreadex(filename):
    """读取图片文件，支持中文路径
    
    Args:
        filename: 图片文件路径
        
    Returns:
        numpy.ndarray: 读取的图像
        
    Raises:
        ImageLoadError: 读取图像失败时抛出
    """
    try:
        # 用以下方法可以解决路径中有汉字的问题
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ImageLoadError(f"无法读取图像: {filename}")
        return img
    except Exception as e:
        logger.error(f"读取图像失败: {str(e)}")
        raise ImageLoadError(f"读取图像失败: {str(e)}")

def point_limit(point):
    """限制点的坐标不小于0
    
    Args:
        point: 点坐标 [x, y]
        
    Returns:
        list: 限制后的点坐标
    """
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0
    return point

def deskew(img):
    """根据图片中心矩对图片进行校正
    
    Args:
        img: 输入图像
        
    Returns:
        numpy.ndarray: 校正后的图像
    """
    m = cv2.moments(img)  # 计算图像中的中心矩(最高到三阶)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * 20 * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def image_resize(img, target_width=None, target_height=None):
    """调整图像大小
    
    Args:
        img: 输入图像
        target_width: 目标宽度
        target_height: 目标高度
        
    Returns:
        numpy.ndarray: 调整大小后的图像
    """
    height, width = img.shape[:2]
    
    # 如果目标宽度和高度都为None，直接返回原图
    if target_width is None and target_height is None:
        return img
    
    # 如果只有一个目标维度，按比例计算另一个维度
    if target_width is None:
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)
    elif target_height is None:
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
    
    # 调整图像大小
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_img 